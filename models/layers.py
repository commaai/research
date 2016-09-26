import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
from keras.engine import Layer, InputSpec
from keras import backend as K, regularizers, constraints, initializations, activations
from keras.layers.recurrent import Recurrent, time_distributed_dense


class Deconv2D(Layer):
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='tf',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(Deconv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, self.nb_filter, stack_size)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape, name='{}/w'.format(self.name))
        self.b = K.zeros((self.nb_filter,), name='{}/biases'.format(self.name))
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = rows * self.subsample[0]
        cols = cols * self.subsample[1]

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        output_shape = self.get_output_shape_for(x.get_shape().as_list())
        deconv_out = tf.nn.conv2d_transpose(
            x, self.W, output_shape=output_shape, strides=[1, self.subsample[0], self.subsample[1], 1])

        if self.dim_ordering == 'th':
            output = deconv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = deconv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Deconv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Bnorm2D(Layer):
  def __init__(self, epsilon=1e-5, momentum=0.9, weights=None, beta_init='zero',
               gamma_init='normal', **kwargs):
    self.beta_init = initializations.get(beta_init)
    self.gamma_init = initializations.get(gamma_init)
    self.epsilon = epsilon
    self.momentum = momentum
    self.initial_weights = weights
    # self.uses_learning_phase = True
    self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
    super(Bnorm2D, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_spec = [InputSpec(shape=input_shape)]
    shape = input_shape

    self.gamma = self.gamma_init(shape, name='{}/gamma'.format(self.name))
    self.beta = self.beta_init(shape, name='{}/beta'.format(self.name))
    self.trainable_weights = [self.gamma, self.beta]

    if self.initial_weights is not None:
        self.set_weights(self.initial_weights)
        del self.initial_weights
    self.built = True
    self.called_with = None

  def call(self, x, mask=None):
    # out = K.in_train_phase(self.train_bn(x), self.test_bn(x))
    out = self.train_bn(x)
    return out

  def train_bn(self, x):
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema_apply_op = self.ema.apply([batch_mean, batch_var])
    self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
    with tf.control_dependencies([ema_apply_op]):
      mean, var = tf.identity(batch_mean), tf.identity(batch_var)

    out = tf.nn.batch_norm_with_global_normalization(
        x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
    return out

  def test_bn(self, x):
    mean, var = self.ema_mean, self.ema_var
    out = tf.nn.batch_norm_with_global_normalization(
        x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
    return out


class DreamyRNN(Recurrent):
  '''Fully-connected RNN where the output is to be fed back to input.
  # Arguments
      output_dim: dimension of the internal projections and the final output.
      init: weight initialization function.
          Can be the name of an existing function (str),
          or a Theano function (see: [initializations](../initializations.md)).
      inner_init: initialization function of the inner cells.
      activation: activation function.
          Can be the name of an existing function (str),
          or a Theano function (see: [activations](../activations.md)).
      W_regularizer: instance of [WeightRegularizer](../regularizers.md)
          (eg. L1 or L2 regularization), applied to the input weights matrices.
      U_regularizer: instance of [WeightRegularizer](../regularizers.md)
          (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
      b_regularizer: instance of [WeightRegularizer](../regularizers.md),
          applied to the bias.
      dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
      dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
  # References
      - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
  '''
  def __init__(self, output_dim, output_length,
               init='glorot_uniform', inner_init='orthogonal',
               activation='tanh',
               W_regularizer=None, U_regularizer=None, b_regularizer=None,
               dropout_W=0., dropout_U=0., **kwargs):
      self.output_dim = output_dim
      self.output_length = output_length
      self.init = initializations.get(init)
      self.inner_init = initializations.get(inner_init)
      self.activation = activations.get(activation)
      self.W_regularizer = regularizers.get(W_regularizer)
      self.U_regularizer = regularizers.get(U_regularizer)
      self.b_regularizer = regularizers.get(b_regularizer)
      self.dropout_W, self.dropout_U = dropout_W, dropout_U

      if self.dropout_W or self.dropout_U:
          self.uses_learning_phase = True
      super(DreamyRNN, self).__init__(**kwargs)

  def build(self, input_shape):
      self.input_spec = [InputSpec(shape=input_shape)]
      if self.stateful:
          self.reset_states()
      else:
          # initial states: all-zero tensor of shape (output_dim)
          self.states = [None]
      input_dim = input_shape[2]
      self.input_dim = input_dim

      self.V = self.init((self.output_dim, input_dim),
                         name='{}_V'.format(self.name))
      self.W = self.init((input_dim, self.output_dim),
                         name='{}_W'.format(self.name))
      self.U = self.inner_init((self.output_dim, self.output_dim),
                               name='{}_U'.format(self.name))
      self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
      self.ext_b = K.zeros((input_dim,), name='{}_ext_b'.format(self.name))

      self.regularizers = []
      if self.W_regularizer:
          self.W_regularizer.set_param(self.W)
          self.regularizers.append(self.W_regularizer)
      if self.U_regularizer:
          self.U_regularizer.set_param(self.U)
          self.regularizers.append(self.U_regularizer)
      if self.b_regularizer:
          self.b_regularizer.set_param(self.b)
          self.regularizers.append(self.b_regularizer)

      self.trainable_weights = [self.W, self.U, self.b, self.V, self.ext_b]

      if self.initial_weights is not None:
          self.set_weights(self.initial_weights)
          del self.initial_weights

  def reset_states(self):
      assert self.stateful, 'Layer must be stateful.'
      input_shape = self.input_spec[0].shape
      if not input_shape[0]:
          raise Exception('If a RNN is stateful, a complete ' +
                          'input_shape must be provided (including batch size).')
      if hasattr(self, 'states'):
          K.set_value(self.states[0],
                      np.zeros((input_shape[0], self.output_dim)))
      else:
          self.states = [K.zeros((input_shape[0], self.output_dim))]

  def preprocess_input(self, x):
      if self.consume_less == 'cpu':
          input_shape = self.input_spec[0].shape
          input_dim = input_shape[2]
          timesteps = input_shape[1]
          return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                        input_dim, self.output_dim,
                                        timesteps)
      else:
          return x

  def step(self, x, states):
    prev_output = states[0]
    B_U = states[1]
    B_W = states[2]

    if self.consume_less == 'cpu':
        h = x
    else:
        h = K.dot(x * B_W, self.W) + self.b

    output = self.activation(h + K.dot(prev_output * B_U, self.U))
    return output, [output]

  def dream(self, x, states):
    prev_st = states[0]
    prev_x = tf.stop_gradient(K.dot(prev_st, self.V) + self.ext_b)
    B_U = states[1]
    B_W = states[2]
    h = K.dot(prev_x * B_W, self.W) + self.b

    output = self.activation(h + K.dot(prev_st * B_U, self.U))
    return output, [output]

  def get_constants(self, x):
      constants = []
      if 0 < self.dropout_U < 1:
          ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
          ones = K.concatenate([ones] * self.output_dim, 1)
          B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
          constants.append(B_U)
      else:
          constants.append(K.cast_to_floatx(1.))
      if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
          input_shape = self.input_spec[0].shape
          input_dim = input_shape[-1]
          ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
          ones = K.concatenate([ones] * input_dim, 1)
          B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
          constants.append(B_W)
      else:
          constants.append(K.cast_to_floatx(1.))
      return constants

  def get_config(self):
      config = {'output_dim': self.output_dim,
                'init': self.init.__name__,
                'inner_init': self.inner_init.__name__,
                'activation': self.activation.__name__,
                'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                'dropout_W': self.dropout_W,
                'dropout_U': self.dropout_U}
      base_config = super(DreamyRNN, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

  def call(self, x, mask=None):
    # input shape: (nb_samples, time (padded with zeros), input_dim)
    # note that the .build() method of subclasses MUST define
    # self.input_spec with a complete input shape.
    input_shape = self.input_spec[0].shape
    if K._BACKEND == 'tensorflow':
        if not input_shape[1]:
            raise Exception('When using TensorFlow, you should define '
                            'explicitly the number of timesteps of '
                            'your sequences.\n'
                            'If your first layer is an Embedding, '
                            'make sure to pass it an "input_length" '
                            'argument. Otherwise, make sure '
                            'the first layer has '
                            'an "input_shape" or "batch_input_shape" '
                            'argument, including the time axis. '
                            'Found input shape at layer ' + self.name +
                            ': ' + str(input_shape))
    if self.stateful:
        initial_states = self.states
    else:
        initial_states = self.get_initial_states(x)
    constants = self.get_constants(x)
    preprocessed_input = self.preprocess_input(x)

    last_output, outputs_0, states = K.rnn(self.step, preprocessed_input,
                                           initial_states,
                                           go_backwards=self.go_backwards,
                                           mask=mask,
                                           constants=constants,
                                           unroll=self.unroll,
                                           input_length=input_shape[1])
    timer = K.zeros((2, self.output_length, 2))
    last_output, outputs, states = K.rnn(self.dream, timer,
                                         states, go_backwards=self.go_backwards,
                                         mask=mask,
                                         constants=constants,
                                         input_length=self.output_length,
                                         unroll=self.unroll)

    last_output = K.dot(last_output, self.V) + self.ext_b
    outputs = K.concatenate([outputs_0, outputs], axis=1)
    outputs = K.dot(K.reshape(outputs, (-1, self.output_dim)), self.V) + self.ext_b

    ishape = K.shape(x)
    if K._BACKEND == "tensorflow":
      ishape = x.get_shape().as_list()
    outputs = K.reshape(outputs, (-1, ishape[1]+self.output_length, ishape[2]))

    if self.stateful:
      self.updates = []
      for i in range(len(states)):
        self.updates.append((self.states[i], states[i]))

    if self.return_sequences:
      return outputs
    else:
      return last_output

  def get_output_shape_for(self, input_shape):
    if self.return_sequences:
      return (input_shape[0], input_shape[1]+self.output_length, input_shape[2])
    else:
      return (input_shape[0], input_shape[2])


class CondDreamyRNN(Recurrent):
  def __init__(self, output_dim, output_length, control_dim=2,
               init='glorot_uniform', inner_init='orthogonal',
               activation='tanh',
               W_regularizer=None, U_regularizer=None, b_regularizer=None,
               dropout_W=0., dropout_U=0., **kwargs):
      self.output_dim = output_dim
      self.output_length = output_length
      self.init = initializations.get(init)
      self.inner_init = initializations.get(inner_init)
      self.activation = activations.get(activation)
      self.W_regularizer = regularizers.get(W_regularizer)
      self.U_regularizer = regularizers.get(U_regularizer)
      self.b_regularizer = regularizers.get(b_regularizer)
      self.dropout_W, self.dropout_U = dropout_W, dropout_U
      self.control_dim = control_dim

      if self.dropout_W or self.dropout_U:
          self.uses_learning_phase = True
      super(CondDreamyRNN, self).__init__(**kwargs)

  def build(self, input_shape):
      self.input_spec = [InputSpec(shape=input_shape)]
      if self.stateful:
          self.reset_states()
      else:
          # initial states: all-zero tensor of shape (output_dim)
          self.states = [None]
      input_dim = input_shape[2]
      self.input_dim = input_dim

      self.V = self.init((self.output_dim, input_dim-self.control_dim),
                         name='{}_V'.format(self.name))
      self.W = self.init((input_dim, self.output_dim),
                         name='{}_W'.format(self.name))
      self.U = self.inner_init((self.output_dim, self.output_dim),
                               name='{}_U'.format(self.name))
      self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
      self.ext_b = K.zeros((input_dim-self.control_dim,), name='{}_ext_b'.format(self.name))

      self.regularizers = []
      if self.W_regularizer:
          self.W_regularizer.set_param(self.W)
          self.regularizers.append(self.W_regularizer)
      if self.U_regularizer:
          self.U_regularizer.set_param(self.U)
          self.regularizers.append(self.U_regularizer)
      if self.b_regularizer:
          self.b_regularizer.set_param(self.b)
          self.regularizers.append(self.b_regularizer)

      self.trainable_weights = [self.W, self.U, self.b, self.V, self.ext_b]

      if self.initial_weights is not None:
          self.set_weights(self.initial_weights)
          del self.initial_weights

  def reset_states(self):
      assert self.stateful, 'Layer must be stateful.'
      input_shape = self.input_spec[0].shape
      if not input_shape[0]:
          raise Exception('If a RNN is stateful, a complete ' +
                          'input_shape must be provided (including batch size).')
      if hasattr(self, 'states'):
          K.set_value(self.states[0],
                      np.zeros((input_shape[0], self.output_dim)))
      else:
          self.states = [K.zeros((input_shape[0], self.output_dim))]

  def preprocess_input(self, x):
      if self.consume_less == 'cpu':
          input_shape = self.input_spec[0].shape
          input_dim = input_shape[2]
          timesteps = input_shape[1]
          return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                        input_dim, self.output_dim,
                                        timesteps)
      else:
          return x

  def step(self, x, states):
    prev_output = states[0]
    B_U = states[1]
    B_W = states[2]

    if self.consume_less == 'cpu':
        h = x
    else:
        h = K.dot(x * B_W, self.W) + self.b

    output = self.activation(h + K.dot(prev_output * B_U, self.U))
    return output, [output]

  def dream(self, x, states):
    prev_st = states[0]
    controls = x[:, :self.control_dim]
    prev_x = K.concatenate([controls, tf.stop_gradient(K.dot(prev_st, self.V) + self.ext_b)], axis=1)
    B_U = states[1]
    B_W = states[2]
    h = K.dot(prev_x * B_W, self.W) + self.b

    output = self.activation(h + K.dot(prev_st * B_U, self.U))
    return output, [output]

  def get_constants(self, x):
      constants = []
      if 0 < self.dropout_U < 1:
          ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
          ones = K.concatenate([ones] * self.output_dim, 1)
          B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
          constants.append(B_U)
      else:
          constants.append(K.cast_to_floatx(1.))
      if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
          input_shape = self.input_spec[0].shape
          input_dim = input_shape[-1]
          ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
          ones = K.concatenate([ones] * input_dim, 1)
          B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
          constants.append(B_W)
      else:
          constants.append(K.cast_to_floatx(1.))
      return constants

  def get_config(self):
      config = {'output_dim': self.input_dim-self.control_dim,
                'init': self.init.__name__,
                'inner_init': self.inner_init.__name__,
                'activation': self.activation.__name__,
                'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                'dropout_W': self.dropout_W,
                'dropout_U': self.dropout_U}
      base_config = super(CondDreamyRNN, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

  def call(self, x, mask=None):
    # input shape: (nb_samples, time (padded with zeros), input_dim)
    # note that the .build() method of subclasses MUST define
    # self.input_spec with a complete input shape.
    input_shape = self.input_spec[0].shape
    if K._BACKEND == 'tensorflow':
        if not input_shape[1]:
            raise Exception('When using TensorFlow, you should define '
                            'explicitly the number of timesteps of '
                            'your sequences.\n'
                            'If your first layer is an Embedding, '
                            'make sure to pass it an "input_length" '
                            'argument. Otherwise, make sure '
                            'the first layer has '
                            'an "input_shape" or "batch_input_shape" '
                            'argument, including the time axis. '
                            'Found input shape at layer ' + self.name +
                            ': ' + str(input_shape))
    if self.stateful:
        initial_states = self.states
    else:
        initial_states = self.get_initial_states(x)
    constants = self.get_constants(x)
    preprocessed_input = self.preprocess_input(x)

    last_output, outputs_0, states = K.rnn(self.step, preprocessed_input[:, :input_shape[1]-self.output_length, :],
                                           initial_states,
                                           go_backwards=self.go_backwards,
                                           mask=mask,
                                           constants=constants,
                                           unroll=self.unroll,
                                           input_length=input_shape[1])
    last_output, outputs, states = K.rnn(self.dream, preprocessed_input[:, input_shape[1]-self.output_length:, :],
                                         states, go_backwards=self.go_backwards,
                                         mask=mask,
                                         constants=constants,
                                         input_length=self.output_length,
                                         unroll=self.unroll)

    last_output = K.dot(last_output, self.V) + self.ext_b
    outputs = K.concatenate([outputs_0, outputs], axis=1)
    outputs = K.dot(K.reshape(outputs, (-1, self.output_dim)), self.V) + self.ext_b

    ishape = K.shape(x)
    if K._BACKEND == "tensorflow":
      ishape = x.get_shape().as_list()
    outputs = K.reshape(outputs, (-1, input_shape[1], ishape[2]-self.control_dim))

    if self.stateful:
      self.updates = []
      for i in range(len(states)):
        self.updates.append((self.states[i], states[i]))

    if self.return_sequences:
      return outputs
    else:
      return last_output

  def get_output_shape_for(self, input_shape):
    if self.return_sequences:
      return (input_shape[0], input_shape[1], self.output_dim)
    else:
      return (input_shape[0], self.output_dim)
