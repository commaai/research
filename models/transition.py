import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import cv2
import tensorflow as tf
from layers import DreamyRNN
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential
from keras import initializations
import autoencoder  # this is our autoencoder with GAN cost function
from utils import load, save
from functools import partial
learning_rate = .0008
beta1 = .5
z_dim = 512  # dimensions of the lattent space where guassian samples are taken from
time = 5  # how long teacher forcing
out_leng = 10  # how long should the transition model go on its own after teacher forcing
normal = partial(initializations.normal, scale=.02)
G_file_path = "./outputs/results_autoencoder/G_weights.keras"
E_file_path = "./outputs/results_autoencoder/E_weights.keras"


def mean_normal(shape, mean=1., scale=0.02, name=None):
    return K.variable(np.random.normal(loc=mean, scale=scale, size=shape), name=name)


def cleanup(data):
  X = data[0]
  sh = X.shape
  X = X.reshape((-1, 3, 160, 320))
  X = np.asarray([cv2.resize(x.transpose(1, 2, 0), (160, 80)) for x in X])
  X = X/127.5 - 1.
  X = X.reshape((sh[0], (time+out_leng)*4, 80, 160, 3))
  Z = np.random.normal(0, 1, (X.shape[0], z_dim))
  return Z, X[:, ::4]


def transition(batch_size, dim=1000):
    model = Sequential()
    model.add(DreamyRNN(unroll=True, output_dim=z_dim, output_length=out_leng-1, return_sequences=True,
                        activation="tanh", batch_input_shape=(batch_size, time, z_dim)))
    return model


def get_model(sess, image_shape=(80, 160, 3), gf_dim=64, df_dim=64, batch_size=64,
              name="transition", gpu=0):
    K.set_session(sess)
    checkpoint_dir = './outputs/results_' + name
    with tf.variable_scope(name):
      # sizes
      ch = image_shape[2]
      rows = [image_shape[0]/i for i in [16, 8, 4, 2, 1]]
      cols = [image_shape[1]/i for i in [16, 8, 4, 2, 1]]

      G = autoencoder.generator(7*(time+out_leng-1), gf_dim, ch, rows, cols)
      G.compile("sgd", "mse")
      E = autoencoder.encoder(batch_size*(time+out_leng), df_dim, ch, rows, cols)
      E.compile("sgd", "mse")

      G.trainable = False
      E.trainable = False

      # nets
      T = transition(batch_size)
      T.compile("sgd", "mse")
      t_vars = T.trainable_weights
      print "T.shape: ", T.output_shape

      Img = Input(batch_shape=(batch_size, time+out_leng,) + image_shape)
      I = K.reshape(Img, (batch_size*(time+out_leng),)+image_shape)
      code = E(I)[0]
      code = K.reshape(code, (batch_size, time+out_leng, z_dim))
      target = code[:, 1:, :]
      inp = code[:, :time, :]
      out = T(inp)
      G_dec = G(K.reshape(out[:7, :, :], (-1, z_dim)))

      # costs
      loss = tf.reduce_mean(tf.square(target - out))
      print "Transition variables:"
      for v in t_vars:
        print v.name

      t_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=t_vars)

      tf.initialize_all_variables().run()

    # summaries
    sum_loss = tf.scalar_summary("loss", loss)
    sum_e_mean = tf.histogram_summary("e_mean", code)
    sum_out = tf.histogram_summary("out", out)
    sum_dec = tf.image_summary("E", G_dec)

    # saver
    saver = tf.train.Saver()
    t_sum = tf.merge_summary([sum_e_mean, sum_out, sum_dec, sum_loss])
    writer = tf.train.SummaryWriter("/tmp/logs/"+name, sess.graph)

    # functions
    def train_d(images, z, counter, sess=sess):
      return 0, 0, 0

    def train_g(images, z, counter, sess=sess):
      outputs = [loss, G_dec, t_sum, t_optim]
      outs = sess.run(outputs, feed_dict={Img: images, K.learning_phase(): 1})
      gl, samples, sums = outs[:3]
      writer.add_summary(sums, counter)
      images = images.reshape((-1, 80, 160, 3))[:64]
      samples = samples.reshape((-1, 80, 160, 3))[:64]
      return gl, samples, images

    def f_load():
      try:
        return load(sess, saver, checkpoint_dir, name)
      except:
        print("Loading weights via Keras")
        T.load_weights(checkpoint_dir+"/T_weights.keras")

    def f_save(step):
      save(sess, saver, checkpoint_dir, step, name)
      T.save_weights(checkpoint_dir+"/T_weights.keras", True)

    def sampler(z, x):
      video = np.zeros((128, 80, 160, 3))
      print "Sampling..."
      for i in range(128):
        print i
        x = x.reshape((-1, 80, 160, 3))
        # code = E.predict(x, batch_size=batch_size*(time+1))[0]
        code = sess.run([E(I)[0]], feed_dict={I: x, K.learning_phase(): 1})[0]
        code = code.reshape((batch_size, time+out_leng, z_dim))
        inp = code[:, :time]
        outs = T.predict(inp, batch_size=batch_size)
        # imgs = G.predict(out, batch_size=batch_size)
        imgs = sess.run([G_dec], feed_dict={out: outs, K.learning_phase(): 1})[0]
        video[i] = imgs[0]
        x = x.reshape((batch_size, time+out_leng, 80, 160, 3))
        x[0, :-1] = x[0, 1:]
        x[0, -1] = imgs[0]

      video = video.reshape((batch_size, 2, 80, 160, 3))
      return video[:, 0], video[:, 1]

    G.load_weights(G_file_path)
    E.load_weights(E_file_path)

    return train_g, train_d, sampler, f_save, f_load, [G, E, T]
