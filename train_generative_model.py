#!/usr/bin/env python

"""
Usage:
>> ./server.py
>> ./train_generator.py autoencoder
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import argparse
import time
from keras import callbacks as cbks
import logging
import tensorflow as tf
import numpy as np

from server import client_generator
from models.utils import save_images
mixtures = 1


def old_cleanup(data):
  X = data[0]
  if X.shape[1] == 1:
    X = X[:, -1, :]/127.5 - 1.
  return X


def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X = cleanup(tup)
    yield X


def train_model(name, g_train, d_train, sampler, generator, samples_per_epoch, nb_epoch,
                z_dim=100, verbose=1, callbacks=[],
                validation_data=None, nb_val_samples=None,
                saver=None):
    """
    Main training loop.
    modified from Keras fit_generator
    """
    self = {}
    epoch = 0
    counter = 0
    out_labels = ['g_loss', 'd_loss', 'd_loss_fake', 'd_loss_legit', 'time']  # self.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    history = cbks.History()
    callbacks = [cbks.BaseLogger()] + callbacks + [history]
    if verbose:
        callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)

    callbacks._set_params({
        'nb_epoch': nb_epoch,
        'nb_sample': samples_per_epoch,
        'verbose': verbose,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    while epoch < nb_epoch:
      callbacks.on_epoch_begin(epoch)
      samples_seen = 0
      batch_index = 0
      while samples_seen < samples_per_epoch:
        z, x = next(generator)
        # build batch logs
        batch_logs = {}
        if type(x) is list:
          batch_size = len(x[0])
        elif type(x) is dict:
          batch_size = len(list(x.values())[0])
        else:
          batch_size = len(x)
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_size
        callbacks.on_batch_begin(batch_index, batch_logs)

        t1 = time.time()
        d_losses = d_train(x, z, counter)
        z, x = next(generator)
        g_loss, samples, xs = g_train(x, z, counter)
        outs = (g_loss, ) + d_losses + (time.time() - t1, )
        counter += 1

        # save samples
        if batch_index % 100 == 0:
          join_image = np.zeros_like(np.concatenate([samples[:64], xs[:64]], axis=0))
          for j, (i1, i2) in enumerate(zip(samples[:64], xs[:64])):
            join_image[j*2] = i1
            join_image[j*2+1] = i2
          save_images(join_image, [8*2, 8],
                      './outputs/samples_%s/train_%s_%s.png' % (name, epoch, batch_index))

          samples, xs = sampler(z, x)
          join_image = np.zeros_like(np.concatenate([samples[:64], xs[:64]], axis=0))
          for j, (i1, i2) in enumerate(zip(samples[:64], xs[:64])):
            join_image[j*2] = i1
            join_image[j*2+1] = i2
          save_images(join_image, [8*2, 8],
                      './outputs/samples_%s/test_%s_%s.png' % (name, epoch, batch_index))

        for l, o in zip(out_labels, outs):
            batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)

        # construct epoch logs
        epoch_logs = {}
        batch_index += 1
        samples_seen += batch_size

      if saver is not None:
        saver(epoch)

      callbacks.on_epoch_end(epoch, epoch_logs)
      epoch += 1

    # _stop.set()
    callbacks.on_train_end()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generative model trainer')
  parser.add_argument('model', type=str, default="bn_model", help='Model definitnion file')
  parser.add_argument('--name', type=str, default="autoencoder", help='Name of the model.')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  # parser.add_argument('--time', type=int, default=1, help='How many temporal frames in a single input.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--loadweights', dest='loadweights', action='store_true', help='Start from checkpoint.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  MODEL_NAME = args.model
  logging.info("Importing get_model from {}".format(args.model))
  exec("from models."+MODEL_NAME+" import get_model")
  # try to import `cleanup` from model file
  try:
    exec("from models."+MODEL_NAME+" import cleanup")
  except:
    cleanup = old_cleanup

  model_code = open('models/'+MODEL_NAME+'.py').read()

  if not os.path.exists("./outputs/results_"+args.name):
      os.makedirs("./outputs/results_"+args.name)
  if not os.path.exists("./outputs/samples_"+args.name):
      os.makedirs("./outputs/samples_"+args.name)

  with tf.Session() as sess:
    g_train, d_train, sampler, saver, loader, extras = get_model(sess=sess, name=args.name, batch_size=args.batch, gpu=args.gpu)

    # start from checkpoint
    if args.loadweights:
      loader()

    train_model(args.name, g_train, d_train, sampler,
                gen(20, args.host, port=args.port),
                samples_per_epoch=args.epochsize,
                nb_epoch=args.epoch, verbose=1, saver=saver
                )
