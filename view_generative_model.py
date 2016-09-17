#!/usr/bin/env python

"""
Usage:
>> server.py --time 60 --batch 64
>> ./make_gif.py transition --name transition --time 15 --batch 64
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import time
import cv2
from keras import callbacks as cbks
from keras import backend as K
import logging
import tensorflow as tf
import numpy as np
from scipy.misc import imsave, imresize
from tqdm import *

from server import client_generator
mixtures = 1


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='MiniBatch server')
  parser.add_argument('model', type=str, default="transition", help='Model definitnion file')
  parser.add_argument('--name', type=str, default="transition", help='Name of the model.')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--time', type=int, default=1, help='How many temporal frames in a single input.')
  parser.add_argument('--batch', type=int, default=256, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
  parser.add_argument('--loadweights', dest='loadweights', action='store_true', help='Start from checkpoint.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  MODEL_NAME = args.model
  logging.info("Importing get_model from {}".format(args.model))
  exec("from models."+MODEL_NAME+" import get_model, load, save")
  # try to import `cleanup` from model file
  try:
    exec("from models."+MODEL_NAME+" import cleanup")
  except:
    cleanup = old_cleanup

  model_code = open('models/'+MODEL_NAME+'.py').read()

  with tf.Session() as sess:
    K.set_session(sess)
    g_train, d_train, sampler, saver, loader, [G, E, T] = get_model(sess=sess, name=args.name, batch_size=args.batch, gpu=args.gpu)

    print("loading weights...")
    G.load_weights("./outputs/results_autoencoder/G_weights.keras".format(args.name))
    E.load_weights("./outputs/results_autoencoder/E_weights.keras".format(args.name))
    checkpoint_dir = './outputs/results_' + args.name
    T.load_weights(checkpoint_dir+"/T_weights.keras")

    if not os.path.exists("./video_"+args.name):
      os.makedirs("./video_"+args.name)

    # get data
    data = client_generator(hwm=20, host="localhost", port=5557)
    X = next(data)[0]  # [:, ::2]
    sh = X.shape
    X = X.reshape((-1, 3, 160, 320))
    X = np.asarray([cv2.resize(x.transpose(1, 2, 0), (160, 80)) for x in X])
    X = X/127.5 - 1.
    x = X.reshape((sh[0], args.time, 80, 160, 3))

    # estimate frames
    z_dim = 512
    I = E.input
    E_out = E(I)
    O = G.input
    G_out = G(O)
    print "Sampling..."
    for i in tqdm(range(128)):
      x = x.reshape((-1, 80, 160, 3))
      # code = E.predict(x, batch_size=args.batch*args.time)[0]
      code = sess.run([E_out[0]], feed_dict={I: x, K.learning_phase(): 1})[0]
      code = code.reshape((args.batch, args.time, z_dim))
      inp = code[:, :5]  # context is based on the first 5 frames only
      outs = T.predict(inp, batch_size=args.batch)
      imgs = sess.run([G_out], feed_dict={O: outs.reshape((-1, z_dim)), K.learning_phase(): 1})[0]
      # imgs = G.predict(outs[:, 0], batch_size=args.batch)
      x = x.reshape((args.batch, args.time, 80, 160, 3))
      x[0, :-1] = x[0, 1:]
      x[0, -1] = imgs[0]
      imsave("video_"+args.name+"/%03d.png" % i, imresize(imgs[0], (160, 320)))

    cmd = "ffmpeg -y -i ./video_"+args.name+"/%03d.png ./video_"+args.name+"/output.gif -vf fps=1"
    print(cmd)
    os.system(cmd)
