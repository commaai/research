"""
This file is named after `dask` for historical reasons. We first tried to
use dask to coordinate the hdf5 buckets but it was slow and we wrote our own
stuff.
"""
import numpy as np
import h5py
import time
import logging
import traceback

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def concatenate(camera_names, time_len):
  logs_names = [x.replace('camera', 'log') for x in camera_names]

  angle = []  # steering angle of the car
  speed = []  # speed of the car
  hdf5_camera = []  # the camera hdf5 files need to continue open
  c5x = []
  filters = []
  lastidx = 0

  for cword, tword in zip(camera_names, logs_names):
    try:
      with h5py.File(tword, "r") as t5:
        c5 = h5py.File(cword, "r")
        hdf5_camera.append(c5)
        x = c5["X"]
        c5x.append((lastidx, lastidx+x.shape[0], x))

        speed_value = t5["speed"][:]
        steering_angle = t5["steering_angle"][:]

        # approximate alignment
        idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")
        angle.append(steering_angle[idxs])
        speed.append(speed_value[idxs])

        goods = np.abs(angle[-1]) <= 200

        filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
        lastidx += goods.shape[0]
        # check for mismatched length bug
        print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
        if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
          raise Exception("bad shape")

    except IOError:
      import traceback
      traceback.print_exc()
      print "failed to open", tword

  angle = np.concatenate(angle, axis=0)
  speed = np.concatenate(speed, axis=0)
  filters = np.concatenate(filters, axis=0).ravel()
  print "training on %d/%d examples" % (filters.shape[0], angle.shape[0])
  return c5x, angle, speed, filters, hdf5_camera


first = True


def datagen(filter_files, time_len=1, batch_size=256, ignore_goods=False):
  """
  Parameters:
  -----------
  leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
  """
  global first
  assert time_len > 0
  filter_names = sorted(filter_files)

  logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))

  c5x, angle, speed, filters, hdf5_camera = concatenate(filter_names, time_len=time_len)
  filters_set = set(filters)

  logger.info("camera files {}".format(len(c5x)))

  X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')
  angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
  speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

  while True:
    try:
      start = time.time()

      for count in range(time_len):
        # first choose a rnd hdf5 bucket, then a rnd index
        # c5x[i] = (idx_start, idx_end, hdf5 dataset "X")
        idx_hdf5 = np.random.choice(len(hdf5_camera))
        (idx_start, idx_end, x) = c5x[idx_hdf5]
        idx = np.random.randint(idx_start, idx_end-time_len+1)

        # get X_BATCH
        X_batch[count] = x[(idx-idx_start):(idx-idx_start+time_len)]
        angle_batch[count] = np.copy(angle[idx:(idx+time_len)])[:, None]
        speed_batch[count] = np.copy(speed[idx:(idx+time_len)])[:, None]

      # sanity check
      assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

      logging.debug("load image : {}s".format(time.time()-start))
      print("%6.2f ms" % ((time.time()-start)*1000.0))

      if first:
        print("shape of X_batch: (%s)" % ', '.join(map(str, X_batch.shape)))
        print("shape of angle: (%s)" % ', '.join(map(str, angle_batch.shape)))
        print("shape of speed: (%s)" % ', '.join(map(str, speed_batch.shape)))
        first = False

      yield (X_batch, angle_batch, speed_batch)

    except KeyboardInterrupt:
      raise

    except:
      traceback.print_exc()
      pass
