#!/usr/bin/env python

from itertools import izip
import numpy as np
import h5py
from progress.bar import Bar
import sys

import rospy
import rosbag
from sensor_msgs.msg import Imu, Image

def main():
  if len(sys.argv) < 2:
    print("Usage: {} dataset_name".format(sys.argv[0]))
    exit(1)

  file_name = sys.argv[1]
  log_file = h5py.File('../dataset/log/{}.h5'.format(file_name))
  camera_file = h5py.File('../dataset/camera/{}.h5'.format(file_name))  

  zipped_log = izip(
    log_file['times'],
    log_file['fiber_accel'],
    log_file['fiber_gyro'])

  with rosbag.Bag('{}.bag'.format(file_name), 'w') as bag:
    bar = Bar('Camera', max=len(camera_file['X']))
    for i, img_data in enumerate(camera_file['X']):
      m_img = Image()
      m_img.header.stamp = rospy.Time.from_sec(0.01 * i)
      m_img.height = img_data.shape[1]
      m_img.width = img_data.shape[2]
      m_img.step = 3 * img_data.shape[2]
      m_img.encoding = 'rgb8'
      m_img.data = np.transpose(img_data, (1, 2, 0)).flatten().tolist()
      
      bag.write('/camera/image_raw', m_img, m_img.header.stamp)
      bar.next()
      
    bar.finish()

    bar = Bar('IMU', max=len(log_file['times']))
    for time, v_accel, v_gyro in zipped_log:
      m_imu = Imu()
      m_imu.header.stamp = rospy.Time.from_sec(time)
      [setattr(m_imu.linear_acceleration, c, v_accel[i]) for i, c in enumerate('xyz')]
      [setattr(m_imu.angular_velocity, c, v_gyro[i]) for i, c in enumerate('xyz')]

      bag.write('/fiber_imu', m_imu, m_imu.header.stamp)
      bar.next()

    bar.finish()

if __name__ == "__main__":
  main()
