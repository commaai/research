# the people's comma

## the paper

[Learning a Driving Simulator](http://arxiv.org/abs/1608.01230)

## the comma.ai driving dataset

7 and a quarter hours of largely highway driving. Enough to train what we had in [Bloomberg](http://www.bloomberg.com/features/2015-george-hotz-self-driving-car/).

## Examples

We present two Machine Learning Experiments to show
possible ways to use this dataset:


<img src="./images/selfsteer.gif">

[Training a steering angle predictor](SelfSteering.md)


<img src="./images/drive_simulator.gif">

[Training a generative image model](DriveSim.md)

## Downloading the dataset

```
./get_data.sh
```

or get it at [archive.org comma dataset](https://archive.org/details/comma-dataset)

45 GB compressed, 80 GB uncompressed

```
dog/2016-01-30--11-24-51 (7.7G)
dog/2016-01-30--13-46-00 (8.5G)
dog/2016-01-31--19-19-25 (3.0G)
dog/2016-02-02--10-16-58 (8.1G)
dog/2016-02-08--14-56-28 (3.9G)
dog/2016-02-11--21-32-47 (13G)
dog/2016-03-29--10-50-20 (12G)
emily/2016-04-21--14-48-08 (4.4G)
emily/2016-05-12--22-20-00 (7.5G)
frodo/2016-06-02--21-39-29 (6.5G)
frodo/2016-06-08--11-46-01 (2.7G)
```

Dataset referenced on this page is copyrighted by comma.ai and published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

## Dataset structure
The dataset consists of 10 videos clips of variable size recorded at 20 Hz
with a camera mounted on the windshield of an Acura ILX 2016. In parallel to the videos
we also recorded some measurements such as car's speed, acceleration,
steering angle, GPS coordinates, gyroscope angles. See the full `log` list [here](Logs.md).
These measurements are transformed into a uniform 100 Hz time base.

The dataset folder structure is the following:
```bash
+-- dataset
|   +-- camera
|   |   +-- 2016-04-21--14-48-08
|   |   ...
|   +-- log
|   |   +-- 2016-04-21--14-48-08
|   |   ...
```

All the files come in hdf5 format and are named with the time they were recorded.
The camera dataset has shape `number_frames x 3 x 160 x 320` and `uint8` type.
One of the `log` hdf5-datasets is called `cam1_ptr` and addresses the alignment
between camera frames and the other measurements.

## Requirements
[anaconda](https://www.continuum.io/downloads)  
[tensorflow-0.9](https://github.com/tensorflow/tensorflow)  
[keras-1.0.6](https://github.com/fchollet/keras)  
[cv2](https://anaconda.org/menpo/opencv3)

## Hiring

Want a job at [comma.ai](http://comma.ai)?

Show us amazing stuff on this dataset

## Credits

Riccardo Biasini, George Hotz, Sam Khalandovsky, Eder Santana, and Niel van der Westhuizen
