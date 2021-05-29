# OrcVIO-Lite

Object residual constrained Visual-Inertial Odometry (OrcVIO) is a visual-inertial odometry pipeline, which is tightly coupled with tracking and optimization over structured object models. It provides accurate trajectory estimation and large-scale object-level mapping from online **Mono+IMU** data.

OrcVIO-Lite only uses **bounding boxs** and no keypoints. The object mapping module and VIO module are implemented in separate ROS nodelets and are decoupled.  

Related publication: [OrcVIO: Object residual constrained Visual-Inertial Odometry](https://arxiv.org/pdf/2007.15107.pdf)



## 1. Prerequisites

This repository was tested on Ubuntu 18.04 with [ROS Melodic](http://wiki.ros.org/melodic/Installation). 

The core algorithm depends on `Eigen`, `Boost`, `Suitesparse`, `Ceres`, `OpenCV`, `Sophus`, `GTest`



## 2. Install

### 2.1 Non-ROS version

```
$ git clone --recursive https://github.com/moshanATucsd/OrcVIO_Lite.git
$ cd OrcVIO_Lite
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=Release ..
$ make
```

### 2.2 ROS version

```
$ git clone --recursive https://github.com/moshanATucsd/OrcVIO_Lite.git
$ cd OrcVIO_Lite/ros_wrapper
$ catkin_make
$ source ./devel/setup.bash
```



## 3. Evaluation 

### 3.0 Unit Tests (Non-ROS version)

The tests are compiled by default and have to be run from top of the project directory.

```
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=True ..
$ make
$ cd .. && ./build/orcvio_tests ; cd -
```

 Or directly run `make test` from `build` directory and the test logs are saved in `build/Testing/Temporary/LastTest.log`.

To debug using VS code, define `VS_DEBUG_MODE`, then run

```
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..
make
```

this will generate the `compile_commands.json` in the `build` folder.



### 3.2 Evaluation on EuRoC dataset

Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to PATH_TO_EUROC_DATASET.

#### 3.2.1 Non-ROS Version

- **VIO**

```
$ cd PATH_TO_ORCVIO_LITE/
$ ./run_euroc.sh PATH_TO_EUROC_DATASET
```

- **Object Mapping**

```
 
```



#### 3.2.2 ROS Version

- **TODO**





### 3.3 Evaluation on ERL dataset

Indoor rosbags were collected with [Realsense D435i](https://www.intelrealsense.com/depth-camera-d435i/) in Existential Robotics Lab, University of California San Diego.

- **Download dataset**

  ERL indoor dataset (chairs, monitors, one way): [data1](https://drive.google.com/file/d/1ibw-zyK--qBXF7cKiR32yRaDPVj25YBE/view?usp=sharing)

  ERL indoor dataset (chairs, monitors, round trip): [data2](https://drive.google.com/file/d/1JsMp1DhT9yr_oBVpwHArArmQhshvjcV2/view?usp=sharing)



#### 3.3.1 Non-ROS Version



#### 3.3.2 ROS Version

- **VIO**

In your ros_wrapper directory,

```
$ roslaunch orcvio orcvio_rs_d435i_offline.launch path_bag:=/PATH_TO_ERL_DATASET/
```

- **Object Mapping**







### 3.3 Evaluation on your own dataset



### 3.??? Unclassified demos

- Download [this rosbag](https://drive.google.com/file/d/1rrEoUi1jyiaoN-rcMQHS53vVrFUN57oc/view?usp=sharing) and change the bag file path in the launch file, run `roslaunch orcvio orcvio_kitti_object.launch` in `ros_wrapper` folder will output similar results with [this demo](https://youtu.be/VlnG64WS434)
- Use [this rosbag](https://drive.google.com/file/d/1MnLzq2nWBRPx3OhSz0i26ZvOufre_EWn/view?usp=sharing), run `roslaunch orcvio orcvio_medfield_object.launch`, will output [this demo](https://youtu.be/Ia7vIo3eI5A)
- Use [this rosbag](https://drive.google.com/file/d/1CvRTyRUwLFEktQ7o5bDRHUgHFXZePbX6/view?usp=sharing), run `roslaunch orcvio orcvio_unity_object.launch`, will output [this demo](https://drive.google.com/file/d/1E9wshVRMcZGLZCfYzuwk6Vke3eng5gVd/view?usp=sharing)



- **Do we need phoenix rosbags?**
- in phoenix rosbag, only `outdoor_joy_2020-10-18-12-23-40.bag` has correct IMU timestamps, since there is no objects, `roslaunch orcvio orcvio_phoenix.launch` is used, and the demo is [here](https://drive.google.com/file/d/11FR1gr_GGz-QS45kUd2IqzMF3TuKk80X/view?usp=sharing), VIO only works for a while and when the robot encounters a bumpy road it drifts away, vibration isolation is suggested for realsense sensor. 
- phoenix rosbag with object detections are added for the demo, using launch file `roslaunch orcvio orcvio_phoenix_object.launch`, the rosbag is [here](https://drive.google.com/file/d/19QVxveLFZpXXGUTOftmD49VKiLsrsnG6/view?usp=sharing), and here is the [demo](https://drive.google.com/file/d/1PM8naJfea7jYfd4JYa2PTrxj9YbjpmN0/view?usp=sharing) 

 



## TODO

- [ ] Detailed instruction on using own data
- [ ] Add packages (image_converter, starmap, etc.) and refine launch files
- [ ] Add test examples on the KITTI rosbag 
- [ ] May add object mapping examples for KITTI, seems impossible for euroc
- [ ] Remove arl, medfield, etc.  
- [ ] Finalize the example ERL data 



## Citation

```
@inproceedings{orcvio,
  title = {OrcVIO: Object residual constrained Visual-Inertial Odometry},
  author={M. {Shan} and Q. {Feng} and N. {Atanasov}},
  year = {2020},
  booktitle={IEEE Intl. Conf. on Intelligent Robots and Systems (IROS).},
  url = {https://moshanatucsd.github.io/orcvio_githubpage/},
  pdf = {https://arxiv.org/abs/2007.15107}
}
```



## Acknowledgement

balabalabalas



## License

```
MIT License
Copyright (c) 2020 ERL at UCSD
```



## Reference 

- [LARVIO](https://github.com/PetWorm/LARVIO)
- [MSCKF](https://github.com/KumarRobotics/msckf_vio)
- [OpenVINS](https://github.com/rpng/open_vins)
- [OpenVINS eva](https://github.com/symao/open_vins)