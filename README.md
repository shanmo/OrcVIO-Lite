# OrcVIO-Lite

### About 

- Object residual constrained Visual-Inertial Odometry (OrcVIO) is a visual-inertial odometry pipeline, which is tightly coupled with tracking and optimization over structured object models. It provides accurate trajectory estimation and large-scale object-level mapping from online **Mono+IMU** data.

- OrcVIO-Lite only uses **bounding boxs** and no keypoints. The object mapping module and VIO module are implemented in separate ROS nodelets and are decoupled.  

- Related publication: [OrcVIO: Object residual constrained Visual-Inertial Odometry](https://arxiv.org/pdf/2007.15107.pdf), this is the journal version submitted to T-RO. [Project website](https://moshan.cf/orcvio_githubpage/)

### Citation

```
@inproceedings{shan2020orcvio,
  title={OrcVIO: Object residual constrained Visual-Inertial Odometry},
  author={Shan, Mo and Feng, Qiaojun and Atanasov, Nikolay},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5104--5111},
  year={2020},
  organization={IEEE}
}   
```

## 1. Prerequisites

This repository was tested on Ubuntu 18.04 with [ROS Melodic](http://wiki.ros.org/melodic/Installation). 

The core algorithm depends on `Eigen`, `Boost`, `Suitesparse`, `Ceres`, `OpenCV`, `Sophus`, `GTest`


## 2. Installation

### 2.1 Non-ROS version

```
$ git clone --recursive https://github.com/shanmo/OrcVIO-Lite.git
$ cd OrcVIO-Lite
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=Release ..
$ make
```

### 2.2 ROS version

- The ROS version also depends on [catkin simple](https://github.com/catkin/catkin_simple), please put it in the `src` folder

```
$ git clone --recursive https://github.com/shanmo/OrcVIO-Lite.git
$ cd OrcVIO-Lite/ros_wrapper
$ catkin_make
$ source ./devel/setup.bash
```

## 3. Evaluation 

### 3.1 Non-ROS Version

- **Download dataset**

  Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to ***PATH_TO_EUROC_DATASET/***.

- **VIO**

```
$ cd PATH_TO_ORCVIO_LITE/
$ ./run_euroc.sh PATH_TO_EUROC_DATASET/
```



### 3.2 ROS Version

- **Download dataset**

  [ERL indoor dataset (chairs, monitors)](https://www.dropbox.com/s/mxin2et8io2nsab/erl_hallway_round_trip.bag?dl=0)

  * Indoor rosbags were collected with [Realsense D435i](https://www.intelrealsense.com/depth-camera-d435i/) in Existential Robotics Lab, University of California San Diego.

- **VIO**

```
$ cd OrcVIO_Lite/ros_wrapper/
$ roslaunch orcvio orcvio_vio_rs_d435i.launch path_bag:=PATH_TO_ERL_DATASET/
```

- **VIO & Mapping**

```
$ cd OrcVIO_Lite/ros_wrapper/
$ roslaunch orcvio orcvio_mapping_rs_d435i.launch path_bag:=PATH_TO_ERL_DATASET/
```



## 4. TODO

- Instruction on preparing datasets
- More demos on public datasets

 


## License

```
MIT License
Copyright (c) 2021 ERL at UCSD
```



## Reference 

- [LARVIO](https://github.com/PetWorm/LARVIO)
- [MSCKF](https://github.com/KumarRobotics/msckf_vio)
- [OpenVINS](https://github.com/rpng/open_vins)
- [OpenVINS eval](https://github.com/symao/open_vins)
