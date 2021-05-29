# Build

- Clone the repository recursively

``` shellsession
git clone --recurse-submodules <repository_url> orcvio-lite
```

- Install dependencies and run catkin build:

``` shellsession
$ cd orcvio-lite/ros_wrapper
$ export ROS_DISTRO=melodic
$ apt-get update && \
    apt-get install -y python-pip && \
    pip install --upgrade pip && \
    pip install catkin-tools catkin-pkg rospkg rosdep && \
    cp rosdep.yaml /etc/ros/rosdep/sources.list.d/19-orcvio-lite.yaml && \
    echo "yaml /etc/ros/rosdep/sources.list.d/19-orcvio-lite.yaml" > /etc/ros/rosdep/sources.list.d/19-orcvio-lite.list && \
    rosdep init && \
    rosdep update && \
    rosdep install -y -r --ignore-src --from-path src 
```


- Activate the environment

``` shellsession
$ source /opt/ros/$ROS_DISTRO/setup.bash
```

- Compile

``` shellsession
$ catkin init -w .
$ catkin build
$ source devel/setup.bash
```

# Message 

- object detection message is [wm_od_interface_msgs](https://github.com/kschmeckpeper/object_detection)
