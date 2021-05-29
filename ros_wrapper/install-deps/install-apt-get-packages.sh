echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
add-apt-repository -y ppa:ignaciovizzo/opencv3-nonfree

apt-get update && \
    apt-get -y install \
            git \
            libceres-dev \
            libglew-dev \
            libopencv-dev \
            libsm-dev \
            libsuitesparse-dev \
            libxrender-dev \
            python-dev \
            python-pip \
            python3-pip \
            ros-melodic-cv-bridge \
            ros-melodic-eigen-conversions \
            ros-melodic-pcl-conversions \
            ros-melodic-pcl-ros \
            ros-melodic-pluginlib \
            ros-melodic-random-numbers \
            ros-melodic-ros-base \
            ros-melodic-rosbash \
            ros-melodic-roslaunch \
            ros-melodic-rviz \
            ros-melodic-tf-conversions \
            rsync \
            stow \
            unzip \
            virtualenv \
    && \
    rm -rf /var/lib/apt/lists/*
