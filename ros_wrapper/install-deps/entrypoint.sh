#!/usr/bin/env bash

if [ -f /arl-unity-ros/install/setup.bash ]; then
    source /arl-unity-ros/install/setup.bash
fi


if [ -f /home/root/orcvio_cpp/ros_wrapper/setup.bash ]; then
    cd /home/root/orcvio_cpp/ros_wrapper/ && {
        source setup.bash;
        cd -; }
fi

if [ "$#" -eq 0 ]; then
  exec bash
else
  exec "$@"
fi  

