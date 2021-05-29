DIR="$(pwd)"
#source /opt/ros/melodic/setup.bash
source $DIR/devel/setup.bash

#export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages
#export PYTHONPATH=/usr/local/lib/python3.6/dist-packages:$PYTHONPATH
export ROS_PYTHON_VERSION=2
export DATA_DIR=$HOME/dataset
source $DIR/.tox/py27/bin/activate
source $DIR/dependencies/activate.sh
