SCRIPTDIR="$( cd "$( dirname "$0" 2>/dev/null || pwd)" && pwd )"
echo $SCRIPTDIR
export CATKIN_WORKSPACE=$SCRIPTDIR

sudo sh install-deps/install-apt-get-packages.sh
source /opt/ros/melodic/setup.bash
sh install-deps/run-rosdep.sh

{
    export INSTALL_PREFIX=${CATKIN_WORKSPACE}/dependencies
    mkdir -p $INSTALL_PREFIX
    export SUDO=sudo
    [ -d $INSTALL_PREFIX/share/Ceres/ ] || \
        make -f install-deps/install-ceres.mk
    [ -d $INSTALL_PREFIX/share/Pangolin/ ] || \
        make -f install-deps/install-pangolin.mk
    [ -f $INSTALL_PREFIX/include/sophus/se3.hpp ] || \
        make -f install-deps/install-sophus.mk
    [ -f $INSTALL_PREFIX/share/catkin_simple/cmake/catkin_simpleConfig.cmake ] || \
        make -f install-deps/install-catkin-simple.mk
    [ -d $INSTALL_PREFIX/lib/libtorch.so ] || \
        sh install-deps/install-libtorch.sh
}
export CMAKE_PREFIX_PATH=$INSTALL_PREFIX

PYVER=2.7
virtualenv --python=python${PYVER} $CATKIN_WORKSPACE/.tox/py${PYVER/./}
source $CATKIN_WORKSPACE/.tox/py${PYVER/./}/bin/activate
pip install --no-cache -r install-deps/pip-requirements.txt
sed -e "s!%ROOT_DIR%!$INSTALL_PREFIX!g" "$CATKIN_WORKSPACE/install-deps/activate.sh.template" > "$INSTALL_PREFIX/activate.sh"

source $INSTALL_PREFIX/activate.sh

cd $CATKIN_WORKSPACE
catkin init -w $CATKIN_WORKSPACE
catkin config --extend /opt/ros/melodic/
catkin build
source setup.bash

