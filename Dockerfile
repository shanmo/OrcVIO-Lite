ARG BASE_IMG=wecacuee/ros-melodic-bionic-nvidia
FROM $BASE_IMG

ENV ORCVIO_DIR /home/root/orcvio_cpp/
COPY cache $ORCVIO_DIR/cache
COPY include $ORCVIO_DIR/include
COPY src $ORCVIO_DIR/src
COPY cmake $ORCVIO_DIR/cmake
ENV CMAKE_MODULE_PATH $ORCVIO_DIR/cmake:$CMAKE_MODULE_PATH

COPY config $ORCVIO_DIR/config
ENV CATKIN_WORKSPACE $ORCVIO_DIR/ros_wrapper
RUN mkdir -p $CATKIN_WORKSPACE/install-deps/
COPY ros_wrapper/install-deps $CATKIN_WORKSPACE/install-deps/
COPY ros_wrapper/setup_once.bash $CATKIN_WORKSPACE/
COPY ros_wrapper/setup.bash $CATKIN_WORKSPACE/
COPY ros_wrapper/src $CATKIN_WORKSPACE/src
RUN cd $CATKIN_WORKSPACE && bash setup_once.bash

RUN mv $CATKIN_WORKSPACE/install-deps/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN echo "cd $CATKIN_WORKSPACE && { source $CATKIN_WORKSPACE/setup.bash; cd -; }" >> /$HOME/.bashrc
ENTRYPOINT [ /entrypoint.sh ]
