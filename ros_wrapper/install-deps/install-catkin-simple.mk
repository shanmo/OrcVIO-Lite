#!/usr/bin/make -f
SHELL=/bin/bash -l
ROOT_DIR?=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

CATKIN_SIMPLE_VERSION?=0.1.1
SOURCE_PREFIX?=$(HOME)/.local/src/
STOW_PREFIX?=$(SOURCE_PREFIX)/stow

CATKIN_SIMPLE_DIR?=$(SOURCE_PREFIX)/catkin-simple-$(CATKIN_SIMPLE_VERSION)
CATKIN_SIMPLE_INSTALL_DIR?=$(STOW_PREFIX)/catkin-simple-$(CATKIN_SIMPLE_VERSION)
CATKIN_SIMPLE_INSTALLED?=$(CATKIN_SIMPLE_INSTALL_DIR)/share/catkin_simple/cmake/catkins_simpleConfig.cmake
CATKIN_SIMPLE_STOWED?=$(INSTALL_PREFIX)/share/catkin_simple/cmake/catkins_simpleConfig.cmake


$(CATKIN_SIMPLE_STOWED): $(CATKIN_SIMPLE_INSTALLED)
	stow --ignore='.*\.sh|.*\.bash|.*\.zsh|_setup_util.py|\.catkin|\.rosinstall' --dir=$(STOW_PREFIX) --target=$(INSTALL_PREFIX) catkin-simple-$(CATKIN_SIMPLE_VERSION)

$(CATKIN_SIMPLE_INSTALLED): $(CATKIN_SIMPLE_DIR)/CMakeLists.txt
	-mkdir -p $(dir $<)/build
	cd $(dir $<)/build/ \
		&& cmake .. -DCMAKE_INSTALL_PREFIX=$(CATKIN_SIMPLE_INSTALL_DIR) \
	    && $(MAKE) -C $(dir $<)/build install


$(CATKIN_SIMPLE_DIR)/CMakeLists.txt:
	-mkdir $(dir $(CATKIN_SIMPLE_DIR))
	git clone  https://github.com/catkin/catkin_simple.git $(@D)
	cd $(@D) && git checkout $(CATKIN_SIMPLE_VERSION)
	touch $@
