#!/usr/bin/make -f
SHELL=/bin/bash -l
ROOT_DIR?=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

SOPHUS_VERSION?=v1.0.0
SOURCE_PREFIX?=$(HOME)/.local/src/
STOW_PREFIX?=$(SOURCE_PREFIX)/stow

SOPHUS_DIR?=$(SOURCE_PREFIX)/sophus-$(SOPHUS_VERSION)
SOPHUS_INSTALL_DIR?=$(STOW_PREFIX)/sophus-$(SOPHUS_VERSION)
SOPHUS_INSTALLED?=$(SOPHUS_INSTALL_DIR)/lib/libSophus.so
SOPHUS_STOWED?=$(INSTALL_PREFIX)/lib/libSophus.so

sophus: $(SOPHUS_STOWED)

$(SOPHUS_STOWED): $(SOPHUS_INSTALLED)
	stow --dir=$(STOW_PREFIX) --target=$(INSTALL_PREFIX) sophus-$(SOPHUS_VERSION)

$(SOPHUS_INSTALLED): $(SOPHUS_DIR)/CMakeLists.txt \
                     $(SOPHUS_DIR)/.sys-dependencies
	-mkdir -p $(dir $<)/build
	cd $(dir $<)/build/ \
		&& . $(dir $<)/.sys-dependencies \
		&& cmake .. -DCMAKE_INSTALL_PREFIX=$(SOPHUS_INSTALL_DIR) \
	    && $(MAKE) -C $(dir $<)/build install

$(SOPHUS_DIR)/.sys-dependencies: #$(if $(isflux), ,$(CUDA_INSTALLED))
	# apt-get install build-essential
	# apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
	# apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
	# echo "module load cuda miniconda3 numpy/py3.6 matlab" > $@
	touch $@

$(SOPHUS_DIR)/CMakeLists.txt:
	-mkdir $(dir $(SOPHUS_DIR))
	git clone  https://github.com/strasdat/Sophus.git $(@D)
	cd $(@D) && git checkout $(SOPHUS_VERSION)
	touch $@

