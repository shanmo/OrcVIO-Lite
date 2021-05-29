#!/usr/bin/make -f
SHELL=/bin/bash -l
ROOT_DIR?=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

CERES_VERSION?=1.9.0
SOURCE_PREFIX?=$(HOME)/.local/src/
STOW_PREFIX?=$(SOURCE_PREFIX)/stow

CERES_DIR?=$(SOURCE_PREFIX)/ceres-$(CERES_VERSION)
CERES_INSTALL_DIR?=$(STOW_PREFIX)/ceres-$(CERES_VERSION)
CERES_INSTALLED?=$(CERES_INSTALL_DIR)/lib/libceres.so
CERES_STOWED?=$(INSTALL_PREFIX)/lib/libceres.so

ceres: $(CERES_STOWED)

$(CERES_STOWED): $(CERES_INSTALLED)
	stow --dir=$(dir $(CERES_INSTALL_DIR)) --target=$(INSTALL_PREFIX) ceres-$(CERES_VERSION)

$(CERES_INSTALLED): $(CERES_DIR)/CMakeLists.txt
	mkdir -p "$(CERES_DIR)/build"
	cd "$(CERES_DIR)/build" \
		&& cmake .. -DCMAKE_INSTALL_PREFIX=$(CERES_INSTALL_DIR) \
	    && make install

$(CERES_DIR)/CMakeLists.txt:
	mkdir -p $(@D)
	git clone https://github.com/ceres-solver/ceres-solver.git "$(@D)"
	cd "$(@D)" && git checkout $(CERES_VERSION)
	touch $@

echo-%:
	@echo $($*)
