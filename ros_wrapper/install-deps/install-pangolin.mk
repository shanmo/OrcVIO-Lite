#!/usr/bin/make -f
SHELL=/bin/bash -l
ROOT_DIR?=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PANGOLIN_VERSION?=v0.5
SOURCE_PREFIX?=$(HOME)/.local/src/
STOW_PREFIX?=$(SOURCE_PREFIX)/stow

PANGOLIN_DIR?=$(SOURCE_PREFIX)/pangolin-$(PANGOLIN_VERSION)
PANGOLIN_INSTALL_DIR?=$(STOW_PREFIX)/pangolin-$(PANGOLIN_VERSION)
PANGOLIN_INSTALLED?=$(PANGOLIN_INSTALL_DIR)/lib/libpangolin.a
PANGOLIN_STOWED?=$(INSTALL_PREFIX)/lib/libpangolin.a

pangolin: $(PANGOLIN_STOWED)

$(PANGOLIN_STOWED): $(PANGOLIN_INSTALLED)
	stow --dir=$(dir $(PANGOLIN_INSTALL_DIR)) --target=$(INSTALL_PREFIX) pangolin-$(PANGOLIN_VERSION)

$(PANGOLIN_INSTALLED): $(PANGOLIN_DIR)/CMakeLists.txt
	mkdir -p "$(PANGOLIN_DIR)/build"
	cd "$(PANGOLIN_DIR)/build" \
		&& cmake .. -DCMAKE_INSTALL_PREFIX=$(PANGOLIN_INSTALL_DIR) \
	    && cmake --build . \
		&& make install

$(PANGOLIN_DIR)/CMakeLists.txt:
	mkdir -p $(@D)
	git clone https://github.com/stevenlovegrove/Pangolin.git "$(@D)" || true
	cd "$(@D)" && git checkout $(PANGOLIN_VERSION)
	touch $@

echo-%:
	@echo $($*)
