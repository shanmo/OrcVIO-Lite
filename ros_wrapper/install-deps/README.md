![Dockerfile_with_deps](https://github.com/wecacuee/orcvio-backend/workflows/Dockerfile_with_deps/badge.svg)

# Scripts to help install required system packages

1. First install the required apt packages.
```
bash ./install-apt-get-packages.sh
```

2. The you can run the scripts to install catkin-simple, sophus and opencv.
```
$ INSTALL_PREFIX=$(HOME)/.local/<project_name> SOURCE_PREFIX=$(HOME)/.local/src make -f install-catkin-simple.mk 
```
```
$ INSTALL_PREFIX=$(HOME)/.local/<project_name> SOURCE_PREFIX=$(HOME)/.local/src make -f install-sophus.mk 
```
```
$ INSTALL_PREFIX=$(HOME)/.local/<project_name> SOURCE_PREFIX=$(HOME)/.local/src make -f install-opencv.mk 
```
