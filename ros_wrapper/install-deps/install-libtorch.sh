VER=1.4.0
CUVER=cu101
URL=https://download.pytorch.org/libtorch/$CUVER/libtorch-cxx11-abi-shared-with-deps-$VER.zip
SOURCE_PREFIX=${HOME}/.local/src
STOW_PREFIX=${SOURCE_PREFIX}/stow
if [ ! -f "${STOW_PREFIX}/libtorch-$VER/share/cmake/Torch/TorchConfig.cmake" ]; then
    [ -f "${SOURCE_PREFIX}/libtorch-cxx11-abi-shared-with-deps-$VER.zip" ] ||
        wget $URL -O ${SOURCE_PREFIX}/libtorch-cxx11-abi-shared-with-deps-$VER.zip
    mkdir -p $STOW_PREFIX && \
        unzip ${SOURCE_PREFIX}/libtorch-cxx11-abi-shared-with-deps-$VER.zip -d $STOW_PREFIX
    mv $STOW_PREFIX/libtorch $STOW_PREFIX/libtorch-$VER
fi

patch --quiet -d "${STOW_PREFIX}/libtorch-$VER" -p 1 -u  < install-deps/libtorch-1.4.0-bug-cuda.cmake.patch
stow --dir=${STOW_PREFIX} --target=$INSTALL_PREFIX libtorch-$VER
