#!/bin/bash

set -e

# fake sudo function to be used by docker build
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# update sources.list
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
sudo echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# install packages
sudo apt-get update -y && \
sudo apt-get install -y --no-install-recommends \
    build-essential \
    intel-oneapi-mpi \
    intel-oneapi-mpi-devel

LIBDIR1="/opt/intel/oneapi/mpi/latest/lib"

# update ldconfig
echo "${LIBDIR1}" | sudo tee /etc/ld.so.conf.d/intel-oneapi-mpi.conf >/dev/null
sudo ldconfig
