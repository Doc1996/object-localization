#!/usr/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade

python3 --version
pip --version

python3 -m pip install numpy
python3 -m pip install opencv-python