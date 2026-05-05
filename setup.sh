#!/bin/bash

CUDA_HOME=/usr/local/cuda

echo Install uv and uv packages
pip install uv
uv venv
uv sync --python 3.12

echo Install cuda 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb 
chmod a+x .
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-4

echo Setup toolchain
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get dist-upgrade -y

source .venv/bin/activate
python -m ipykernel install --user --name s4casting --display-name "Python (s4casting)"

# Install nbstripout to strip output from notebooks on git commits
uv run nbstripout --install

bash data/get_data.sh
