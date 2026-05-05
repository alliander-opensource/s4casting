<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# Installation Guide

## Setup Options
You can run the code either:
- **Remotely** using an AWS SageMaker environment.
- **Locally** on your own machine.  

The setup differs slightly due to AWS S3 access and GPU availability.

## Setting Up a SageMaker Environment 

**Important:** SageMaker instances incur costs while running. Make sure to **stop your instance** when you're done.

### 0. Open SageMaker Studio
- Go to the AWS Console and search for “Amazon SageMaker AI”.
- Open “SageMaker Studio”.
- Click “Open Studio” to launch the environment.

### 1. Choose Your SageMaker Space
You can use different SageMaker spaces:
- **Code Editor (VSCode)**  -> Recommended for a full development workflow (terminal, Git, debugging)
- **JupyterLab**

Choose based on your preference.

### 2. Configure Your Instance
Recommended settings:
- **Instance type:** `ml.g5.xlarge`  
  See Vantage for pricing and GPU details.
- **Volume size:**  
  - Measurement data: < 20 GB  
  - Weather data: < 1 GB  
  - Logs: potentially large  
  → Recommended: **250 GB**  
  > Note: Increasing storage after initial allocation may cause errors.

### 3. Clone the Repository
Use either:
- GitHub integration in VSCode, or  
- Git CLI (see Confluence guide)

> Ask for the configurations from the administrator.

### 5. Run the Setup Script
On first boot:
```bash
cd s4casting
bash setup.sh
```
This will:
- Install python package manager (uv) 
- Install CUDA 
- Install common development packages / toolchain support

> Takes a few minutes. Only needed once per instance.

### 6. Download data 
You can download all the data from aws with the following command: 

```bash
bash data/get_data.sh
```

### 7. You are ready to go!

## Setting Up a Local Environment

  > Note: These local setup instructions target WSL (Ubuntu) or native Linux. Use WSL on Windows.

### 1. Install and Configure uv
Uv is a python dependency manager, a replacement for the traditional `requirements.txt` file. Create the virtual environment and install dependencies:
```bash
pip install uv
uv venv
uv sync --python 3.12
```

### 2. Download the Data
From the home directory run:
```bash
./data/get_data.sh`
```

### 3. Install CUDA (optional)
If you want to train locally with GPU acceleration, you need to install CUDA.

There are several ways to do this depending on your environment, for example using system packages:

```bash
# Add NVIDIA keyring and repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-4
```
