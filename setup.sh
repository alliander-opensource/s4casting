#!/bin/bash
# Set up the development environment.
#
# Works on the remote Ubuntu instances (installs CUDA + toolchain) and on
# macOS / other systems (python environment only).
#
# Usage:
#   bash setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
OS="$(uname -s)"

echo Install uv and uv packages
if ! command -v uv >/dev/null 2>&1; then
    # Prefer pip; on PEP 668 "externally-managed" Pythons (e.g. Homebrew)
    # fall back to the standalone installer, which needs no Python at all.
    pip install --only-binary ':all:' uv || {
        curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    }
fi

uv venv
uv sync --python 3.12

if [ "${OS}" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
    export CUDA_HOME=/usr/local/cuda

    echo Install cuda
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-4

    echo Setup toolchain
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get dist-upgrade -y
else
    echo "Skipping CUDA and toolchain install (Ubuntu/Debian only, detected ${OS})."
fi

# shellcheck disable=SC1091
source "${REPO_DIR}/.venv/bin/activate"
python -m ipykernel install --user --name s4casting --display-name "Python (s4casting)"

# Install nbstripout to strip output from notebooks on git commits
uv run nbstripout --install

bash "${REPO_DIR}/data/get_data.sh"
