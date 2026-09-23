#!/bin/bash
# Set up the development environment.
#
# Works on the remote Ubuntu instances (installs CUDA + toolchain) and on
# macOS / other systems (python environment only).
#
# Usage:
#   bash setup.sh                 # install packages from the public internet (PyPI)
#   bash setup.sh --artifactory   # install packages via the Alliander Artifactory
#                                 # index, using ARTIFACTORY_USERNAME/ARTIFACTORY_TOKEN
#                                 # from .env (see .env.example)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
ARTIFACTORY_INDEX="https://alliander.jfrog.io/artifactory/api/pypi/pypi-all/simple"
OS="$(uname -s)"

PACKAGE_SOURCE="internet"
for arg in "$@"; do
    case "$arg" in
        --artifactory) PACKAGE_SOURCE="artifactory" ;;
        --internet)    PACKAGE_SOURCE="internet" ;;
        -h|--help)
            echo "Usage: bash setup.sh [--artifactory | --internet]"
            echo "  --artifactory  install python packages via Artifactory (credentials from .env)"
            echo "  --internet     install python packages from public PyPI (default)"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option '${arg}'. Run 'bash setup.sh --help'." >&2
            exit 1
            ;;
    esac
done

if [ "${PACKAGE_SOURCE}" = "artifactory" ]; then
    echo "Configure pip to use Artifactory"
    # Exports PIP_CONFIG_FILE so pip below resolves through Artifactory.
    # shellcheck disable=SC1091
    source "${REPO_DIR}/setup_pip.sh"
fi

echo Install uv and uv packages
if ! command -v uv >/dev/null 2>&1; then
    # Prefer pip (routes through Artifactory when PIP_CONFIG_FILE is set); on
    # PEP 668 "externally-managed" Pythons (e.g. Homebrew) fall back to the
    # standalone installer, which needs no Python at all.
    pip install --only-binary ':all:' uv || {
        curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    }
fi

if [ "${PACKAGE_SOURCE}" = "artifactory" ]; then
    echo "Authenticate uv against Artifactory"
    bash "${REPO_DIR}/setup_uv.sh"
    # Make uv resolve through Artifactory instead of public PyPI. This
    # re-resolves uv.lock against the Artifactory index; do not commit the
    # rewritten lockfile.
    export UV_DEFAULT_INDEX="${ARTIFACTORY_INDEX}"
fi

uv venv
uv sync --python 3.12

if [ "${PACKAGE_SOURCE}" = "artifactory" ]; then
    # uv sync re-resolves uv.lock against the Artifactory index. Restore the
    # committed (public PyPI) lockfile so the rewrite can never be committed;
    # the .venv built above is unaffected.
    git -C "${REPO_DIR}" restore uv.lock 2>/dev/null || true
fi

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
