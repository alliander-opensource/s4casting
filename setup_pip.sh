# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

#!/bin/bash
# Configure pip to use the Alliander Artifactory package index instead of
# public PyPI. Reads the Artifactory credentials from .env and writes a
# gitignored runtime config (pip.conf.local) that pip is pointed at via
# PIP_CONFIG_FILE.
#
# SOURCE this script so PIP_CONFIG_FILE is exported into your shell:
#   source setup_pip.sh
#
# It is also invoked automatically by `setup.sh --artifactory` (remote setup).

_auth_pip() {
    local repo_dir
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [ ! -f "${repo_dir}/.env" ]; then
        echo "ERROR: No .env file found. Copy .env.example to .env and configure your" >&2
        echo "       Artifactory credentials first. See docs/INSTALLATION.md." >&2
        return 1
    fi

    set -a
    # shellcheck disable=SC1091
    source "${repo_dir}/.env"
    set +a

    if [ -z "${ARTIFACTORY_USERNAME:-}" ] || [ "${ARTIFACTORY_USERNAME}" = "..." ] || \
       [ -z "${ARTIFACTORY_TOKEN:-}" ] || [ "${ARTIFACTORY_TOKEN}" = "..." ]; then
        echo "ERROR: Artifactory configuration is missing. Set ARTIFACTORY_USERNAME and" >&2
        echo "       ARTIFACTORY_TOKEN in your .env file. See docs/INSTALLATION.md." >&2
        return 1
    fi

    PIP_CONFIG_FILE="${repo_dir}/pip.conf.local"
    export PIP_CONFIG_FILE
    sed -e "s|<AL_NUMBER>|${ARTIFACTORY_USERNAME}|g" \
        -e "s|<ARTIFACTORY_TOKEN>|${ARTIFACTORY_TOKEN}|g" \
        "${repo_dir}/pip.conf" > "${PIP_CONFIG_FILE}"

    echo "pip configured to use Artifactory (PIP_CONFIG_FILE=${PIP_CONFIG_FILE})."
}

_auth_pip
