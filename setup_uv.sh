# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

#!/bin/bash
# Authenticate uv against the Alliander Artifactory package index. Reads the
# Artifactory credentials from .env and runs `uv auth login`, which persists
# the credentials in uv's own credential store (outside this repo).
#
# Run standalone:
#   bash setup_uv.sh
#
# It is also invoked automatically by `setup.sh --artifactory` (remote setup).

_auth_uv() {
    local repo_dir artifactory_url
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    artifactory_url="https://alliander.jfrog.io/artifactory/api/pypi/pypi-all"

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

    uv auth login "${artifactory_url}" \
        --username "${ARTIFACTORY_USERNAME}" \
        --password "${ARTIFACTORY_TOKEN}"

    echo "uv authenticated against Artifactory."
}

_auth_uv
