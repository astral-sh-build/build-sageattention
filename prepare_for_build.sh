#!/bin/bash
# Script to prepare the build environment for SageAttention.
#
# Example usage:
#   ./prepare_for_build.sh v2.2.0

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <sageattention_version>"
    echo "Example: $0 v2.2.0"
    exit 1
fi

SAGEATTENTION_VERSION=$1

# Apply patches.
patch_dir="${ROOT}/build_scripts/patches/${SAGEATTENTION_VERSION}"

# Not all SageAttention versions need patches.
if [ ! -d "${patch_dir}" ]; then
    echo "Warning: nothing to patch: patches/${SAGEATTENTION_VERSION} directory does not exist"
else
    for patch in "${patch_dir}"/*.patch; do
        # Skip if no patch files exist (only .gitkeep)
        if [ -f "${patch}" ]; then
            patch -p1 -d "${ROOT}" -i "${patch}"
        fi
    done
fi
