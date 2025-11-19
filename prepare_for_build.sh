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

if [ ! -d "${patch_dir}" ]; then
    echo "No patches to apply for SageAttention version ${SAGEATTENTION_VERSION}"
else
    for patch in "${patch_dir}"/*.patch; do
        if [ -f "${patch}" ]; then
            patch -p1 -d "${ROOT}" -i "${patch}"
        fi
    done
fi

# Ensure libcuda.so exists in the CUDA stubs directory for linking.
# This is particularly important for ARM builds where the stub might not exist.
if [ -n "${CUDA_STUB_LIBDIR}" ] && [ -d "${CUDA_STUB_LIBDIR}" ]; then
    echo "Ensuring libcuda.so exists in ${CUDA_STUB_LIBDIR}"
    cd "${CUDA_STUB_LIBDIR}"

    # Check if libcuda.so already exists
    if [ ! -f "libcuda.so" ]; then
        # Try to find libcuda.so.1 first (most common)
        if [ -f "libcuda.so.1" ]; then
            echo "Creating symlink libcuda.so -> libcuda.so.1"
            ln -sf libcuda.so.1 libcuda.so
        else
            # Look for any libcuda.so.* variant
            stub_lib=$(ls libcuda.so.* 2>/dev/null | head -n1)
            if [ -n "${stub_lib}" ]; then
                echo "Creating symlink libcuda.so -> ${stub_lib}"
                ln -sf "${stub_lib}" libcuda.so
            else
                echo "ERROR: No libcuda stub library found in ${CUDA_STUB_LIBDIR}"
                echo "Available files:"
                ls -la
                exit 1
            fi
        fi
    else
        echo "libcuda.so already exists"
    fi

    cd "${ROOT}"
fi
