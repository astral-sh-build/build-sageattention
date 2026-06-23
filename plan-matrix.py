# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json
import os

from packaging.version import Version

# Add or remove versions as needed based on sageattention compatibility.
# SageAttention requires PyTorch >= 2.3.0
SAGEATTENTION_SUPPORTED_TORCH_VERSIONS = [
    "2.4.1",
    "2.5.1",
    "2.6.0",
    "2.7.1",
    "2.8.0",
    "2.9.1",
    "2.10.0",
    "2.11.0",
    "2.12.1",
]

ARCH_TORCH_PAIRS = {
    "x86_64": ["2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0", "2.9.1", "2.10.0", "2.11.0", "2.12.1"],
    "aarch64": ["2.6.0", "2.7.1", "2.8.0", "2.9.1", "2.10.0", "2.11.0", "2.12.1"],
}

# Supported Python versions for each PyTorch version.
# SageAttention requires Python >= 3.9
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_PYTHON_SUPPORT = {
    "2.4": ["3.9", "3.10", "3.11", "3.12"],
    "2.5": ["3.9", "3.10", "3.11", "3.12"],
    "2.6": ["3.9", "3.10", "3.11", "3.12"],
    "2.7": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.8": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.9": ["3.10", "3.11", "3.12", "3.13"],
    "2.10": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "2.11": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "2.12": ["3.10", "3.11", "3.12", "3.13", "3.14"],
}

# Minimum and maximum CUDA versions for each PyTorch version.
# SageAttention requires CUDA >= 12.0
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
PYTORCH_CUDA_RANGES: dict[str, tuple[str, str]] = {
    "2.4": ("11.8", "12.4"),
    "2.5": ("11.8", "12.4"),
    "2.6": ("11.8", "12.6"),
    "2.7": ("11.8", "12.8"),
    "2.8": ("11.8", "12.9"),
    "2.9": ("12.6", "13.0"),
    "2.10": ("12.6", "13.0"),
    "2.11": ("12.6", "13.0"),
    "2.12": ("12.6", "13.2"),
}

# Actual CUDA versions to build against for each PyTorch version.
# SageAttention requires CUDA >= 12.0, so we only build for CUDA 12.x+
PYTORCH_CUDA_VERSIONS: dict[tuple[str, str], list[str]] = {
    ("2.4", "x86_64"): ["12.1", "12.4"],
    ("2.4", "aarch64"): ["12.4"],
    ("2.5", "x86_64"): ["12.1", "12.4"],
    ("2.5", "aarch64"): ["12.4"],
    ("2.6", "x86_64"): ["12.4", "12.6"],
    ("2.6", "aarch64"): ["12.6"],
    ("2.7", "x86_64"): ["12.6", "12.8"],
    ("2.7", "aarch64"): ["12.8"],
    ("2.8", "x86_64"): ["12.6", "12.8", "12.9"],
    ("2.8", "aarch64"): ["12.9"],
    ("2.9", "x86_64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.9", "aarch64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.10", "x86_64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.10", "aarch64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.11", "x86_64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.11", "aarch64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.12", "x86_64"): ["12.6", "13.0", "13.2"],
    ("2.12", "aarch64"): ["12.6", "13.0", "13.2"],
}

# CUDA architectures to build against for each PyTorch version.
TORCH_CUDA_ARCH_LIST = {
    # https://github.com/pytorch/pytorch/blob/d990dada86a8ad94882b5c23e859b88c0c255bda/torch/utils/cpp_extension.py#L1938
    ("2.4", "12.1"): "7.0;7.5;8.0;8.6+PTX",
    ("2.4", "12.4"): "7.0;7.5;8.0;8.6;9.0+PTX",
    # https://github.com/pytorch/pytorch/blob/32f585d9346e316e554c8d9bf7548af9f62141fc/torch/utils/cpp_extension.py#L1937
    ("2.5", "12.1"): "7.0;7.5;8.0;8.6+PTX",
    ("2.5", "12.4"): "7.0;7.5;8.0;8.6;9.0+PTX",
    # https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/.ci/manywheel/build_cuda.sh#L60
    ("2.6", "12.4"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.6", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    # https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/.ci/manywheel/build_cuda.sh#L55
    ("2.7", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.7", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    # https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/.ci/manywheel/build_cuda.sh#L56
    ("2.8", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.8", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.8", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    # https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/.ci/manywheel/build_cuda.sh#L56
    ("2.9", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.9", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.9", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.9", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    # PyTorch 2.10 (following same pattern as 2.9)
    ("2.10", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.10", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.10", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.10", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    # PyTorch 2.11 (following same pattern as 2.10)
    ("2.11", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.11", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.11", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.11", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    ("2.12", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.12", "13.0"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.12", "13.2"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
}

# The glibc version to use for each PyTorch version, for manylinux builds.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_GLIBC_VERSION: dict[str, str] = {
    "2.4": "2_17",
    "2.5": "2_17",
    "2.6": "2_24",
    "2.7": "2_24",
    "2.8": "2_24",
    "2.9": "2_24",
    "2.10": "2_24",
    "2.11": "2_24",
    "2.12": "2_28",
}

AUDITWHEEL_BLANKET_EXCLUDES = [
    "libcuda.so",
    "libcuda.so.1",
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_cuda_cpp.so",
    "libtorch_cuda_cu.so",
    "libcufile_rdma.so",
    "libcufile_rdma.so.1",
    "libcufile.so.1",
    "libcufile.so.0",
    "libcufile.so",
    "libcurand.so.10",
]

AUDITWHEEL_CUDA_VERSION_EXCLUDES = {
    "11": [
        "libcudart.so.11",
        "libcudart.so.11.0",
    ],
    "12": [
        "libcudart.so.12",
        "libcudart.so.12.0",
    ],
    "13": [
        "libcudart.so.13",
        "libcudart.so.13.0",
    ],
}


# Matrix exclusions.
EXCLUSIONS = [
    # No exclusions yet.
]


def main() -> None:
    # Every matrix member is a primary 5-tuple of:
    # `torch-version`: the PyTorch version as "X.Y.Z", e.g. "2.7.0"
    # `python-version`: the Python version as "3.X", e.g. "3.10"
    # `cuda-version`: the CUDA version as "X.Y.Z", e.g. "12.8.1"
    # `cxx11-abi`: "TRUE" or "FALSE"
    # `target-arch`: the target architecture, e.g. "x86_64" or "aarch64"

    rows = []
    for target_arch, torch_versions in ARCH_TORCH_PAIRS.items():
        for torch_version in torch_versions:
            if torch_version not in SAGEATTENTION_SUPPORTED_TORCH_VERSIONS:
                continue

            torch_version_parsed = Version(torch_version)
            torch_x_y = f"{torch_version_parsed.major}.{torch_version_parsed.minor}"
            for python_version in TORCH_PYTHON_SUPPORT[torch_x_y]:
                cuda_versions = PYTORCH_CUDA_VERSIONS[(torch_x_y, target_arch)]
                for cuda_version in cuda_versions:
                    cuda_version_parsed = Version(cuda_version)

                    # The CXX11 ABI became the default in PyTorch 2.7.0, but was also used in
                    # PyTorch 2.6.0 (but _only_ for the CUDA 12.6 builds).
                    #
                    # See: https://pytorch.org/blog/pytorch2-6/
                    cxx11_abi = torch_version_parsed >= Version("2.7.0") or (
                        torch_version_parsed == Version("2.6.0")
                        and cuda_version_parsed >= Version("12.6")
                    )

                    row = {
                        "target-arch": target_arch,
                        "torch-version": str(torch_version_parsed),
                        "python-version": python_version,
                        "cuda-version": cuda_version,
                        "cxx11-abi": "TRUE" if cxx11_abi else "FALSE",
                    }

                    if row not in EXCLUSIONS:
                        rows.append(row)

    # Transform each row to add various nice-to-have representations of fields.
    for row in rows:
        # `CI_*` variables: same as the original ones.
        row["CI_CUDA_VERSION"] = row["cuda-version"]
        row["CI_TORCH_VERSION"] = row["torch-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]

        # `MATRIX_CUDA_VERSION`: XY instead of X.Y
        cuda_version = Version(row["cuda-version"])
        row["MATRIX_CUDA_VERSION"] = f"{cuda_version.major}{cuda_version.minor}"

        # `MATRIX_TORCH_VERSION`: `torch-version`, but only X.Y, no patch
        torch_version = Version(row["torch-version"])
        row["MATRIX_TORCH_VERSION"] = f"{torch_version.major}.{torch_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # `MANYLINUX_CUDA_VERSION`: X.Y instead of X.Y.Z
        row["MANYLINUX_CUDA_VERSION"] = f"{cuda_version.major}.{cuda_version.minor}"

        # `MANYLINUX_CUDA_COMPAT_VERSION`: X-Y instead of X.Y.Z
        row["MANYLINUX_CUDA_COMPAT_VERSION"] = (
            f"{cuda_version.major}-{cuda_version.minor}"
        )

        # MANYLINUX_GLIBC_VERSION: the glibc version to use for manylinux builds.
        row["MANYLINUX_GLIBC_VERSION"] = TORCH_GLIBC_VERSION[
            row["MATRIX_TORCH_VERSION"]
        ]

        # `CI_AUDITWHEEL_EXCLUDES`: `--exclude {lib}` for each lib that should
        # be excluded when running `auditwheel repair`.
        cuda_major = str(cuda_version.major)
        auditwheel_excludes = (
            AUDITWHEEL_BLANKET_EXCLUDES
            + AUDITWHEEL_CUDA_VERSION_EXCLUDES.get(cuda_major, [])
        )
        row["CI_AUDITWHEEL_EXCLUDES"] = " ".join(
            f"--exclude {lib}" for lib in auditwheel_excludes
        )

        row["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST[
            (
                f"{torch_version.major}.{torch_version.minor}",
                f"{cuda_version.major}.{cuda_version.minor}",
            )
        ]

        # RUNNER: the GitHub Actions runner to use.
        if row["target-arch"] == "x86_64":
            row["RUNNER"] = "depot-ubuntu-24.04-8"
        elif row["target-arch"] == "aarch64":
            row["RUNNER"] = "depot-ubuntu-24.04-arm-8"
        else:
            raise ValueError(f"Unknown target arch: {row['target-arch']}")

    # For PR builds, limit matrix to a single entry for faster CI.
    if os.environ.get("LIMIT_MATRIX") == "1":
        rows = rows[-1:]
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
