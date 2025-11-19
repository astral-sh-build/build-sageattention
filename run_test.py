"""
Modal test script for SageAttention wheel builds.

This script tests a single wheel from GitHub Actions on GPU-enabled machines across
all supported CUDA versions for that PyTorch version. The wheel is downloaded during
the image build using the GITHUB_TOKEN.

Run with:
    WHEEL_NAME=<name> GITHUB_OWNER=<owner> GITHUB_REPO=<repo> GITHUB_RUN_ID=<run_id> GITHUB_TOKEN=<token> uv run --with modal modal run run_test.py

NOTE: Modal currently only supports x86_64 architecture GPUs (NVIDIA A10G, A100, H100, etc.).
If you're testing an aarch64 wheel, you'll need to build an x86_64 version first.
"""

import os
import re
import sys

import modal

# GitHub Actions information (from environment variables)
WHEEL_NAME = os.environ.get("WHEEL_NAME")
GITHUB_OWNER = os.environ.get("GITHUB_OWNER")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_RUN_ID = os.environ.get("GITHUB_RUN_ID")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not WHEEL_NAME:
    raise ValueError("WHEEL_NAME environment variable must be set")
if not GITHUB_OWNER:
    raise ValueError("GITHUB_OWNER environment variable must be set")
if not GITHUB_REPO:
    raise ValueError("GITHUB_REPO environment variable must be set")
if not GITHUB_RUN_ID:
    raise ValueError("GITHUB_RUN_ID environment variable must be set")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable must be set")

# CUDA versions to test for each PyTorch major.minor version
# SageAttention requires CUDA >= 12.0
CUDA_TEST_VERSIONS: dict[str, list[str]] = {
    "2.4": ["12.1", "12.4"],
    "2.5": ["12.1", "12.4"],
    "2.6": ["12.4", "12.6"],
    "2.7": ["12.6", "12.8"],
    "2.8": ["12.6", "12.8", "12.9"],
    "2.9": ["12.6", "12.8", "12.9"],
}


def parse_wheel_filename(wheel_path: str) -> dict[str, str]:
    """Parse the wheel filename to extract build information.

    Supports sageattention wheel naming patterns.
    """
    # Try sageattention pattern
    pattern = r"sageattention-(?P<version>[^+]+)\+cu(?P<cuda_ver>[\d.]+)torch(?P<torch_ver>[\d.]+)cxx11abi(?P<cxx11_abi>\w+)-cp(?P<py_ver>\d+)-"
    match = re.search(pattern, wheel_path)

    if not match:
        raise ValueError(f"Could not parse wheel filename: {wheel_path}")

    info = match.groupdict()
    # Parse version manually without packaging.version
    torch_ver_parts = info["torch_ver"].split(".")
    info["torch_xy"] = f"{torch_ver_parts[0]}.{torch_ver_parts[1]}"
    return info


def get_pytorch_cuda_index_url(cuda_version: str) -> str:
    """Get the PyTorch index URL for a specific CUDA version."""
    cuda_suffix = cuda_version.replace(".", "")
    return f"https://download.pytorch.org/whl/cu{cuda_suffix}"


# Parse wheel information
wheel_info = parse_wheel_filename(WHEEL_NAME)
torch_version = wheel_info["torch_ver"]
torch_xy = wheel_info["torch_xy"]
py_ver_num = wheel_info["py_ver"]
python_version = f"{py_ver_num[0]}.{py_ver_num[1:]}"

# Get CUDA versions to test
cuda_versions_to_test = CUDA_TEST_VERSIONS.get(torch_xy, [])
if not cuda_versions_to_test:
    raise ValueError(f"No CUDA test versions configured for PyTorch {torch_xy}")

print(f"Setting up tests for {WHEEL_NAME}")
print(f"  PyTorch {torch_version}, testing with CUDA versions: {cuda_versions_to_test}")

# Create a single app
app = modal.App("test-sageattention")

# Create a function that will download the wheel
def download_wheel_func():
    import io
    import os
    import requests
    import zipfile

    # GitHub API settings
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    GITHUB_OWNER = os.environ["GITHUB_OWNER"]
    GITHUB_REPO = os.environ["GITHUB_REPO"]
    GITHUB_RUN_ID = os.environ["GITHUB_RUN_ID"]
    WHEEL_NAME = os.environ["WHEEL_NAME"]

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get artifacts for this run
    artifacts_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{GITHUB_RUN_ID}/artifacts"
    response = requests.get(artifacts_url, headers=headers)
    response.raise_for_status()

    # Find the artifact matching our wheel name
    artifacts = response.json()["artifacts"]
    artifact = None
    for a in artifacts:
        if a["name"] == WHEEL_NAME:
            artifact = a
            break

    if not artifact:
        raise ValueError(f"Could not find artifact {WHEEL_NAME}")

    # Download the artifact
    download_url = artifact["archive_download_url"]
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()

    # Extract the wheel from the zip
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith(".whl"):
                with open(f"/{name}", "wb") as f:
                    f.write(zip_ref.read(name))
                print(f"Downloaded: {name}")
                break


# Create images for each CUDA version
# SageAttention requires triton >= 3.0.0
images = {}
for cuda_version in cuda_versions_to_test:
    images[cuda_version] = (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("git")
        .pip_install(
            f"torch=={torch_version}",
            index_url=get_pytorch_cuda_index_url(cuda_version),
        )
        .pip_install("packaging", "ninja", "requests", "triton>=3.0.0")
        .env({
            "GITHUB_TOKEN": GITHUB_TOKEN,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_RUN_ID": GITHUB_RUN_ID,
            "WHEEL_NAME": WHEEL_NAME,
        })
        .run_function(download_wheel_func)
        .pip_install(f"/{WHEEL_NAME}")
    )


def _run_sageattention_test(cuda_version: str):
    """Core test logic for SageAttention - parameterized by CUDA version."""
    import subprocess

    import torch

    print("=" * 80)
    print(f"Testing with CUDA {cuda_version}")
    print("=" * 80)
    print()

    print("=" * 80)
    print("GPU Information")
    print("=" * 80)
    subprocess.run(["nvidia-smi"], check=True)
    print()

    print("=" * 80)
    print("Python and Package Versions")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    print()

    print("=" * 80)
    print("Testing SageAttention")
    print("=" * 80)

    print("✓ SageAttention imported")

    from sageattention import sageattn

    print("✓ Imported sageattn")

    # Basic SageAttention test
    # SageAttention works similarly to Flash Attention but with quantization
    batch_size, seqlen, nheads, headdim = 2, 2048, 16, 128
    q = torch.randn(
        batch_size, nheads, seqlen, headdim, dtype=torch.float16, device="cuda"
    )
    k = torch.randn(
        batch_size, nheads, seqlen, headdim, dtype=torch.float16, device="cuda"
    )
    v = torch.randn(
        batch_size, nheads, seqlen, headdim, dtype=torch.float16, device="cuda"
    )

    output = sageattn(q, k, v)
    print(f"✓ SageAttention computation works: {output.shape}")

    # Verify output shape
    expected_shape = (batch_size, nheads, seqlen, headdim)
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )
    print(f"✓ Output shape is correct: {output.shape}")

    print()
    print("=" * 80)
    print(f"All tests passed for CUDA {cuda_version}!")
    print("=" * 80)

    return {
        "status": "success",
        "cuda_test_version": cuda_version,
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0))
        if torch.cuda.is_available()
        else None,
    }


# Static test functions for each CUDA version
# NOTE: SageAttention requires Ampere GPUs or newer (compute capability 8.0+)
# Modal GPU options: "any", "a10g", "a100", "h100", "l4", "l40s"
# Ampere+ options: "a10g", "a100", "h100", "l4", "l40s"


@app.function(image=images.get("12.1"), gpu="a10g", timeout=600)
def test_cuda121():
    return _run_sageattention_test("12.1")


@app.function(image=images.get("12.4"), gpu="a10g", timeout=600)
def test_cuda124():
    return _run_sageattention_test("12.4")


@app.function(image=images.get("12.6"), gpu="a10g", timeout=600)
def test_cuda126():
    return _run_sageattention_test("12.6")


@app.function(image=images.get("12.8"), gpu="a10g", timeout=600)
def test_cuda128():
    return _run_sageattention_test("12.8")


@app.function(image=images.get("12.9"), gpu="a10g", timeout=600)
def test_cuda129():
    return _run_sageattention_test("12.9")


# Map CUDA versions to their test functions
test_functions = {
    "12.1": test_cuda121,
    "12.4": test_cuda124,
    "12.6": test_cuda126,
    "12.8": test_cuda128,
    "12.9": test_cuda129,
}


@app.local_entrypoint()
def main():
    """Main entry point - runs all tests in parallel."""
    print("=" * 80)
    print("SageAttention Multi-CUDA Version Test Suite")
    print("=" * 80)
    print(f"Wheel: {WHEEL_NAME}")
    print(f"PyTorch version: {torch_version}")
    print(f"Testing against CUDA versions: {cuda_versions_to_test}")
    print("=" * 80)
    print()
    print("Running all tests in parallel...")
    print()

    # Run all tests in parallel
    tasks = []
    for cuda_version in cuda_versions_to_test:
        tasks.append(test_functions[cuda_version].spawn())

    # Wait for all tasks to complete
    results = {}
    for i, cuda_version in enumerate(cuda_versions_to_test):
        try:
            results[cuda_version] = tasks[i].get()
        except Exception as e:
            results[cuda_version] = {"status": "failed", "error": str(e)}

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Wheel: {WHEEL_NAME}")
    print(f"PyTorch: {torch_version}")
    print()

    failed = []
    for cuda_version in cuda_versions_to_test:
        result = results[cuda_version]
        if isinstance(result, Exception):
            status = "✗ FAIL"
            failed.append(cuda_version)
            print(f"CUDA {cuda_version}: {status}")
            print(f"  Error: {result}")
        elif result.get("status") == "success":
            status = "✓ PASS"
            print(f"CUDA {cuda_version}: {status}")
            print(f"  GPU: {result.get('gpu_name', 'unknown')}")
            print(f"  PyTorch CUDA: {result.get('cuda_version', 'unknown')}")
        else:
            status = "✗ FAIL"
            failed.append(cuda_version)
            print(f"CUDA {cuda_version}: {status}")
            print(f"  Error: {result.get('error', 'unknown error')}")

    print("=" * 80)

    if failed:
        print(f"\n❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(cuda_versions_to_test)} tests passed!")
