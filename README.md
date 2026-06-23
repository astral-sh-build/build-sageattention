# build-sageattention

Pre-built Linux wheels for [SageAttention](https://github.com/thu-ml/SageAttention), quantized
attention kernels optimized for Ampere, Ada, and Hopper GPUs, across Python, PyTorch, CUDA, and CPU
architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index for each CUDA
version. Each wheel has a local version suffix that identifies the CUDA, PyTorch, and C++ ABI it
was built against, such as `sageattention==2.2.0+cu12.8torch2.10.0cxx11abiTRUE`, and requires the
matching PyTorch minor release.

Pre-built wheels are available on [Astral's GPU indexes](https://wheels.astral.sh/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add sageattention --index astral-cu128=https://wheels.astral.sh/simple/cu128/
```

This configures the index and uses it as the source for `sageattention`:

```toml
[tool.uv.sources]
sageattention = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astral.sh/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astral.sh/simple/cu128/ sageattention
```

## Supported versions

Wheels are available for the following `sageattention` versions:

- [`2.2.0`](https://github.com/astral-sh-build/build-sageattention/releases/tag/v2.2.0-r1)

The latest release, SageAttention 2.2.0, supports the following combinations:

| PyTorch | Python    | `x86_64` CUDA          | `aarch64` CUDA         |
| ------- | --------- | ---------------------- | ---------------------- |
| 2.4.1   | 3.9–3.12  | 12.1, 12.4             | —                      |
| 2.5.1   | 3.9–3.12  | 12.1, 12.4             | —                      |
| 2.6.0   | 3.9–3.12  | 12.4, 12.6             | 12.6                   |
| 2.7.1   | 3.9–3.13  | 12.6, 12.8             | 12.8                   |
| 2.8.0   | 3.9–3.13  | 12.6, 12.8, 12.9       | 12.9                   |
| 2.9.1   | 3.10–3.13 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.10.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.11.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.12.1  | 3.10–3.14 | 12.6, 13.0, 13.2       | 12.6, 13.0, 13.2       |

## License

build-sageattention is licensed under the [Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
