import importlib
from importlib.metadata import version

import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    device = torch.device("cuda")
    return device


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("sageattention") == "2.2.0+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("module_name", ["sageattention"])
def test_native_module(device: torch.device, module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("causal", [False, True])
def test_quantized_attention(device: torch.device, causal: bool) -> None:
    from sageattention import sageattn

    torch.manual_seed(0)
    query, key, value = (
        torch.randn((2, 4, 128, 64), device=device, dtype=torch.float16)
        for _ in range(3)
    )
    actual = sageattn(query, key, value, tensor_layout="HND", is_causal=causal)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.float(), key.float(), value.float(), is_causal=causal
    ).to(query.dtype)
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)
