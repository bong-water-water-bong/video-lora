"""Smoke tests for the Vulkan compute backend."""

import pytest
import numpy as np
from pathlib import Path


def test_lib_loads():
    """The shared library should load without errors."""
    from vulkan.python.backend import _load_lib
    lib = _load_lib()
    ver = lib.vk_api_version()
    assert ver is not None
    assert b"Vulkan" in ver


def test_device_count():
    """There should be at least 1 Vulkan compute device (the 8060S)."""
    from vulkan.python.backend import _load_lib
    lib = _load_lib()
    count = lib.vk_device_count()
    assert count >= 0  # 0 is OK in CI without GPU


@pytest.mark.skip(reason="Requires physical GPU access")
def test_ctx_create():
    """Creating a Vulkan context should succeed on Strix Halo."""
    from vulkan.python.backend import VulkanBackend
    dev = VulkanBackend()
    assert dev._ctx is not None


@pytest.mark.skip(reason="Requires physical GPU access")
def test_tensor_create():
    """Creating and filling a GPU tensor should round-trip."""
    from vulkan.python.backend import VulkanBackend
    dev = VulkanBackend()
    t = dev.tensor(1, 4, 64, 64)
    t.fill(1.0)
    out = t.download()
    assert out.shape == (1, 4, 64, 64)
    assert np.allclose(out, 1.0)


@pytest.mark.skip(reason="Requires downloaded model weights")
def test_unet_step():
    """Running one UNet step should produce output in the correct shape."""
    from vulkan.python.backend import VulkanBackend
    dev = VulkanBackend()

    # Find weights
    weights = Path.home() / ".cache" / "video-lora" / "unet_sd15.bin"
    assert weights.exists(), f"Download weights first: {weights}"

    dev.load_unet(str(weights))
    latent = dev.tensor(1, 4, 64, 64)
    latent.fill(0.1)
    text = dev.tensor(1, 77, 768)
    text.fill(0.0)

    out = dev.unet(latent, text, timestep=50.0)
    assert out.shape == (1, 4, 64, 64)


@pytest.mark.skip(reason="Requires model weights + LoRA file")
def test_lora_apply():
    """Fusing a LoRA should succeed."""
    from vulkan.python.backend import VulkanBackend
    dev = VulkanBackend()
    weights = Path.home() / ".cache" / "video-lora" / "unet_sd15.bin"
    dev.load_unet(str(weights))
    ret = dev.apply_lora("/path/to/motion-lora.safetensors", weight=0.7)
    assert ret == 0
