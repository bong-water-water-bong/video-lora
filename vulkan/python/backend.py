"""Vulkan compute backend for SD1.5/AnimateDiff inference.

Drop-in replacement for ``torch.device("cuda")`` when running on AMD GPUs
via Vulkan compute (no CUDA/ROCm required).

Usage::

    from vulkan.python.backend import VulkanBackend, VulkanTensor

    device = VulkanBackend()
    noise = device.tensor(1, 4, 64, 64)  # [B, C, H, W] latent
    noise.fill(0.1)

    text_emb = device.tensor(1, 77, 768)
    text_emb.upload(numpy_array)

    ts = 50.0  # timestep
    out = device.unet(noise, text_emb, ts)
"""

import ctypes
import numpy as np
from pathlib import Path

_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    # Search paths for the shared library
    candidates = [
        Path(__file__).parent.parent / "libvk_diffusion.so",
        Path(__file__).parent.parent.parent / "vulkan" / "libvk_diffusion.so",
        Path("/usr/local/lib/libvk_diffusion.so"),
    ]
    for path in candidates:
        if path.exists():
            _LIB = ctypes.CDLL(str(path))
            break
    else:
        raise RuntimeError(
            "libvk_diffusion.so not found. Build it first:\n"
            "  cd vulkan && make"
        )

    _LIB.vk_ctx_create.restype = ctypes.c_void_p
    _LIB.vk_ctx_destroy.argtypes = [ctypes.c_void_p]
    _LIB.vk_api_version.restype = ctypes.c_char_p
    return _LIB


class VulkanTensor:
    """GPU tensor managed by the Vulkan backend.

    Wraps a ``vk_tensor_t*`` with shape ``[B, C, H, W]``.
    """

    def __init__(self, handle: ctypes.c_void_p, shape: tuple):
        self._handle = handle
        self.shape = shape  # (B, C, H, W)

    @property
    def b(self) -> int: return self.shape[0]
    @property
    def c(self) -> int: return self.shape[1]
    @property
    def h(self) -> int: return self.shape[2]
    @property
    def w(self) -> int: return self.shape[3]

    def upload(self, data: np.ndarray):
        """Upload a numpy array to the GPU buffer."""
        lib = _load_lib()
        lib.vk_tensor_upload.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
        lib.vk_tensor_upload(self._handle, data.astype(np.float32).ravel())

    def download(self) -> np.ndarray:
        """Download GPU buffer to a numpy array."""
        lib = _load_lib()
        out = np.zeros(self.b * self.c * self.h * self.w, dtype=np.float32)
        lib.vk_tensor_download.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
        lib.vk_tensor_download(self._handle, out)
        return out.reshape(self.shape)

    def fill(self, value: float):
        lib = _load_lib()
        lib.vk_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.vk_tensor_fill(self._handle, value)

    def __del__(self):
        lib = _load_lib()
        lib.vk_tensor_destroy.argtypes = [ctypes.c_void_p]
        lib.vk_tensor_destroy(self._handle)


class VulkanBackend:
    """Vulkan compute backend — drop-in for ``torch.device("cuda")``.

    Initialises a Vulkan compute context on the Radeon 8060S (or any
    Vulkan 1.3 compute-capable GPU).
    """

    def __init__(self):
        self._lib = _load_lib()
        self._ctx = self._lib.vk_ctx_create()
        self._unet_handle = None

        ver = self._lib.vk_api_version()
        print(f"[VulkanBackend] {ver.decode()} — context active")

    def tensor(self, b: int, c: int, h: int, w: int) -> VulkanTensor:
        """Create a GPU tensor with shape ``[B, C, H, W]``."""
        self._lib.vk_tensor_create.argtypes = [ctypes.c_void_p, ctypes.c_uint32] * 4
        self._lib.vk_tensor_create.restype = ctypes.c_void_p
        handle = self._lib.vk_tensor_create(self._ctx, b, c, h, w)
        return VulkanTensor(handle, (b, c, h, w))

    def load_unet(self, weights_path: str):
        """Load SD1.5 UNet weights from a file and prepare pipelines."""
        self._lib.vk_unet_create.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.vk_unet_create.restype = ctypes.c_void_p
        self._unet_handle = self._lib.vk_unet_create(self._ctx, weights_path.encode())
        print(f"[VulkanBackend] UNet loaded from {weights_path}")

    def unet(self, latent: VulkanTensor, text_emb: VulkanTensor,
             timestep: float) -> VulkanTensor:
        """Run one denoising step through the UNet."""
        if self._unet_handle is None:
            raise RuntimeError("Call load_unet() first")

        out = self.tensor(*latent.shape)
        self._lib.vk_unet_step.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_float, ctypes.c_void_p]
        self._lib.vk_unet_step(self._unet_handle,
                               latent._handle, text_emb._handle,
                               ctypes.c_float(timestep), out._handle)
        return out

    def apply_lora(self, lora_path: str, weight: float = 0.7) -> int:
        """Fuse a LoRA into the loaded UNet weights."""
        self._lib.vk_unet_apply_lora.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
        self._lib.vk_unet_apply_lora.restype = ctypes.c_int
        ret = self._lib.vk_unet_apply_lora(self._unet_handle, lora_path.encode(), weight)
        return ret

    def __del__(self):
        if self._unet_handle:
            self._lib.vk_unet_destroy(self._unet_handle)
        if self._ctx:
            self._lib.vk_ctx_destroy(self._ctx)
