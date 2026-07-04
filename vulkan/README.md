# Vulkan Compute Backend for SD1.5 / AnimateDiff

Low-level Vulkan compute shaders for accelerated video generation,
inspired by the Zinc engine's architecture but targeting diffusion models.

## Architecture

```
vulkan/
├── shaders/          # GLSL compute shaders (compiled to SPIR-V via glslc)
│   ├── conv2d.comp           # 2D convolution (UNet res/blocks)
│   ├── group_norm.comp       # Group normalization
│   ├── silu.comp             # SiLU activation
│   ├── attention.comp        # Self-attention / cross-attention
│   ├── elementwise.comp      # Add, multiply, scale
│   ├── upsample.comp         # Nearest-neighbor upsample
│   ├── downsample.comp       # Average pool downsample
│   ├── lora_merge.comp       # Fuse LoRA weights into base weights
│   ├── vae_decoder.comp      # VAE decoder conv
│   └── time_embed.comp       # Time step embedding
├── src/              # C Vulkan wrapper (called from Python via ctypes)
│   ├── vk_ctx.h              # Vulkan context (device, queue, memory)
│   ├── vk_ctx.c              # Context implementation
│   ├── vk_pipeline.h         # Compute pipeline management
│   ├── vk_pipeline.c         # Pipeline implementation
│   ├── vk_tensor.h           # Tensor abstraction (NCHW GPU buffer)
│   ├── vk_tensor.c           # Tensor implementation
│   ├── vk_unet.h             # SD1.5 UNet inference orchestration
│   ├── vk_unet.c             # UNet implementation
│   └── vk_api.h              # Single C API header for Python ctypes
├── python/           # Python ctypes bindings
│   ├── __init__.py
│   ├── backend.py            # VulkanBackend — drop-in for torch.device
│   ├── tensor.py             # VulkanTensor — GPU tensor wrapper
│   └── unet.py               # UNet inference using Vulkan
├── build_ci.sh       # CI build script (glslc + gcc)
└── Makefile           # Build targets
```

## Build Dependencies

- `glslc` (from Vulkan SDK or `shaderc` package)
- `gcc` or `clang` with `-lvulkan`
- Vulkan headers + loader (`libvulkan-dev`)

```bash
make shaders   # compile .comp → .spv
make lib       # build libvk_diffusion.so
make test      # run sanity checks
```

## Calling Convention (Python → C)

```python
from vulkan.python.backend import VulkanBackend

# Drop-in replacement for torch.device("cuda")
device = VulkanBackend()

# Run UNet inference
noise = device.tensor(batch, 4, 64, 64)  # latent noise
text_emb = device.tensor(1, 77, 768)      # CLIP text embedding
ts = device.tensor([timestep])            # timestep

latent = device.unet(noise, text_emb, ts)  # → denoised latent
```

## Shader Pattern

All shaders follow Zinc's pattern:
```glsl
#version 460
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Input  { float in_data[]; };
layout(set = 0, binding = 1) buffer Output { float out_data[]; };

layout(push_constant) uniform PushConstants {
    uint N;        // total elements
    uint C;        // channels
    uint H;        // height
    uint W;        // width
    uint stride;   // conv stride
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N) return;
    // ... compute ...
}
```
