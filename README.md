# Video LoRA

Video generation with LoRA support on Strix Halo. Supports **Wan2.2**, **LTX-Video**, **AnimateDiff**, and **CogVideoX**.

## Models & LoRAs

| Model | LoRA Support | Size | Notes |
|-------|-------------|------|-------|
| **Wan2.2-Fun** | Reward LoRAs, Camera Control LoRAs | 1.3B / 14B | Best open-source T2V + I2V |
| **LTX-Video** | IC LoRA detailer (in-context) | 13B | Video-to-video control |
| **AnimateDiff** | Motion + Style LoRAs | 1.5B | Largest community LoRA ecosystem |
| **CogVideoX** | Transformer LoRA | 2B / 5B | Good for coherent motion |
| **Stable Video Diff.** | UNet LoRA | 2.5B | Image-to-video |

## Quick Start

### Python (CPU — works everywhere)

```bash
pip install -e ".[dev]"

# AnimateDiff (SD1.5 + motion module) on CPU
video-lora generate --model animatediff --prompt "cat walking, cinematic" --frames 8

# List available models and LoRAs
video-lora list-models

# Wan2.2 image-to-video
video-lora generate --model wan --prompt "a cat walking" --input-image ./cat.png --frames 16

# Benchmark speed
video-lora benchmark --model wan --frames 8 --steps 10
```

### Zig + Vulkan (GPU — Strix Halo Radeon 8060S)

```bash
cd vulkan
zig build -Doptimize=ReleaseFast
./zig-out/bin/video-lora-vk --prompt "cinematic dolly zoom through cherry blossoms" --frames 16
./zig-out/bin/video-lora-vk --prompt "cat walking" --lora ./motion-lora.safetensors
```

Requires Zig 0.15.2+ and `glslc` (for shader compilation).

## Project Structure

```
video-lora/
├── src/video_lora/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (generate, list-models, benchmark)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wan.py          # Wan2.2 pipeline + LoRA + I2V
│   │   ├── ltx.py          # LTX-Video pipeline + LoRA + I2V
│   │   ├── animatediff.py  # AnimateDiff pipeline + LoRA
│   │   └── cogvideo.py     # CogVideoX pipeline + LoRA + I2V
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lora_loader.py  # Unified LoRA loading (single + multi)
│   │   ├── pipeline.py     # Base pipeline abstraction
│   │   └── scheduler.py    # Scheduler config registry
│   └── utils/
│       ├── __init__.py
│       └── export.py       # Export to GIF/MP4/frames
├── tests/
│   ├── __init__.py          # Comprehensive test suite
│   └── test_models.py
├── docs/
│   └── BENCHMARKS.md        # Speed and memory benchmarks
├── pyproject.toml
├── .github/workflows/ci.yml
└── README.md
```

## CLI Reference

```bash
video-lora generate [OPTIONS]
  --model, -m    {wan, ltx, animatediff, cogvideo}
  --prompt, -p   Text prompt (required)
  --input-image  Input image for I2V generation
  --lora         LoRA path (HF repo ID or .safetensors)
  --lora-weight  LoRA merge weight (default: 0.7)
  --frames       Number of frames (default: 16)
  --steps        Denoising steps (default: 50)
  --width        Output width (default: 640)
  --height       Output height (default: 480)
  --seed         Random seed
  --output, -o   Output path
  --no-progress  Disable progress bars

video-lora benchmark [OPTIONS]
  --model, -m    {wan, ltx, animatediff, cogvideo}
  --frames       Number of frames (default: 8)
  --steps        Denoising steps (default: 10)
  --runs         Benchmark runs (default: 3)

video-lora list-models
```

## Performance

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed benchmarks.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1 | Python backends (Wan, LTX, AnimateDiff, CogVideoX) | ✅ |
| 2 | LoRA loading + fusion | ✅ |
| 3 | Image-to-video | ✅ |
| 4 | CLI + benchmark | ✅ |
| 5 | GLSL shaders (conv, norm, act, attn, lora) | ✅ 6 shaders to SPIR-V |
| 6 | Zig Vulkan context + buffers | ✅ |
| 7 | SPIR-V pipeline + dispatchers | ✅ Conv2d, Attention, LoRA |
| 8 | UNet forward pass orchestration | 🔧 wired — needs real weights |
| 9 | AnimateDiff temporal attention | 📋 |
| 10 | API server (OpenAI-compatible) | 📋 |
| 11 | NPU video path (INT8 xclbin) | 📋 |
| 12 | Docker packaging | 📋 |
