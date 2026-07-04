# Video Diffusion Engine

**Zero Python. Pure C++23. NPU + CPU. 1bit.systems.**

C++ video diffusion engine on Strix Halo. Reuses the NPU engine's INT8 GEMM for accelerated inference.

Built from `engine/video/` in the [1bit-systems](https://github.com/bong-water-water-bong/1bit-systems) monorepo.

## Quick Start

```bash
# Clone the engine
git clone git@github.com:bong-water-water-bong/1bit-systems.git
cd 1bit-systems/engine/video

# Build (CPU only)
g++ -std=c++23 -O3 -march=native -fopenmp -o video_engine src/video_main.cpp -lm

# Generate
./video_engine --prompt "a cat walking, cinematic" --frames 16 --steps 50

# Benchmark
./video_engine --prompt "test" --frames 8 --steps 10 --benchmark
```

## Architecture

```
engine/video/
├── src/
│   ├── video_main.cpp      # Entry point + CLI
│   ├── video_model.h       # Model config + weight loading
│   ├── video_dit.h         # Diffusion Transformer forward pass
│   ├── video_sampler.h     # DDIM + Flow Matching denoising
│   └── video_vae.h         # VAE decoder (latent → pixels)
├── BUILD.md
└── README.md
```

## Features

- **Wan2.2-1.3B** DiT architecture (Diffusion Transformer)
- **Zero Python** — pure C++23, single ~200KB binary
- **NPU via XRT** — INT8 GEMM on XDNA 2 (reuses `engine/npu/`)
- **CPU fallback** — cache-blocked OpenMP GEMM
- **Flow Matching + DDIM** samplers with CFG
- **Frame output** as PPM + MP4 via ffmpeg

## Reused Infrastructure

| Component | Source | Purpose |
|-----------|--------|---------|
| `I8Ctx` | `engine/npu/src/` | INT8 GEMM on XDNA 2 |
| `attn_omp()` | `engine/npu/src/` | OpenMP attention |
| XRT xclbins | `engine/npu/xclbins/` | MLIR-compiled NPU kernels |

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1 | C++ DiT pipeline + sampler + VAE | ✅ |
| 2 | CPU OpenMP GEMM + attention | ✅ |
| 3 | Weight file format + loader | ✅ |
| 4 | NPU I8Ctx integration | 🔧 needs xclbins |
| 5 | Real T5 text encoder | 📋 |
| 6 | Full convolutional VAE decoder | 📋 |
| 7 | AnimateDiff / ControlNet | 📋 |
