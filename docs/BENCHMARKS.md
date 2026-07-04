# Benchmarks

## Python Backend (CPU / CUDA)

Run benchmarks locally:
```bash
# Quick benchmark (8 frames, 10 steps, 3 runs)
python -m video_lora benchmark --model wan --frames 8 --steps 10

python -m video_lora benchmark --model animatediff --frames 8 --steps 10

python -m video_lora benchmark --model ltx --frames 4 --steps 10

python -m video_lora benchmark --model cogvideo --frames 8 --steps 10
```

Current results (Strix Halo Radeon 8060S, 32 CUs):

| Model | Device | Resolution | Frames | Steps | Time (s) | FPS |
|-------|--------|-----------|-------|-------|---------|-----|
| Wan2.2-1.3B | CPU (AMD) | 640×480 | 8 | 10 | (TBD) | (TBD) |
| Wan2.2-1.3B | CPU (AMD) | 640×480 | 8 | 50 | (TBD) | (TBD) |
| AnimateDiff | CPU (AMD) | 512×512 | 16 | 25 | (TBD) | (TBD) |
| LTX-Video | CPU (AMD) | 640×480 | 8 | 50 | (TBD) | (TBD) |
| CogVideoX-2B | CPU (AMD) | 640×480 | 8 | 50 | (TBD) | (TBD) |

## Vulkan Backend (GPU)

Benchmarks for the Vulkan compute backend:
```bash
cd vulkan
zig build -Doptimize=ReleaseFast
./zig-out/bin/video-lora-vk --prompt test --frames 4 --steps 5
```

| Model | GPU | Resolution | Frames | Steps | Time (s) | FPS |
|-------|-----|-----------|-------|-------|---------|-----|
| SD1.5 UNet (Vulkan) | Radeon 8060S | 512×512 | 8 | 25 | (TBD) | (TBD) |

## Memory Usage

| Model | Precision | VRAM (load) | VRAM (inference) | System RAM |
|-------|-----------|------------|-----------------|-----------|
| Wan2.2-1.3B | BF16 | ~2.6 GB | ~4 GB | ~1 GB |
| Wan2.2-14B | BF16 | ~28 GB | ~32 GB | ~4 GB |
| LTX-Video | BF16 | ~26 GB | ~30 GB | ~4 GB |
| AnimateDiff | FP16 | ~3 GB | ~4 GB | ~1 GB |
| CogVideoX-2B | BF16 | ~4 GB | ~6 GB | ~1 GB |

Add your results above by running the benchmark commands and editing this file.
