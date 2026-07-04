# Video LoRA

Video generation with LoRA support on Strix Halo. Supports **Wan2.2**, **LTX-Video**, **AnimateDiff**, and **CogVideoX**.

## Models & LoRAs

| Model | LoRA Support | Size | Notes |
|-------|-------------|------|-------|
| **Wan2.2-Fun** | Reward LoRAs, Camera Control LoRAs | 1.3B / 14B | Best open-source T2V |
| **LTX-Video** | IC LoRA detailer (in-context) | 13B | Video-to-video control |
| **AnimateDiff** | Motion + Style LoRAs | 1.5B | Largest community LoRA ecosystem |
| **CogVideoX** | Transformer LoRA | 2B / 5B | Good for coherent motion |
| **Stable Video Diff.** | UNet LoRA | 2.5B | Image-to-video |

## Quick Start

```bash
pip install -e ".[dev]"

# Wan2.2 with reward LoRA
python -m video_lora generate --model wan --prompt "cinematic dolly zoom through cherry blossoms" --lora alibaba-pai/Wan2.2-Fun-Reward-LoRAs

# LTX-Video with detailer LoRA
python -m video_lora generate --model ltx --prompt "astronaut in jungle, muted colors" --lora Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8

# AnimateDiff with motion LoRA
python -m video_lora generate --model animatediff --prompt "panda eating bamboo" --lora guoyww/animatediff-motion-lora-zoom-in
```

## Project Structure

```
video-lora/
├── src/video_lora/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── wan.py          # Wan2.2 pipeline + LoRA
│   │   ├── ltx.py          # LTX-Video pipeline + LoRA
│   │   ├── animatediff.py  # AnimateDiff pipeline + LoRA
│   │   └── cogvideo.py     # CogVideoX pipeline + LoRA
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lora_loader.py  # Unified LoRA loading
│   │   ├── pipeline.py     # Base pipeline abstraction
│   │   └── scheduler.py    # Scheduler configs
│   └── utils/
│       ├── __init__.py
│       └── export.py       # Export to GIF/MP4
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── pyproject.toml
├── .github/workflows/ci.yml
└── README.md
```
