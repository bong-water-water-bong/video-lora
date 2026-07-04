"""CLI entry point for video-lora."""

import argparse
import time
from pathlib import Path
from typing import Optional


def register_models():
    """Lazy-import model backends so CLI stays fast."""
    models = {}

    try:
        from .models.animatediff import AnimateDiffVideo
        models["animatediff"] = lambda: AnimateDiffVideo()
    except ImportError:
        pass

    try:
        from .models.wan import WanVideo
        models["wan"] = lambda: WanVideo()
    except ImportError:
        pass

    try:
        from .models.ltx import LTXVideo
        models["ltx"] = lambda: LTXVideo()
    except ImportError:
        pass

    try:
        from .models.cogvideo import CogVideoXVideo
        models["cogvideo"] = lambda: CogVideoXVideo()
    except ImportError:
        pass

    return models


MODEL_DESCRIPTIONS = {
    "wan": "Wan2.2 (1.3B / 14B) — best open-source T2V, reward LoRAs + camera control",
    "ltx": "LTX-Video (13B) — video-to-video control, IC LoRA detailer",
    "animatediff": "AnimateDiff (1.5B base) — largest community LoRA ecosystem (1000+)",
    "cogvideo": "CogVideoX (2B / 5B) — transformer-based, coherent motion",
}


def main():
    parser = argparse.ArgumentParser(
        description="Video LoRA Generator — Wan2.2, LTX-Video, AnimateDiff, CogVideoX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Text-to-video (Wan2.2)\n"
            "  video-lora generate --model wan --prompt 'cinematic dolly zoom'\n\n"
            "  # Image-to-video (Wan2.2)\n"
            "  video-lora generate --model wan --prompt 'a cat walking' --input-image input.png\n\n"
            "  # AnimateDiff with LoRA\n"
            "  video-lora generate --model animatediff --prompt 'cat walking' --lora ./motion.safetensors\n\n"
            "  # Longer video\n"
            "  video-lora generate --model ltxi --prompt 'waterfall' --frames 32 --steps 30\n\n"
            "  # Benchmark\n"
            "  video-lora benchmark --model wan --frames 8 --steps 10\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # `generate` subcommand
    gen = sub.add_parser("generate", help="Generate a video")
    gen.add_argument(
        "--model", "-m", required=True,
        choices=["wan", "ltx", "animatediff", "cogvideo"],
        help="Model backend",
    )
    gen.add_argument("--prompt", "-p", required=True, help="Text prompt")
    gen.add_argument("--input-image", "-i", type=Path,
                     help="Input image for image-to-video generation")
    gen.add_argument("--lora", help="LoRA path (HF repo ID or local file)")
    gen.add_argument("--lora-weight", type=float, default=0.7,
                     help="LoRA merge weight (default: 0.7)")
    gen.add_argument("--frames", type=int, default=16,
                     help="Number of frames to generate (default: 16)")
    gen.add_argument("--steps", type=int, default=50,
                     help="Number of denoising steps (default: 50)")
    gen.add_argument("--width", type=int, default=640,
                     help="Output width (default: 640)")
    gen.add_argument("--height", type=int, default=480,
                     help="Output height (default: 480)")
    gen.add_argument("--seed", type=int, help="Random seed for reproducibility")
    gen.add_argument("--output", "-o", type=Path,
                     help="Output path (auto-generated if not specified)")
    gen.add_argument("--no-progress", action="store_true",
                     help="Disable progress bars")
    gen.add_argument("--no-lora-fuse", action="store_true",
                     help="Don't fuse LoRA — load as adapter only")

    # `list-models` subcommand
    sub.add_parser("list-models", help="List available models and LoRAs")

    # `benchmark` subcommand
    bench = sub.add_parser("benchmark", help="Benchmark generation speed")
    bench.add_argument(
        "--model", "-m", required=True,
        choices=["wan", "ltx", "animatediff", "cogvideo"],
        help="Model backend to benchmark",
    )
    bench.add_argument("--prompt", default="a cat walking, cinematic lighting",
                       help="Test prompt (default: 'a cat walking, cinematic lighting')")
    bench.add_argument("--frames", type=int, default=8,
                       help="Number of frames (default: 8)")
    bench.add_argument("--steps", type=int, default=10,
                       help="Number of inference steps (default: 10, low for speed)")
    bench.add_argument("--runs", type=int, default=3,
                       help="Number of benchmark runs (default: 3)")

    args = parser.parse_args()

    if args.command == "list-models":
        print("Available models and their top LoRAs:")
        print()
        for key, desc in MODEL_DESCRIPTIONS.items():
            print(f"  {key:<14s} {desc}")
        print()
        print("Wan2.2 LoRA sources:")
        print("  - alibaba-pai/Wan2.2-Fun-Reward-LoRAs (HuggingFace)")
        print("  - alibaba-pai/Wan2.2-Fun-A14B-Control-Camera (HuggingFace)")
        print("LTX-Video LoRA sources:")
        print("  - Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8 (HuggingFace)")
        print("AnimateDiff LoRA sources:")
        print("  - 1000+ community LoRAs on CivitAI / HuggingFace")
        print("CogVideoX LoRA sources:")
        print("  - THUDM/CogVideoX-2b-LoRA (HuggingFace)")
        return

    if args.command == "benchmark":
        return _run_benchmark(args)

    # --- Generate ---
    models = register_models()
    if args.model not in models:
        print(f"Model '{args.model}' not available. Install extras:")
        model_extras = {
            "wan": "",
            "ltx": "ltx",
            "animatediff": "",
            "cogvideo": "",
        }
        extra = model_extras.get(args.model, args.model)
        if extra:
            print(f"  pip install -e '.[{extra}]'")
        else:
            print("  pip install -e .")
        return

    pipe = models[args.model]()
    output = pipe.generate(
        prompt=args.prompt,
        lora_path=args.lora,
        lora_weight=args.lora_weight,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output=args.output,
        input_image=args.input_image,
        progress=not args.no_progress,
        num_inference_steps=args.steps,
    )
    print(f"\n✓ Output saved to: {output}")
    print(f"  {args.frames} frames at {args.width}×{args.height}")


def _run_benchmark(args):
    """Run speed benchmark for a model."""
    import torch

    models = register_models()
    if args.model not in models:
        print(f"Model '{args.model}' not available.")
        return

    print(f"Benchmarking {args.model}...")
    print(f"  Frames: {args.frames}, Steps: {args.steps}, Runs: {args.runs}")
    print(f"  Prompt: '{args.prompt}'")
    print()

    pipe = models[args.model]()

    # Warmup
    print("  Warming up...")
    pipe.generate(
        prompt=args.prompt,
        num_frames=args.frames,
        num_inference_steps=min(args.steps, 2),
        progress=False,
    )

    timings = []
    for run in range(args.runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        pipe.generate(
            prompt=args.prompt,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            progress=False,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"  Run {run + 1}: {elapsed:.2f}s ({args.frames / elapsed:.2f} fps)")

    avg = sum(timings) / len(timings)
    print(f"\n  Average: {avg:.2f}s ({args.frames / avg:.2f} fps)")
    print(f"  Best:    {min(timings):.2f}s ({args.frames / min(timings):.2f} fps)")
    print(f"  {args.steps} steps × {args.frames} frames = {args.steps * args.frames} total denoising steps")


if __name__ == "__main__":
    main()
