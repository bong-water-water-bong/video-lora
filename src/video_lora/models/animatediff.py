"""AnimateDiff pipeline — best community LoRA ecosystem."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from tqdm import tqdm

from ..core.pipeline import VideoPipeline


class AnimateDiffVideo(VideoPipeline):
    """AnimateDiff with motion module + LoRA support."""

    def __init__(
        self,
        model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter = MotionAdapter.from_pretrained(motion_adapter)
        self.pipe = AnimateDiffPipeline.from_pretrained(
            model_id,
            motion_adapter=adapter,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # Memory optimizations
        self.pipe.enable_attention_slicing()
        if device == "cpu":
            self.pipe.to("cpu")
        else:
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe.to(device)

        self.device = device

    def generate(
        self,
        prompt: str,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.7,
        num_frames: int = 16,
        width: int = 640,
        height: int = 480,
        seed: Optional[int] = None,
        output: Optional[Path] = None,
        input_image: Optional[Path] = None,
        progress: bool = True,
        num_inference_steps: int = 25,
    ) -> Path:
        if output is None:
            output = Path(f"animatediff_{abs(hash(prompt))}.gif")

        if lora_path:
            self.load_lora(lora_path, lora_weight)

        generator = torch.Generator().manual_seed(seed) if seed else None

        kwargs = dict(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=7.5,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Progress callback
        if progress:
            pbar = tqdm(total=num_inference_steps, desc="Denoising", unit="step")
            kwargs["callback"] = lambda step, _ts, _latents: pbar.update(1)
            kwargs["callback_steps"] = 1

        frames = self.pipe(**kwargs).frames[0]

        if progress:
            pbar.close()

        export_to_gif(frames, str(output))
        return output

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        from ..core.lora_loader import load_lora_into_pipe
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        self.pipe.unload_lora_weights()
