"""CogVideoX pipeline — transformer-based video generation with coherent motion."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from ..core.pipeline import VideoPipeline


class CogVideoXVideo(VideoPipeline):
    """CogVideoX with transformer LoRA support (2B / 5B variants)."""

    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-2b",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

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
        num_inference_steps: int = 50,
    ) -> Path:
        if output is None:
            output = Path(f"cogvideo_{abs(hash(prompt))}.mp4")

        if lora_path:
            self.load_lora(lora_path, lora_weight)

        generator = torch.Generator().manual_seed(seed) if seed else None

        kwargs = dict(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=6.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        if input_image:
            from PIL import Image
            kwargs["image"] = Image.open(input_image)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=num_inference_steps, desc="Denoising", unit="step")
            kwargs["callback"] = lambda step, _ts, _latents: pbar.update(1)
            kwargs["callback_steps"] = 1

        video = self.pipe(**kwargs).frames[0]

        if progress:
            pbar.close()
        export_to_video(video, str(output), fps=8)
        return output

    def load_lora(self, lora_path: str, weight: float = 0.7) -> None:
        from ..core.lora_loader import load_lora_into_pipe
        load_lora_into_pipe(self.pipe, lora_path, weight)

    def unload_lora(self) -> None:
        self.pipe.unload_lora_weights()
