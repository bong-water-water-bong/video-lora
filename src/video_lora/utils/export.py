"""Export utilities — GIF, MP4, frame inspection."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


def export_to_gif(
    frames: List[Image.Image],
    output_path: Union[str, Path],
    fps: int = 8,
    loop: int = 0,
) -> Path:
    """Export a list of PIL frames to an animated GIF."""
    output_path = Path(output_path)
    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )
    return output_path


def export_to_mp4(
    frames: List[Image.Image],
    output_path: Union[str, Path],
    fps: int = 8,
    pix_fmt_in: str = "rgb24",
) -> Path:
    """Export frames to MP4 via OpenCV."""
    import cv2

    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")

    frame_array = [np.array(f) for f in frames]
    h, w, _ = frame_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frame_array:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


def export_frames(
    frames: List[Image.Image],
    output_dir: Union[str, Path],
    prefix: str = "frame",
    fmt: str = "png",
) -> Path:
    """Export frames as individual image files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(output_dir / f"{prefix}_{i:04d}.{fmt}")
    return output_dir


def auto_export(
    frames: List[Image.Image],
    output_path: Union[str, Path],
    fps: int = 8,
) -> Path:
    """Auto-detect format from file extension and export."""
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    if ext == ".gif":
        return export_to_gif(frames, output_path, fps=fps)
    elif ext == ".mp4":
        return export_to_mp4(frames, output_path, fps=fps)
    else:
        # Default to GIF
        if ext:
            output_path = output_path.with_suffix(".gif")
        else:
            output_path = output_path.with_suffix(".gif")
        return export_to_gif(frames, output_path, fps=fps)
