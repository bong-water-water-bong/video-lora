"""OpenAI-compatible API server for video-lora generation.

Run:
  python -m video_lora.server --port 8082

Then:
  curl -X POST http://localhost:8082/v1/video/generations \
    -H "Content-Type: application/json" \
    -d '{"model":"wan","prompt":"a cat walking","num_frames":8}'
"""

import argparse
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise ImportError(
        "API server requires: pip install fastapi uvicorn pydantic"
    )

from .cli import register_models

app = FastAPI(title="Video LoRA API", version="0.2.0")

# In-memory job store (consider redis for production)
jobs: dict[str, dict] = {}

OUTPUT_DIR = Path("/tmp/video-lora-outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GenerationRequest(BaseModel):
    model: str = Field(..., description="Model name: wan, ltx, animatediff, cogvideo")
    prompt: str = Field(..., description="Text prompt")
    input_image: Optional[str] = Field(None, description="Path or URL to input image")
    lora: Optional[str] = Field(None, description="LoRA path or HF repo ID")
    lora_weight: float = Field(0.7, description="LoRA merge weight")
    num_frames: int = Field(16, description="Number of frames", ge=1, le=81)
    width: int = Field(640, description="Output width", ge=64, le=1920)
    height: int = Field(480, description="Output height", ge=64, le=1920)
    num_inference_steps: int = Field(50, description="Denoising steps", ge=1, le=200)
    seed: Optional[int] = Field(None, description="Random seed")


class GenerationResponse(BaseModel):
    id: str
    status: str
    output: Optional[str] = None
    message: Optional[str] = None


@app.get("/v1/models")
def list_models():
    """List available video generation models."""
    models = register_models()
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "video-lora",
            }
            for name in models
        ],
    }


@app.post("/v1/video/generations", response_model=GenerationResponse)
async def create_generation(req: GenerationRequest):
    """Create a video generation job."""
    job_id = f"vg-{uuid.uuid4().hex[:12]}"

    models = register_models()
    if req.model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' not available. Choose from: {list(models.keys())}",
        )

    pipe = models[req.model]()
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    # Run generation (blocking — swap to background task for production)
    try:
        input_image_path = Path(req.input_image) if req.input_image else None
        if req.input_image and not input_image_path.exists():
            raise HTTPException(status_code=400, detail=f"Input image not found: {req.input_image}")

        result = pipe.generate(
            prompt=req.prompt,
            lora_path=req.lora,
            lora_weight=req.lora_weight,
            num_frames=req.num_frames,
            width=req.width,
            height=req.height,
            seed=req.seed,
            output=output_path,
            input_image=input_image_path,
            progress=False,
            num_inference_steps=req.num_inference_steps,
        )

        jobs[job_id] = {
            "id": job_id,
            "status": "completed",
            "output": str(result),
        }

        return GenerationResponse(
            id=job_id,
            status="completed",
            output=str(result),
        )
    except Exception as e:
        jobs[job_id] = {"id": job_id, "status": "failed", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/video/generations/{job_id}")
def get_generation(job_id: str):
    """Get generation status and result."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return GenerationResponse(
        id=job["id"],
        status=job["status"],
        output=job.get("output"),
        message=job.get("error"),
    )


@app.get("/v1/video/generations/{job_id}/output")
def get_generation_output(job_id: str):
    """Download the generated video file."""
    if job_id not in jobs or "output" not in jobs[job_id]:
        raise HTTPException(status_code=404, detail="Output not found")
    output_path = Path(jobs[job_id]["output"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(str(output_path), media_type="video/mp4")


def main():
    parser = argparse.ArgumentParser(description="Video LoRA API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"Video LoRA API server starting on http://{args.host}:{args.port}")
    print(f"  POST /v1/video/generations — create generation")
    print(f"  GET  /v1/video/generations/{{id}} — get status")
    print(f"  GET  /v1/video/generations/{{id}}/output — download video")
    print(f"  GET  /v1/models — list models")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
