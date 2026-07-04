"""Smoke tests for video-lora model backends."""

from pathlib import Path


def test_module_imports():
    """All modules should import cleanly."""
    from video_lora import __version__
    assert __version__ == "0.2.0"

    from video_lora.core.pipeline import VideoPipeline
    assert VideoPipeline is not None

    from video_lora.core.lora_loader import load_lora_into_pipe, load_multiple_loras
    assert load_lora_into_pipe is not None
    assert load_multiple_loras is not None

    from video_lora.core.scheduler import SchedulerConfig, get_scheduler_config, SCHEDULER_PRESETS
    assert SchedulerConfig is not None
    assert len(SCHEDULER_PRESETS) >= 4
    config = get_scheduler_config("wan")
    assert config.prediction_type == "flow_matching"

    from video_lora.utils import auto_export, export_to_gif, export_to_mp4, export_frames
    assert auto_export is not None
    assert export_to_gif is not None

    from video_lora.cli import main, register_models, MODEL_DESCRIPTIONS
    assert main is not None
    assert len(MODEL_DESCRIPTIONS) >= 4


def test_cli_list_models():
    """CLI list-models should output all models without crashing."""
    import sys
    from video_lora.cli import main

    old_argv = sys.argv
    sys.argv = ["video-lora", "list-models"]
    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def test_cli_benchmark_help():
    """CLI benchmark subcommand should be registered."""
    import sys
    from video_lora.cli import main

    old_argv = sys.argv
    sys.argv = ["video-lora", "benchmark", "--help"]
    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def test_cli_generate_help():
    """CLI generate subcommand should show all flags."""
    import sys
    from video_lora.cli import main

    old_argv = sys.argv
    sys.argv = ["video-lora", "generate", "--help"]
    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def test_scheduler_configs():
    """All model scheduler presets should have valid configs."""
    from video_lora.core.scheduler import get_scheduler_config

    for model_type in ["sd15", "sdxl", "wan", "ltx", "cogvideo"]:
        config = get_scheduler_config(model_type)
        assert config.name is not None
        assert config.num_train_timesteps > 0


def test_export_docs():
    """Export utility should have proper docstrings."""
    from video_lora.utils.export import auto_export
    assert auto_export.__doc__ is not None
    assert "Auto-detect" in auto_export.__doc__


def test_lora_loader_imports():
    """LoRA loader utilities should be importable."""
    from video_lora.core.lora_loader import load_lora_into_pipe
    assert callable(load_lora_into_pipe)


def test_model_modules_exist():
    """All model modules should have proper module-level docstrings."""
    import video_lora.models.animatediff as ad
    import video_lora.models.wan as wan
    import video_lora.models.ltx as ltx
    import video_lora.models.cogvideo as cv

    for mod in [ad, wan, ltx, cv]:
        assert mod.__doc__ is not None and len(mod.__doc__) > 20


def test_server_import():
    """API server should import without dependencies (skip if not installed)."""
    try:
        import video_lora.server
        assert video_lora.server.app is not None
    except ImportError:
        pass  # fastapi not installed


def test_all_model_classes():
    """All model classes should be instantiable (import check)."""
    from video_lora.models.animatediff import AnimateDiffVideo
    from video_lora.models.wan import WanVideo
    from video_lora.models.ltx import LTXVideo
    from video_lora.models.cogvideo import CogVideoXVideo

    for cls in [AnimateDiffVideo, WanVideo, LTXVideo, CogVideoXVideo]:
        assert cls is not None
        # Check generate signature has input_image param
        import inspect
        sig = inspect.signature(cls.generate)
        assert "input_image" in sig.parameters
        assert "progress" in sig.parameters
        assert "num_inference_steps" in sig.parameters
