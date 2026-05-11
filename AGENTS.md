# Agent Rules

This project is Stage 1 of the VSRNet Engineering Analysis Assistant.

- Do not retrain VSRNet.
- Do not rewrite the VSRNet architecture.
- Prefer wrappers around existing code.
- Use placeholders if heavy components are missing.
- Avoid TensorFlow.
- Use `pathlib` and `argparse`.
- Keep code beginner-friendly and readable.
- Keep Stage 1 focused on runnable pipeline structure, RAG retrieval, metrics loading, and template report generation.
- For Stage 2, wrap the existing `infer_vsrnet_video.py` script when possible.
- Keep fallback behavior robust when checkpoints, videos, YOLO, or optional metrics dependencies are missing.
- Keep the Stage 1 command compatible: `python main_pipeline.py --question "What problem does VSRNet solve?"`.
