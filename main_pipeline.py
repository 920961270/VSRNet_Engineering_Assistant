"""Command-line pipeline for the VSRNet Engineering Assistant."""

import argparse
import json
from pathlib import Path

from modules.detection import run_yolo_detection
from modules.llm_report import generate_llm_additional_analysis
from modules.llm_report import generate_rule_based_analysis
from modules.metrics import compute_metrics
from modules.rag_retriever import DEFAULT_EMBEDDING_MODEL_PATH, build_rag_index
from modules.rag_retriever import retrieve_project_context
from modules.report_generator import generate_markdown_report
from modules.restoration import DEFAULT_VSRNET_REPO_PATH, run_vsrnet_restoration


STARTER_DOCUMENTS = {
    "vsrnet_summary.txt": """VSRNet is a lightweight video restoration network designed for vibration-degraded monitoring videos.

The project focuses on restoring video affected by physical vibration, rolling-shutter-like artifacts, and motion blur. It is intended for monitoring scenarios where stable visual evidence matters for later perception tasks.

Stage 1 of this assistant does not run real VSRNet inference. Stage 2 attempts to wrap the existing local VSRNet inference script when the input video and checkpoint are available.
""",
    "method_notes.txt": """VSRNet uses a practical multi-frame restoration setting. The original repository includes sliding-window inference for frame sequences and full videos.

The engineering workflow should treat restoration as one part of a larger closed loop: restore degraded video, evaluate visual quality, inspect downstream detection stability, and summarize what the metrics imply.

Vibration intensity in the project is controlled by exposure proportion and displacement amplitudes. Larger beta values usually mean longer motion integration, while larger amplitudes mean stronger visible vibration.
""",
    "evaluation_notes.txt": """Evaluation should include both image-quality metrics and perception-oriented stability metrics.

Useful visual metrics include PSNR and SSIM when paired ground truth is available. Useful downstream metrics include jitter, IoU continuity, confidence variance, FPS, and inference time.

Stage 2 computes feasible metrics from available videos and detection CSV files. Missing values should be reported honestly as unavailable rather than guessed.
""",
    "experiment_logs.txt": """Stage 2 MVP log:

- Restoration uses the existing VSRNet video inference script when the repo, input video, and checkpoint are available.
- Detection uses Ultralytics YOLO when installed.
- Metrics are computed from detection CSV and available videos.
- RAG retrieves local VSRNet project notes from the documents folder.
- The final report is template-based and does not call an LLM API.
""",
}


def ensure_starter_documents(doc_folder):
    """Create starter txt files when the document folder is empty."""
    doc_folder = Path(doc_folder)
    doc_folder.mkdir(parents=True, exist_ok=True)

    existing_documents = list(doc_folder.glob("*.txt"))
    if existing_documents:
        return

    print(f"No documents found in {doc_folder}. Creating starter notes...")
    for file_name, content in STARTER_DOCUMENTS.items():
        file_path = doc_folder / file_name
        file_path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"Created {file_path}")


def _absolute_path(path):
    """Return an absolute path without requiring it to exist."""
    return Path(path).expanduser().resolve(strict=False)


def _validate_nonempty_file(path, label):
    """Validate that a reused file exists and has content."""
    if not path:
        raise ValueError(f"{label} path was not provided.")

    file_path = _absolute_path(path)
    if not file_path.exists():
        raise ValueError(f"{label} does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"{label} is not a file: {file_path}")
    if file_path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {file_path}")
    return file_path


def _load_metrics_json(path):
    """Load and validate a reused metrics JSON file."""
    metrics_path = _validate_nonempty_file(path, "metrics JSON")
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8")), metrics_path
    except json.JSONDecodeError as exc:
        raise ValueError(f"metrics JSON is not valid JSON: {metrics_path} ({exc})") from exc


def _fail_reuse(message):
    """Stop the pipeline for invalid explicit skip/reuse requests."""
    print(f"ERROR: {message}")
    raise SystemExit(2)


def _metadata_paths(input_video, gt_video, restored_video, detection_csv, metrics_json=None):
    """Build path metadata for the metric summary."""
    metadata = {}
    if input_video:
        metadata["degraded_video_path"] = str(_absolute_path(input_video))
    if gt_video:
        metadata["gt_video_path"] = str(_absolute_path(gt_video))
    if restored_video:
        metadata["restored_video_path"] = str(_absolute_path(restored_video))
    if detection_csv:
        metadata["detection_csv_path"] = str(_absolute_path(detection_csv))
    if metrics_json:
        metadata["metrics_json_path"] = str(_absolute_path(metrics_json))
    return metadata


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the VSRNet Engineering Analysis Assistant."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Engineering question to answer with project RAG context.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional degraded input video path.",
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Optional ground-truth video path for aligned PSNR/SSIM.",
    )
    parser.add_argument(
        "--vsrnet-checkpoint",
        default="models/vsrnet/best.pth",
        help="VSRNet checkpoint path. Missing checkpoints trigger placeholder restoration.",
    )
    parser.add_argument(
        "--vsrnet-repo",
        default=str(DEFAULT_VSRNET_REPO_PATH),
        help="Local VSRNet repository path containing infer_vsrnet_video.py.",
    )
    parser.add_argument(
        "--embedding-model",
        default=str(DEFAULT_EMBEDDING_MODEL_PATH),
        help="Local sentence-transformers model path.",
    )
    parser.add_argument(
        "--documents",
        default="documents",
        help="Folder containing VSRNet txt documents for RAG.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Deprecated Stage 1 option. Metrics are computed in Stage 2.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model name or path for Ultralytics.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output folder for restored video, detection CSV, metrics, and report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of RAG chunks to retrieve.",
    )
    parser.add_argument(
        "--use-llm-report",
        action="store_true",
        help="Enable an optional LLM-generated engineering analysis section.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "ollama", "none"],
        default="none",
        help="LLM provider for --use-llm-report.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional LLM model name, such as gpt-4o-mini, llama3.1, qwen2.5, or deepseek-r1.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for OpenAI or Ollama LLM HTTP requests.",
    )
    parser.add_argument(
        "--restored-video",
        default=None,
        help="Existing restored video to reuse. If valid, VSRNet restoration is skipped.",
    )
    parser.add_argument(
        "--skip-restoration",
        action="store_true",
        help="Skip VSRNet restoration. Requires --restored-video.",
    )
    parser.add_argument(
        "--detection-csv",
        default=None,
        help="Existing detection_results.csv to reuse. If valid, YOLO detection is skipped.",
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip YOLO detection. Requires --detection-csv.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Existing metrics.json to reuse with --skip-metrics.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip metrics computation. Requires --metrics-json.",
    )
    return parser.parse_args()


def main():
    """Run the full assistant pipeline."""
    args = parse_args()

    output_dir = _absolute_path(args.output)
    doc_folder = Path(args.documents)

    print("\n=== VSRNet Engineering Assistant: Stage 2 ===")
    print(f"Question: {args.question}")

    if args.metrics:
        print("Note: --metrics is kept for Stage 1 compatibility but Stage 2 recomputes metrics.")

    ensure_starter_documents(doc_folder)

    print("\n[1/5] VSRNet restoration...")
    if args.skip_restoration and not args.restored_video:
        _fail_reuse("--skip-restoration requires --restored-video.")

    if args.restored_video:
        try:
            restored_video_path = _validate_nonempty_file(
                args.restored_video,
                "restored video",
            )
        except ValueError as exc:
            _fail_reuse(str(exc))

        print(f"VSRNet restoration: skipped, reused existing restored video: {restored_video_path}")
        restoration_outputs = {
            "restored_video": str(restored_video_path),
            "restored_frames_dir": None,
            "used_placeholder": False,
            "skipped": True,
            "reused": True,
            "warning": None,
            "input_video": str(_absolute_path(args.input)) if args.input else None,
            "model_path": str(_absolute_path(args.vsrnet_checkpoint))
            if args.vsrnet_checkpoint
            else None,
            "note_path": None,
            "restoration_total_time_sec": None,
            "inference_time_ms": None,
            "restored_frame_count": None,
        }
    else:
        restoration_outputs = run_vsrnet_restoration(
            input_video=args.input,
            model_path=args.vsrnet_checkpoint,
            output_dir=output_dir,
            repo_path=args.vsrnet_repo,
        )

    print("\n[2/5] YOLO detection/tracking...")
    if args.skip_detection and not args.detection_csv:
        _fail_reuse("--skip-detection requires --detection-csv.")

    if args.detection_csv:
        try:
            detection_csv_path = _validate_nonempty_file(
                args.detection_csv,
                "detection CSV",
            )
        except ValueError as exc:
            _fail_reuse(str(exc))

        print(f"YOLO detection/tracking: skipped, reused existing detection CSV: {detection_csv_path}")
        detection_outputs = {
            "detection_csv": str(detection_csv_path),
            "annotated_output_dir": None,
            "used_placeholder": False,
            "skipped": True,
            "reused": True,
            "warning": None,
            "note_path": None,
        }
    else:
        detection_outputs = run_yolo_detection(
            video_path=restoration_outputs["restored_video"],
            output_dir=output_dir,
            yolo_model=args.yolo_model,
        )

    print("\n[3/5] Metrics...")
    if args.skip_metrics and not args.metrics_json:
        _fail_reuse("--skip-metrics requires --metrics-json.")

    metrics_outputs = None
    if args.skip_metrics:
        try:
            metrics, metrics_json_path = _load_metrics_json(args.metrics_json)
        except ValueError as exc:
            _fail_reuse(str(exc))
        print(f"Metrics: skipped, loaded existing metrics JSON: {metrics_json_path}")
        metrics.update(
            _metadata_paths(
                args.input,
                args.gt,
                restoration_outputs["restored_video"],
                detection_outputs["detection_csv"],
                metrics_json_path,
            )
        )
        metrics["metrics_source"] = "reused"
        metrics_outputs = {
            "skipped": True,
            "reused": True,
            "metrics_json": str(metrics_json_path),
            "metrics_csv": None,
            "warning": None,
        }
    else:
        metrics = compute_metrics(
            gt_video=args.gt,
            degraded_video=args.input,
            restored_video=restoration_outputs["restored_video"],
            detection_csv=detection_outputs["detection_csv"],
            output_dir=output_dir,
            restoration_outputs=restoration_outputs,
        )
        metrics.update(
            _metadata_paths(
                args.input,
                args.gt,
                restoration_outputs["restored_video"],
                detection_outputs["detection_csv"],
                output_dir / "metrics.json",
            )
        )
        metrics["metrics_source"] = "computed"
        metrics_outputs = {
            "skipped": False,
            "reused": False,
            "metrics_json": str(output_dir / "metrics.json"),
            "metrics_csv": str(output_dir / "metrics.csv"),
            "warning": None,
        }

    print("\n[4/5] Building RAG index and retrieving context...")
    print("RAG retrieval: real pipeline used")
    model, index, chunks = build_rag_index(
        doc_folder=doc_folder,
        model_path=args.embedding_model,
    )
    retrieved_contexts = retrieve_project_context(
        question=args.question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=args.top_k,
    )

    print("\nRetrieved contexts:")
    for rank, context in enumerate(retrieved_contexts, start=1):
        print(
            f"- {rank}. {context['source']} "
            f"chunk {context['chunk_id']} "
            f"score {context['score']:.4f}"
        )

    print("\n[Analysis] Generating rule-based engineering analysis...")
    rule_based_analysis = generate_rule_based_analysis(
        question=args.question,
        metrics=metrics,
        restoration_outputs=restoration_outputs,
        detection_outputs=detection_outputs,
    )

    llm_additional_analysis = None
    if args.use_llm_report:
        print("\n[LLM] Generating optional LLM additional analysis...")
        llm_additional_analysis, llm_warning = generate_llm_additional_analysis(
            question=args.question,
            metrics=metrics,
            retrieved_contexts=retrieved_contexts,
            rule_based_analysis=rule_based_analysis,
            restoration_outputs=restoration_outputs,
            detection_outputs=detection_outputs,
            provider=args.llm_provider,
            model=args.llm_model,
            output_dir=output_dir,
            llm_timeout=args.llm_timeout,
        )
        if llm_warning:
            print(f"WARNING: {llm_warning}")

    print("\n[5/5] Generating report...")
    print("Report generation: real pipeline used")
    report_path = generate_markdown_report(
        question=args.question,
        metrics=metrics,
        retrieved_contexts=retrieved_contexts,
        output_dir=output_dir,
        restoration_outputs=restoration_outputs,
        detection_outputs=detection_outputs,
        metrics_outputs=metrics_outputs,
        rule_based_analysis=rule_based_analysis,
        llm_additional_analysis=llm_additional_analysis,
    )

    print("\nPipeline complete.")
    print("Output paths:")
    print(f"- Restored video: {restoration_outputs['restored_video']}")
    print(f"- Restored frames dir: {restoration_outputs['restored_frames_dir']}")
    print(f"- Detection CSV: {detection_outputs['detection_csv']}")
    print(f"- Metrics JSON: {metrics_outputs['metrics_json']}")
    print(f"- Metrics CSV: {metrics_outputs['metrics_csv']}")
    print(f"- Engineering report: {report_path}")


if __name__ == "__main__":
    main()
