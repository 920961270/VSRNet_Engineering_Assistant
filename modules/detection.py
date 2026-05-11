"""YOLO detection and tracking wrapper for Stage 2."""

import csv
from pathlib import Path


DETECTION_COLUMNS = [
    "frame_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id",
]


def _write_placeholder_csv(detection_csv, warning):
    """Write a placeholder CSV that keeps the pipeline runnable."""
    with detection_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DETECTION_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": 0,
                "class_id": "",
                "class_name": "placeholder_target",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "track_id": "",
            }
        )

    note_path = detection_csv.with_suffix(".txt")
    note_path.write_text(
        f"Stage 2 detection placeholder.\nReason: {warning}\n",
        encoding="utf-8",
    )
    return note_path


def _box_values(boxes):
    """Convert Ultralytics box tensors into plain Python lists."""
    xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
    confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    track_ids = boxes.id.cpu().tolist() if boxes.id is not None else []
    return xyxy, confidences, classes, track_ids


def _run_yolo_results(model, video_path, output_dir):
    """Prefer tracking, then fall back to prediction if tracking is unavailable."""
    project_dir = output_dir / "yolo_runs"
    run_name = "restored_video"

    try:
        print("Running YOLO tracking...")
        return model.track(
            source=str(video_path),
            persist=True,
            stream=True,
            verbose=False,
            save=True,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"WARNING: YOLO tracking failed, trying detection instead: {exc}")
        return model.predict(
            source=str(video_path),
            stream=True,
            verbose=False,
            save=True,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
        )


def run_yolo_detection(video_path=None, output_dir="outputs", yolo_model="yolov8n.pt"):
    """Run YOLO detection/tracking if Ultralytics is available."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_csv = output_dir / "detection_results.csv"

    if not video_path:
        warning = "No video path was provided for YOLO detection."
        print(f"WARNING: {warning}")
        print(
            "Stage 1 placeholder detection used. Real YOLO detection will be connected in Stage 2."
        )
        note_path = _write_placeholder_csv(detection_csv, warning)
        return {
            "detection_csv": str(detection_csv),
            "annotated_output_dir": None,
            "used_placeholder": True,
            "warning": warning,
            "note_path": str(note_path),
        }

    video_path = Path(video_path)
    if not video_path.exists():
        warning = f"Video for YOLO detection does not exist: {video_path}"
        print(f"WARNING: {warning}")
        print(
            "Stage 1 placeholder detection used. Real YOLO detection will be connected in Stage 2."
        )
        note_path = _write_placeholder_csv(detection_csv, warning)
        return {
            "detection_csv": str(detection_csv),
            "annotated_output_dir": None,
            "used_placeholder": True,
            "warning": warning,
            "note_path": str(note_path),
        }

    try:
        from ultralytics import YOLO
    except ImportError:
        warning = "Ultralytics is not installed. Install it to enable YOLO detection/tracking."
        print(f"WARNING: {warning}")
        print(
            "Stage 1 placeholder detection used. Real YOLO detection will be connected in Stage 2."
        )
        note_path = _write_placeholder_csv(detection_csv, warning)
        return {
            "detection_csv": str(detection_csv),
            "annotated_output_dir": None,
            "used_placeholder": True,
            "warning": warning,
            "note_path": str(note_path),
        }

    try:
        model = YOLO(str(yolo_model))
        results = _run_yolo_results(model, video_path, output_dir)

        row_count = 0
        with detection_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DETECTION_COLUMNS)
            writer.writeheader()

            for frame_id, result in enumerate(results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                names = result.names or getattr(model, "names", {})
                xyxy, confidences, classes, track_ids = _box_values(boxes)

                for box_index, coords in enumerate(xyxy):
                    class_id = int(classes[box_index]) if box_index < len(classes) else None
                    class_name = names.get(class_id, str(class_id)) if class_id is not None else ""
                    track_id = track_ids[box_index] if box_index < len(track_ids) else None
                    writer.writerow(
                        {
                            "frame_id": frame_id,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidences[box_index]
                            if box_index < len(confidences)
                            else "",
                            "x1": coords[0],
                            "y1": coords[1],
                            "x2": coords[2],
                            "y2": coords[3],
                            "track_id": track_id,
                        }
                    )
                    row_count += 1

    except Exception as exc:
        warning = f"YOLO detection failed: {exc}"
        print(f"WARNING: {warning}")
        note_path = _write_placeholder_csv(detection_csv, warning)
        return {
            "detection_csv": str(detection_csv),
            "annotated_output_dir": None,
            "used_placeholder": True,
            "warning": warning,
            "note_path": str(note_path),
        }

    print(f"YOLO detection CSV saved to {detection_csv} with {row_count} rows.")
    return {
        "detection_csv": str(detection_csv),
        "annotated_output_dir": str(output_dir / "yolo_runs" / "restored_video"),
        "used_placeholder": False,
        "warning": None,
        "note_path": None,
    }

