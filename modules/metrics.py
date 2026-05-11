"""Metrics loading and computation for the VSRNet assistant."""

import csv
import json
import math
from pathlib import Path

import numpy as np


SAMPLE_METRICS = {
    "video_name": "sample_vibration_video.mp4",
    "model": "VSRNet",
    "vibration_level": "high",
    "psnr": None,
    "ssim": None,
    "jitter": None,
    "iou_continuity": None,
    "confidence_variance": None,
    "fps": None,
    "inference_time_ms": None,
    "note": "Sample metrics generated for Stage 1 MVP. Replace with real experiment results later.",
}


def load_or_create_metrics(metrics_path=None, output_dir="outputs"):
    """Load a metrics JSON file or create sample metrics for Stage 1 compatibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"

    source_path = Path(metrics_path) if metrics_path else None
    if source_path and source_path.exists():
        print(f"Loading metrics from {source_path}...")
        metrics = json.loads(source_path.read_text(encoding="utf-8"))
    else:
        print("No metrics file found. Creating sample Stage 1 metrics...")
        metrics = dict(SAMPLE_METRICS)

    _save_metrics(metrics, output_dir)
    print(f"Metrics saved to {output_path}")
    return metrics


def _safe_float(value):
    """Convert a CSV value to float if possible."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    """Convert a CSV value to int if possible."""
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_value(row, names):
    """Read the first non-empty value from a row using possible column names."""
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value

    lower_row = {key.lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lower_row.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _infer_vibration_level(video_path):
    """Infer vibration level from the input video filename."""
    if not video_path:
        return None

    name = Path(video_path).name.lower()
    if "low" in name:
        return "low"
    if "mid" in name or "middle" in name:
        return "mid"
    if "high" in name:
        return "high"
    return None


def _video_info(video_path, label, warnings):
    """Read basic video metadata with OpenCV."""
    metrics = {
        f"{label}_frame_count": None,
        f"{label}_fps": None,
        f"{label}_duration_sec": None,
    }

    if not video_path:
        return metrics

    video_path = Path(video_path)
    if not video_path.exists():
        warnings.append(f"{label} video does not exist: {video_path}")
        return metrics

    try:
        import cv2
    except ImportError:
        warnings.append("OpenCV is not installed, so video metadata could not be read.")
        return metrics

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        warnings.append(f"Could not open {label} video: {video_path}")
        return metrics

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    metrics[f"{label}_frame_count"] = frame_count if frame_count > 0 else None
    metrics[f"{label}_fps"] = fps if fps > 0 else None
    if frame_count > 0 and fps > 0:
        metrics[f"{label}_duration_sec"] = frame_count / fps

    return metrics


def _load_detection_rows(detection_csv, warnings):
    """Load detection CSV rows with flexible column names."""
    if not detection_csv:
        warnings.append("No detection CSV was provided.")
        return []

    detection_csv = Path(detection_csv)
    if not detection_csv.exists():
        warnings.append(f"Detection CSV does not exist: {detection_csv}")
        return []

    detections = []
    with detection_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            class_name = (row.get("class_name") or "").strip()
            if class_name == "placeholder_target":
                continue

            frame_id = _safe_int(_first_value(row, ["frame_id", "frame", "frame_index"]))
            confidence = _safe_float(_first_value(row, ["confidence", "conf"]))
            x1 = _safe_float(_first_value(row, ["x1", "xmin"]))
            y1 = _safe_float(_first_value(row, ["y1", "ymin"]))
            x2 = _safe_float(_first_value(row, ["x2", "xmax"]))
            y2 = _safe_float(_first_value(row, ["y2", "ymax"]))
            track_id = _first_value(row, ["track_id", "id"])
            if track_id is not None:
                track_id = str(track_id).strip()
                if track_id == "" or track_id.lower() in {"none", "nan", "null"}:
                    track_id = None

            detections.append(
                {
                    "frame_id": frame_id,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                    if None not in (x1, y1, x2, y2)
                    else None,
                    "track_id": track_id,
                }
            )

    return detections


def _detection_metrics(detections):
    """Compute simple statistics from parsed YOLO detection rows."""
    metrics = {
        "confidence_mean": None,
        "confidence_variance": None,
        "detection_count": len(detections),
        "detected_frame_count": 0,
    }

    confidences = [
        detection["confidence"]
        for detection in detections
        if detection["confidence"] is not None
    ]
    detected_frames = {
        detection["frame_id"]
        for detection in detections
        if detection["frame_id"] is not None
    }

    metrics["detected_frame_count"] = len(detected_frames)
    if confidences:
        confidence_array = np.asarray(confidences, dtype="float32")
        metrics["confidence_mean"] = float(np.mean(confidence_array))
        metrics["confidence_variance"] = float(np.var(confidence_array))

    return metrics


def _valid_box(box):
    """Check whether a detection box is usable."""
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def _center(box):
    """Return center point for a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(box_a, box_b):
    """Compute IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return None
    return inter_area / union


def _select_tracked_detections(detections):
    """Select detections for motion continuity metrics."""
    valid_detections = [
        detection
        for detection in detections
        if detection["frame_id"] is not None and _valid_box(detection["box"])
    ]
    if not valid_detections:
        return [], "none"

    tracked = [detection for detection in valid_detections if detection["track_id"]]
    if tracked:
        by_track = {}
        for detection in tracked:
            by_track.setdefault(detection["track_id"], []).append(detection)

        best_track_id, best_track_detections = max(
            by_track.items(),
            key=lambda item: len({d["frame_id"] for d in item[1]}),
        )
        by_frame = {}
        for detection in best_track_detections:
            frame_id = detection["frame_id"]
            current = by_frame.get(frame_id)
            if current is None or (detection["confidence"] or 0) > (
                current["confidence"] or 0
            ):
                by_frame[frame_id] = detection
        return [by_frame[key] for key in sorted(by_frame)], f"track_id={best_track_id}"

    by_frame = {}
    for detection in valid_detections:
        frame_id = detection["frame_id"]
        current = by_frame.get(frame_id)
        if current is None or (detection["confidence"] or 0) > (
            current["confidence"] or 0
        ):
            by_frame[frame_id] = detection
    return [by_frame[key] for key in sorted(by_frame)], "highest-confidence-per-frame"


def _trajectory_metrics(detections, warnings):
    """Compute jitter and IoU continuity from selected detection boxes."""
    metrics = {"jitter": None, "iou_continuity": None}
    selected, selection_method = _select_tracked_detections(detections)

    displacements = []
    ious = []
    for previous, current in zip(selected, selected[1:]):
        if current["frame_id"] != previous["frame_id"] + 1:
            continue

        prev_center = _center(previous["box"])
        curr_center = _center(current["box"])
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        displacements.append(float(math.sqrt(dx * dx + dy * dy)))

        iou_value = _iou(previous["box"], current["box"])
        if iou_value is not None:
            ious.append(float(iou_value))

    if displacements:
        metrics["jitter"] = float(np.mean(np.asarray(displacements, dtype="float32")))
        print(
            f"Computed jitter from {len(displacements)} consecutive pairs using {selection_method}."
        )
    else:
        warnings.append(
            "Jitter could not be computed because no valid consecutive detections were found."
        )

    if ious:
        metrics["iou_continuity"] = float(np.mean(np.asarray(ious, dtype="float32")))
        print(
            f"Computed IoU continuity from {len(ious)} consecutive pairs using {selection_method}."
        )
    else:
        warnings.append(
            "IoU continuity could not be computed because no valid consecutive boxes were found."
        )

    return metrics


def _resize_restored_to_gt(gt_frame, restored_frame, warnings, state):
    """Resize restored frame to GT size when dimensions differ."""
    if gt_frame.ndim != restored_frame.ndim:
        warnings.append(
            "GT and restored frames have different channel layouts, so PSNR/SSIM were skipped."
        )
        return None

    if gt_frame.ndim == 3 and gt_frame.shape[2] != restored_frame.shape[2]:
        warnings.append(
            "GT and restored frames have different channel counts, so PSNR/SSIM were skipped."
        )
        return None

    if gt_frame.shape[:2] == restored_frame.shape[:2]:
        return restored_frame

    try:
        import cv2
    except ImportError:
        warnings.append("OpenCV is not installed, so restored frames could not be resized.")
        return None

    if not state["resize_warning_printed"]:
        warning = (
            "Restored frames were resized to match GT frame size before PSNR/SSIM computation."
        )
        print(f"WARNING: {warning}")
        warnings.append(warning)
        state["resize_warning_printed"] = True

    gt_height, gt_width = gt_frame.shape[:2]
    return cv2.resize(restored_frame, (gt_width, gt_height))


def _ssim_data_range(gt_frame, restored_frame):
    """Choose an SSIM data range based on frame dtype and value scale."""
    if gt_frame.dtype == np.uint8 and restored_frame.dtype == np.uint8:
        return 255

    max_value = max(float(np.max(gt_frame)), float(np.max(restored_frame)))
    min_value = min(float(np.min(gt_frame)), float(np.min(restored_frame)))
    if min_value >= 0.0 and max_value <= 1.0:
        return 1.0

    return 255


def _safe_ssim_win_size(frame_shape):
    """Return a safe odd SSIM window size, or None if the frame is too small."""
    height, width = frame_shape[:2]
    min_extent = min(height, width)
    if min_extent < 3:
        return None

    win_size = min(7, min_extent)
    if win_size % 2 == 0:
        win_size -= 1
    return win_size if win_size >= 3 else None


def _to_grayscale(frame):
    """Convert a BGR frame to grayscale for robust 2D SSIM."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported frame shape for grayscale SSIM: {frame.shape}")


def _compute_frame_ssim(
    structural_similarity,
    gt_frame,
    restored_frame,
    frame_index,
    state,
):
    """Compute SSIM on grayscale 2D frames."""
    gt_gray = _to_grayscale(gt_frame)
    restored_gray = _to_grayscale(restored_frame)

    if gt_gray.shape != restored_gray.shape:
        raise ValueError(
            "Grayscale frames have different shapes: "
            f"GT {gt_gray.shape}, restored {restored_gray.shape}"
        )

    win_size = _safe_ssim_win_size(gt_gray.shape)
    if win_size is None:
        raise ValueError("Frame is too small for SSIM; minimum height/width is 3 pixels.")

    if not state["ssim_debug_printed"]:
        print(
            "SSIM debug: "
            f"frame_index={frame_index}, "
            f"gt_frame_shape={gt_frame.shape}, "
            f"restored_frame_shape={restored_frame.shape}, "
            f"gt_gray_shape={gt_gray.shape}, "
            f"restored_gray_shape={restored_gray.shape}, "
            f"win_size={win_size}, "
            f"dtype={gt_gray.dtype}"
        )
        state["ssim_debug_printed"] = True

    return structural_similarity(
        gt_gray,
        restored_gray,
        data_range=_ssim_data_range(gt_gray, restored_gray),
        win_size=win_size,
    )


def _aligned_video_quality(gt_video, restored_video, warnings):
    """Compute PSNR and optional SSIM when GT and restored videos are aligned."""
    metrics = {"psnr": None, "ssim": None}

    if not gt_video or not restored_video:
        warnings.append("GT and restored videos were not both provided, so PSNR/SSIM were skipped.")
        return metrics

    gt_video = Path(gt_video)
    restored_video = Path(restored_video)
    if not gt_video.exists() or not restored_video.exists():
        warnings.append("GT or restored video is missing, so PSNR/SSIM were skipped.")
        return metrics

    try:
        import cv2
    except ImportError:
        warnings.append("OpenCV is not installed, so PSNR/SSIM were skipped.")
        return metrics

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        structural_similarity = None
        warnings.append("scikit-image is not installed, so SSIM was skipped.")

    gt_capture = cv2.VideoCapture(str(gt_video))
    restored_capture = cv2.VideoCapture(str(restored_video))
    if not gt_capture.isOpened() or not restored_capture.isOpened():
        warnings.append("Could not open GT or restored video, so PSNR/SSIM were skipped.")
        gt_capture.release()
        restored_capture.release()
        return metrics

    gt_frame_count = int(gt_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    restored_frame_count = int(restored_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if gt_frame_count <= 0 or restored_frame_count <= 0 or gt_frame_count != restored_frame_count:
        warnings.append("GT and restored videos are not frame-aligned, so PSNR/SSIM were skipped.")
        gt_capture.release()
        restored_capture.release()
        return metrics

    psnr_values = []
    ssim_values = []
    state = {"resize_warning_printed": False, "ssim_debug_printed": False}
    frame_index = 0

    while True:
        gt_ok, gt_frame = gt_capture.read()
        restored_ok, restored_frame = restored_capture.read()
        if not gt_ok or not restored_ok:
            break

        if gt_frame.shape != restored_frame.shape:
            restored_frame = _resize_restored_to_gt(
                gt_frame,
                restored_frame,
                warnings,
                state,
            )
            if restored_frame is None:
                psnr_values = []
                ssim_values = []
                break

        diff = gt_frame.astype("float32") - restored_frame.astype("float32")
        mse = float(np.mean(diff * diff))
        if mse == 0:
            psnr_values.append(float("inf"))
        else:
            psnr_values.append(20 * math.log10(255.0 / math.sqrt(mse)))

        if structural_similarity is not None:
            try:
                ssim_value = _compute_frame_ssim(
                    structural_similarity,
                    gt_frame,
                    restored_frame,
                    frame_index,
                    state,
                )
                ssim_values.append(float(ssim_value))
            except (TypeError, ValueError) as exc:
                warnings.append(
                    "SSIM was skipped because frames were incompatible at "
                    f"frame {frame_index}: {exc}. "
                    f"GT shape={gt_frame.shape}, restored shape={restored_frame.shape}"
                )
                structural_similarity = None

        frame_index += 1

    gt_capture.release()
    restored_capture.release()

    if psnr_values:
        metrics["psnr"] = float(np.mean(psnr_values))
    if ssim_values:
        metrics["ssim"] = float(np.mean(ssim_values))

    return metrics


def _save_metrics(metrics, output_dir):
    """Save metrics as JSON and key-value CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    json_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, list):
                writer.writerow([key, " | ".join(str(item) for item in value)])
            else:
                writer.writerow([key, "" if value is None else value])

    return json_path, csv_path


def compute_metrics(
    gt_video=None,
    degraded_video=None,
    restored_video=None,
    detection_csv=None,
    output_dir="outputs",
    restoration_outputs=None,
):
    """Compute feasible Stage 2 metrics and save JSON/CSV outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    metrics = {
        "video_name": Path(restored_video).name if restored_video else None,
        "model": "VSRNet",
        "vibration_level": _infer_vibration_level(degraded_video),
        "degraded_video_path": str(Path(degraded_video).resolve(strict=False))
        if degraded_video
        else None,
        "gt_video_path": str(Path(gt_video).resolve(strict=False)) if gt_video else None,
        "restored_video_path": str(Path(restored_video).resolve(strict=False))
        if restored_video
        else None,
        "detection_csv_path": str(Path(detection_csv).resolve(strict=False))
        if detection_csv
        else None,
        "metrics_source": "computed",
        "metrics_json_path": str((output_dir / "metrics.json").resolve(strict=False)),
        "psnr": None,
        "ssim": None,
        "jitter": None,
        "iou_continuity": None,
        "confidence_mean": None,
        "confidence_variance": None,
        "detection_count": 0,
        "detected_frame_count": 0,
        "restoration_total_time_sec": None,
        "fps": None,
        "inference_time_ms": None,
        "note": "Stage 2 metrics are computed when source files are available. Missing values are left as None.",
    }

    detections = _load_detection_rows(detection_csv, warnings)
    metrics.update(_video_info(degraded_video, "degraded", warnings))
    metrics.update(_video_info(restored_video, "restored", warnings))
    metrics.update(_video_info(gt_video, "gt", warnings))
    metrics.update(_detection_metrics(detections))
    metrics.update(_trajectory_metrics(detections, warnings))
    metrics.update(_aligned_video_quality(gt_video, restored_video, warnings))

    if restoration_outputs:
        metrics["restoration_total_time_sec"] = restoration_outputs.get(
            "restoration_total_time_sec"
        )
        metrics["inference_time_ms"] = restoration_outputs.get("inference_time_ms")
        if restoration_outputs.get("restored_frame_count") is not None:
            metrics["restored_frame_count"] = restoration_outputs["restored_frame_count"]

    if metrics.get("restored_fps") is not None:
        metrics["fps"] = metrics["restored_fps"]

    metrics["warnings"] = warnings
    json_path, csv_path = _save_metrics(metrics, output_dir)

    print(f"Metrics JSON saved to {json_path}")
    print(f"Metrics CSV saved to {csv_path}")

    return metrics
