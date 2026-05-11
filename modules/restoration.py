"""VSRNet restoration wrapper for Stage 2.

This module does not define or rewrite the VSRNet architecture. It wraps the
existing local inference script and makes output handling robust enough for the
rest of the assistant pipeline.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


DEFAULT_VSRNET_REPO_PATH = Path(r"G:\EDVR_3levels\VSRNet")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _absolute_path(path):
    """Return an absolute Path without requiring the path to already exist."""
    return Path(path).expanduser().resolve(strict=False)


def _placeholder_result(input_video, model_path, output_dir, warning):
    """Create placeholder restoration outputs and return a robust result."""
    print(f"ERROR: {warning}")
    print(
        "Stage 1 placeholder restoration used. Real VSRNet inference will be connected in Stage 2."
    )

    output_dir = _absolute_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    note_path = output_dir / "restoration_placeholder.txt"
    restored_video = output_dir / "restored_placeholder.mp4"
    note_path.write_text(
        f"Stage 2 restoration placeholder.\nReason: {warning}\n",
        encoding="utf-8",
    )

    return {
        "restored_video": str(restored_video),
        "restored_frames_dir": None,
        "used_placeholder": True,
        "warning": warning,
        "input_video": str(input_video) if input_video else None,
        "model_path": str(model_path) if model_path else None,
        "note_path": str(note_path),
        "restoration_total_time_sec": None,
        "inference_time_ms": None,
        "restored_frame_count": None,
    }


def _nonzero_file(path):
    """Check whether a file exists and has content."""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def _list_output_folder(output_dir, limit=80):
    """Print a compact listing of files and folders found after inference."""
    output_dir = Path(output_dir)
    print(f"Files found in output folder after inference: {output_dir}")

    if not output_dir.exists():
        print("- Output folder does not exist.")
        return

    paths = sorted(output_dir.rglob("*"), key=lambda item: str(item).lower())
    if not paths:
        print("- Output folder is empty.")
        return

    for path in paths[:limit]:
        kind = "DIR " if path.is_dir() else "FILE"
        size = path.stat().st_size if path.is_file() else ""
        print(f"- {kind} {path} {size}")

    if len(paths) > limit:
        print(f"- ... {len(paths) - limit} more paths omitted")


def _find_frame_files(frames_dir):
    """Find image frames in a directory tree."""
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        return []

    return sorted(
        [
            path
            for path in frames_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: str(path).lower(),
    )


def _candidate_frame_dirs(output_dir, restored_frames_dir, input_stem):
    """Return likely frame output folders in priority order."""
    output_dir = Path(output_dir)
    candidates = [Path(restored_frames_dir)]

    if output_dir.exists():
        for path in output_dir.iterdir():
            if not path.is_dir():
                continue
            name = path.name.lower()
            if (
                input_stem.lower() in name
                or "frame" in name
                or "vsrnet" in name
                or "restored" in name
            ):
                candidates.append(path)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).lower()
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def _read_image_unicode(frame_path):
    """Read an image in a Windows-friendly way for paths with non-ASCII text."""
    import cv2
    import numpy as np

    data = np.fromfile(str(frame_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _input_video_properties(input_video):
    """Read FPS and frame size from the input video."""
    import cv2

    capture = cv2.VideoCapture(str(input_video))
    fps = 25.0
    size = None

    if capture.isOpened():
        read_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if read_fps > 0:
            fps = read_fps
        if width > 0 and height > 0:
            size = (width, height)

    capture.release()
    return fps, size


def _video_frame_count(video_path):
    """Read frame count from a video file when OpenCV can open it."""
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return frame_count if frame_count > 0 else None


def _timing_fields(restored_video, restoration_total_time_sec):
    """Return restoration timing fields for report metrics."""
    restored_frame_count = _video_frame_count(restored_video)
    inference_time_ms = None
    if restored_frame_count:
        inference_time_ms = restoration_total_time_sec * 1000.0 / restored_frame_count

    return {
        "restoration_total_time_sec": restoration_total_time_sec,
        "inference_time_ms": inference_time_ms,
        "restored_frame_count": restored_frame_count,
    }


def _open_video_writer(output_video, fps, frame_size):
    """Open a VideoWriter, with a temp-file fallback for Unicode path trouble."""
    import cv2

    output_video = Path(output_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, frame_size)
    if writer.isOpened():
        return writer, output_video, None

    writer.release()
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, frame_size)
    if writer.isOpened():
        return writer, temp_path, temp_path

    writer.release()
    try:
        temp_path.unlink(missing_ok=True)
    except OSError:
        pass
    return None, output_video, None


def _assemble_frames_to_video(frame_files, input_video, restored_video):
    """Assemble image frames into an mp4 using input-video FPS and frame size."""
    import cv2

    if not frame_files:
        return False, "No frame files were available for mp4 assembly."

    restored_video = Path(restored_video)
    restored_video.parent.mkdir(parents=True, exist_ok=True)

    fps, frame_size = _input_video_properties(input_video)
    first_frame = _read_image_unicode(frame_files[0])
    if first_frame is None:
        return False, f"Could not read the first restored frame: {frame_files[0]}"

    if frame_size is None:
        frame_size = (first_frame.shape[1], first_frame.shape[0])

    writer, writer_path, temp_path = _open_video_writer(restored_video, fps, frame_size)
    if writer is None:
        return False, f"Could not open OpenCV VideoWriter for: {restored_video}"

    written_count = 0
    for frame_path in frame_files:
        frame = _read_image_unicode(frame_path)
        if frame is None:
            print(f"WARNING: Skipping unreadable restored frame: {frame_path}")
            continue
        if (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size)
        writer.write(frame)
        written_count += 1

    writer.release()

    if written_count == 0:
        return False, "No readable restored frames were written to the assembled video."

    if temp_path is not None:
        shutil.copyfile(temp_path, restored_video)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    if not _nonzero_file(restored_video):
        return False, f"Frame assembly finished, but mp4 was not created: {restored_video}"

    print(f"Assembled {written_count} restored frames into {restored_video}")
    return True, None


def _find_existing_mp4(output_dir, expected_video, input_stem, started_at):
    """Find a non-empty mp4 in the output folder when the expected name is absent."""
    expected_video = Path(expected_video)
    if _nonzero_file(expected_video):
        return expected_video

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    mp4_files = []
    for path in output_dir.rglob("*.mp4"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        name = path.name.lower()
        if path.stat().st_mtime >= started_at - 2:
            mp4_files.append(path)
        elif input_stem.lower() in name and ("vsrnet" in name or "restored" in name):
            mp4_files.append(path)

    if not mp4_files:
        return None

    mp4_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return mp4_files[0]


def _print_debug_info(
    repo_path,
    inference_script,
    input_video,
    checkpoint_path,
    output_dir,
    restored_video,
    restored_frames_dir,
    command,
):
    """Print the paths and command used for VSRNet inference."""
    print("Running real VSRNet inference through the existing repository script...")
    print(f"Current working directory: {Path.cwd()}")
    print(f"VSRNet repo path: {repo_path}")
    print(f"Inference script path: {inference_script}")
    print(f"Absolute input video path: {input_video}")
    print(f"Absolute checkpoint path: {checkpoint_path}")
    print(f"Absolute output directory: {output_dir}")
    print(f"Absolute restored video path: {restored_video}")
    print(f"Absolute restored frames directory: {restored_frames_dir}")
    print("Subprocess command:")
    print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in command))


def _real_result(
    restored_video,
    restored_frames_dir,
    input_video,
    checkpoint_path,
    restoration_total_time_sec,
):
    """Build a successful restoration result dictionary."""
    timing = _timing_fields(restored_video, restoration_total_time_sec)
    if timing["inference_time_ms"] is not None:
        print(
            "VSRNet timing: "
            f"total={timing['restoration_total_time_sec']:.3f}s, "
            f"frames={timing['restored_frame_count']}, "
            f"avg={timing['inference_time_ms']:.3f} ms/frame"
        )
    else:
        print(
            "WARNING: Could not compute inference_time_ms because restored frame count was unavailable."
        )

    return {
        "restored_video": str(restored_video),
        "restored_frames_dir": str(restored_frames_dir) if restored_frames_dir else None,
        "used_placeholder": False,
        "warning": None,
        "input_video": str(input_video),
        "model_path": str(checkpoint_path),
        "note_path": None,
        **timing,
    }


def run_vsrnet_restoration(
    input_video=None,
    model_path=None,
    output_dir="outputs",
    repo_path=DEFAULT_VSRNET_REPO_PATH,
):
    """Run VSRNet video restoration when the local repo and checkpoint exist."""
    output_dir = _absolute_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video:
        return _placeholder_result(
            input_video,
            model_path,
            output_dir,
            "No input video was provided.",
        )

    input_video = _absolute_path(input_video)
    if not input_video.exists():
        return _placeholder_result(
            input_video,
            model_path,
            output_dir,
            f"Input video does not exist: {input_video}",
        )

    if not model_path:
        return _placeholder_result(
            input_video,
            model_path,
            output_dir,
            "No VSRNet checkpoint path was provided.",
        )

    checkpoint_path = _absolute_path(model_path)
    if not checkpoint_path.exists():
        return _placeholder_result(
            input_video,
            checkpoint_path,
            output_dir,
            f"VSRNet checkpoint does not exist: {checkpoint_path}",
        )

    repo_path = _absolute_path(repo_path)
    inference_script = repo_path / "infer_vsrnet_video.py"
    if not inference_script.exists():
        return _placeholder_result(
            input_video,
            checkpoint_path,
            output_dir,
            f"VSRNet video inference script was not found: {inference_script}",
        )

    restored_video = output_dir / f"{input_video.stem}_vsrnet_restored.mp4"
    restored_frames_dir = output_dir / f"{input_video.stem}_vsrnet_frames"
    restored_frames_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(inference_script),
        "--input-video",
        str(input_video),
        "--ckpt",
        str(checkpoint_path),
        "--save-video",
        str(restored_video),
    ]

    _print_debug_info(
        repo_path,
        inference_script,
        input_video,
        checkpoint_path,
        output_dir,
        restored_video,
        restored_frames_dir,
        command,
    )

    started_at = time.time()
    subprocess_started_at = time.perf_counter()
    try:
        subprocess.run(
            command,
            cwd=str(repo_path),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        _list_output_folder(output_dir)
        return _placeholder_result(
            input_video,
            checkpoint_path,
            output_dir,
            f"VSRNet inference command failed with exit code {exc.returncode}.",
        )
    except OSError as exc:
        _list_output_folder(output_dir)
        return _placeholder_result(
            input_video,
            checkpoint_path,
            output_dir,
            f"Could not start VSRNet inference: {exc}",
        )
    restoration_total_time_sec = time.perf_counter() - subprocess_started_at

    _list_output_folder(output_dir)

    real_mp4 = _find_existing_mp4(output_dir, restored_video, input_video.stem, started_at)
    if real_mp4 is not None:
        if real_mp4 != restored_video:
            print(f"WARNING: Expected mp4 was absent, but found restored mp4: {real_mp4}")
        print(f"VSRNet restored video saved to {real_mp4}")
        return _real_result(
            real_mp4,
            restored_frames_dir,
            input_video,
            checkpoint_path,
            restoration_total_time_sec,
        )

    for candidate_dir in _candidate_frame_dirs(output_dir, restored_frames_dir, input_video.stem):
        frame_files = _find_frame_files(candidate_dir)
        if not frame_files:
            continue

        print(f"Found {len(frame_files)} restored frame files in {candidate_dir}")
        assembled, warning = _assemble_frames_to_video(frame_files, input_video, restored_video)
        if assembled:
            return _real_result(
                restored_video,
                candidate_dir,
                input_video,
                checkpoint_path,
                restoration_total_time_sec,
            )
        print(f"WARNING: {warning}")

    return _placeholder_result(
        input_video,
        checkpoint_path,
        output_dir,
        "VSRNet inference finished, but no usable non-empty mp4 or restored image frames were produced.",
    )
