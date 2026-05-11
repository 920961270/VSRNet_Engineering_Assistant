# VSRNet Engineering Analysis Assistant

A lightweight engineering assistant for evaluating **VSRNet-based video restoration** under vibration-induced degradation.

This project connects video restoration, downstream YOLO detection, metric computation, local RAG retrieval, deterministic rule-based analysis, and optional LLM commentary into one reproducible engineering pipeline.

> In short: this is not just a restoration demo. It is a small CV + Metrics + RAG + LLM engineering system for analyzing whether restored vibration-degraded videos are useful for downstream perception tasks.

---

## Why This Project Exists

VSRNet is designed for restoring monitoring videos affected by vibration, motion blur, and unstable visual evidence. In real engineering scenarios, however, restoration quality alone is not enough.

We also want to know:

- Does the restored video help downstream object detection?
- Are detections temporally stable across frames?
- Are confidence scores reliable?
- Is the pipeline efficient enough for practical use?
- Can the system generate an interpretable engineering report?

This assistant answers those questions by combining restoration, detection, metrics, retrieval, and report generation.

---

## Core Features

- Wraps an existing VSRNet inference script for video restoration.
- Runs YOLO detection or tracking on restored videos.
- Computes restoration and perception-oriented metrics.
- Retrieves project knowledge from local documents with FAISS + SentenceTransformer.
- Generates deterministic rule-based engineering analysis.
- Optionally adds LLM supplementary analysis through Ollama or OpenAI.
- Validates and sanitizes LLM output to reduce hallucination.
- Supports reuse / report-only mode to avoid repeated expensive inference.

---

## System Overview

```text
Degraded video
    |
    v
VSRNet restoration
    |
    v
Restored video
    |
    v
YOLO detection / tracking
    |
    v
Metrics computation
    |
    +------------------+
    |                  |
    v                  v
RAG retrieval     Rule-based analysis
    |                  |
    +--------+---------+
             v
Optional LLM commentary
             |
             v
Markdown engineering report
```

The rule-based analysis is the reliable source of truth. The LLM section is only a supplementary explanation layer.

---

## Folder Structure

```text
VSRNet_Engineering_Assistant/
|-- main_pipeline.py
|-- modules/
|   |-- restoration.py
|   |-- detection.py
|   |-- metrics.py
|   |-- rag_retriever.py
|   |-- report_generator.py
|   `-- llm_report.py
|-- documents/
|   |-- vsrnet_summary.txt
|   |-- method_notes.txt
|   |-- evaluation_notes.txt
|   `-- experiment_logs.txt
|-- models/
|   |-- vsrnet/
|   `-- all-MiniLM-L6-v2/
|-- inputs/
|-- outputs/
|-- requirements.txt
`-- README.md
```

Recommended local layout:

```text
models/vsrnet/best.pth
models/all-MiniLM-L6-v2/
inputs/high_blur_640_10s.mp4
inputs/GT_640_10s.mp4
documents/*.txt
```

Large files such as videos, checkpoints, model weights, and generated outputs should usually be excluded from GitHub.

---

## Installation

Create and activate a Python environment, then install the base dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install ultralytics
pip install scikit-image
```

Ultralytics is used for YOLO detection or tracking.  
Scikit-image is used for SSIM computation.

If optional packages are missing, the pipeline records warnings and continues when possible.

---

## External Requirements

This assistant wraps an existing VSRNet repository instead of retraining or rewriting the restoration model.

You need:

```text
A local VSRNet repository
A VSRNet checkpoint
An input degraded video
Optional GT video for PSNR / SSIM
Optional YOLO model
Optional local embedding model
Optional Ollama or OpenAI access for LLM commentary
```

Example placeholders:

```text
<PATH_TO_VSRNET_REPO>
<PATH_TO_VSRNET_CHECKPOINT>
<PATH_TO_EMBEDDING_MODEL>
```

For example, the embedding model can be placed here:

```text
models/all-MiniLM-L6-v2/
```

Then use:

```bash
--embedding-model models/all-MiniLM-L6-v2
```

---

## Stage 1: Placeholder MVP

Stage 1 verifies that the pipeline structure works even without heavy models.

```bash
python main_pipeline.py --question "What problem does VSRNet solve?"
```

This mode uses placeholder restoration and detection, then generates a basic engineering report.

It is useful for checking whether the project skeleton, RAG retrieval, and report generation are connected correctly.

---

## Stage 2: Full Engineering Pipeline

A full run performs restoration, detection, metrics, retrieval, and report generation.

```bash
python main_pipeline.py --input inputs/high_blur_640_10s.mp4 --gt inputs/GT_640_10s.mp4 --question "How does VSRNet perform under high vibration?" --vsrnet-repo "<PATH_TO_VSRNET_REPO>" --vsrnet-checkpoint models/vsrnet/best.pth --embedding-model models/all-MiniLM-L6-v2 --documents documents --yolo-model yolov8n.pt --output outputs
```

On Windows PowerShell, one-line commands are recommended to avoid line-continuation mistakes.

If you want to use multiple lines in PowerShell, use the backtick character:

```powershell
python main_pipeline.py `
  --input inputs/high_blur_640_10s.mp4 `
  --gt inputs/GT_640_10s.mp4 `
  --question "How does VSRNet perform under high vibration?" `
  --vsrnet-repo "<PATH_TO_VSRNET_REPO>" `
  --vsrnet-checkpoint models/vsrnet/best.pth `
  --embedding-model models/all-MiniLM-L6-v2 `
  --documents documents `
  --yolo-model yolov8n.pt `
  --output outputs
```

---

## Reuse / Report-Only Modes

Restoration can be slow. Once restored videos, detection CSVs, or metrics are generated, you can reuse them.

### Reuse Restored Video

```bash
python main_pipeline.py --input inputs/high_blur_640_10s.mp4 --gt inputs/GT_640_10s.mp4 --restored-video outputs/high_blur_640_10s_vsrnet_restored.mp4 --skip-restoration --question "How does VSRNet perform under high vibration?" --documents documents --yolo-model yolov8n.pt --output outputs
```

This skips VSRNet inference, reruns YOLO, recomputes metrics, and regenerates the report.

### Reuse Restored Video and Detection CSV

```bash
python main_pipeline.py --input inputs/high_blur_640_10s.mp4 --gt inputs/GT_640_10s.mp4 --restored-video outputs/high_blur_640_10s_vsrnet_restored.mp4 --skip-restoration --detection-csv outputs/detection_results.csv --skip-detection --question "How does VSRNet perform under high vibration?" --documents documents --output outputs
```

This skips restoration and detection, then recomputes metrics and regenerates the report.

### Report-Only / LLM-Only Mode

```bash
python main_pipeline.py --input inputs/high_blur_640_10s.mp4 --gt inputs/GT_640_10s.mp4 --restored-video outputs/high_blur_640_10s_vsrnet_restored.mp4 --skip-restoration --detection-csv outputs/detection_results.csv --skip-detection --metrics-json outputs/metrics.json --skip-metrics --question "How does VSRNet perform under high vibration?" --embedding-model models/all-MiniLM-L6-v2 --documents documents --output outputs --use-llm-report --llm-provider ollama --llm-model qwen2.5:3b --llm-timeout 300
```

This mode is useful when you only want to regenerate the RAG-based report and LLM commentary without rerunning expensive CV inference.

Reusable files are validated before use. In skip mode, missing reused files produce a clear error instead of silently falling back to placeholders.

---

## Metrics

The assistant currently summarizes:

```text
PSNR
SSIM
jitter
IoU continuity
confidence mean
confidence variance
detection count
detected frame count
restoration total time
inference time per frame
video frame count / FPS / duration
warnings
```

Example interpretation:

```text
PSNR around 19 dB and SSIM around 0.7 suggest moderate restoration quality.
IoU continuity above 0.9 suggests strong temporal box consistency.
Low confidence mean indicates that detection reliability still needs improvement.
Inference time above 1000 ms/frame means the current implementation is not real-time.
```

Metric interpretation is intentionally conservative. The project avoids unsupported claims such as “near-perfect restoration” or “real-time performance” unless the metrics clearly support them.

---

## Report Design

The generated report contains:

```text
Metric Summary
Retrieved VSRNet Knowledge
Rule-Based Engineering Analysis
Optional LLM Additional Analysis
Recommendations
Warnings / Limitations
Next Steps
```

### Rule-Based Engineering Analysis

This section is deterministic and should be treated as the reliable source of truth.

It interprets metrics using fixed engineering rules instead of free-form generation.

### LLM Additional Analysis

When enabled, the LLM section is added only as supplementary commentary.

It must not:

```text
override raw metrics
replace rule-based analysis
invent numbers
claim unsupported improvements
claim real-time performance when runtime is slow
```

The assistant saves LLM validation artifacts:

```text
outputs/llm_raw_output.md
outputs/llm_sanitized_output.md
outputs/llm_validation_log.txt
```

The raw output is generated by the LLM.  
The sanitized output is cleaned and safer to include.  
The validation log records what was allowed, sanitized, or rejected.

---

## LLM Options

### Ollama

Install Ollama, pull a local model, then run:

```bash
ollama pull qwen2.5:3b
```

Check installed models:

```bash
ollama list
```

Run the pipeline with Ollama:

```bash
python main_pipeline.py --input inputs/high_blur_640_10s.mp4 --gt inputs/GT_640_10s.mp4 --restored-video outputs/high_blur_640_10s_vsrnet_restored.mp4 --skip-restoration --detection-csv outputs/detection_results.csv --skip-detection --metrics-json outputs/metrics.json --skip-metrics --question "How does VSRNet perform under high vibration?" --embedding-model models/all-MiniLM-L6-v2 --documents documents --output outputs --use-llm-report --llm-provider ollama --llm-model qwen2.5:3b --llm-timeout 300
```

For smaller machines, try:

```bash
--llm-model qwen2.5:1.5b
```

For slower local models, increase:

```bash
--llm-timeout 300
```

or:

```bash
--llm-timeout 600
```

### OpenAI

Set the API key in your environment. Do not write API keys into code or README files.

Linux / macOS:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Then run:

```bash
python main_pipeline.py --input inputs/high_blur_640_10s.mp4 --gt inputs/GT_640_10s.mp4 --question "How does VSRNet perform under high vibration?" --vsrnet-repo "<PATH_TO_VSRNET_REPO>" --vsrnet-checkpoint models/vsrnet/best.pth --embedding-model models/all-MiniLM-L6-v2 --documents documents --yolo-model yolov8n.pt --output outputs --use-llm-report --llm-provider openai --llm-model gpt-4o-mini
```

---

## What Happens If Something Is Missing

Missing input video:

```text
Restoration falls back to placeholder in normal mode.
```

Missing checkpoint:

```text
Restoration falls back to placeholder in normal mode.
```

Missing VSRNet repository:

```text
Restoration falls back to placeholder in normal mode.
```

Missing Ultralytics:

```text
Detection writes placeholder outputs.
```

Missing GT video:

```text
PSNR and SSIM are skipped.
```

Missing Ollama or OpenAI access:

```text
LLM Additional Analysis is skipped or replaced with a validation note.
Rule-Based Engineering Analysis remains available.
```

In reuse / skip mode, missing reused files stop the pipeline with a clear error.

---

## Debug Notes

Some common issues:

```text
--question missing
```

Usually caused by running a README multi-line command incorrectly.

```text
FileNotFoundError for input video
```

Check whether the input path exists and whether the command is run from the project root.

```text
VSRNet finished but output video not found
```

Often caused by relative paths and subprocess working directory mismatch. Use absolute paths internally.

```text
SSIM win_size error
```

Usually caused by color-channel handling in structural similarity. A robust fix is to compute SSIM on grayscale frames or correctly set the channel axis.

```text
Ollama timeout
```

Use a smaller local model or increase `--llm-timeout`.

---

## Current Limitations

- The VSRNet wrapper assumes the external inference script supports `--input-video`, `--ckpt`, and `--save-video`.
- YOLO tracking quality depends on the selected model and environment.
- Metrics are meaningful only when videos are frame-aligned.
- Current inference speed may not be real-time.
- LLM commentary is supplementary and should not be treated as ground truth.
- Retrieval quality depends on the content of `documents/`.

---

## Future Work

- Add batch processing for low / mid / high vibration levels.
- Add baseline comparison with raw input, FastDVDnet, and VRT.
- Improve tracking-based jitter and IoU continuity computation.
- Add visualization plots for metric summaries.
- Add a lightweight web UI or report dashboard.
- Add tests for placeholder fallback and reuse modes.
- Prepare a clean GitHub demo with sample videos and screenshots.

---

## Resume-Friendly Summary

Built a CV + RAG + LLM engineering assistant for VSRNet, integrating video restoration, YOLO-based downstream evaluation, metric computation, FAISS retrieval, deterministic rule-based analysis, and validated LLM commentary with anti-hallucination safeguards.
