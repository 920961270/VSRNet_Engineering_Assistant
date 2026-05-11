# VSRNet Engineering Analysis Report

## User Question

How does VSRNet perform under high vibration?

## Pipeline Status

- VSRNet restoration: Skipped, reused existing restored video: G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\outputs\high_blur_640_10s_vsrnet_restored.mp4
- YOLO detection/tracking: Skipped, reused existing detection CSV: G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\outputs\detection_results.csv
- Metrics: Skipped, loaded existing metrics JSON: G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\outputs\metrics.json
- RAG retrieval: Real pipeline used.
- Report generation: Real pipeline used.

## Pipeline Summary

This Stage 2 pipeline attempts to run real VSRNet restoration through the existing local repository script, then attempts YOLO detection or tracking with Ultralytics if it is installed. If a required heavy component is missing, the assistant records a warning, writes placeholder outputs, and continues to metrics, RAG retrieval, and report generation.

## Metric Summary

| Metric | Value |
| --- | --- |
| video_name | high_blur_640_10s_vsrnet_restored.mp4 |
| model | VSRNet |
| vibration_level | high |
| psnr | 19.3888459769899 |
| ssim | 0.6988866481440009 |
| jitter | 1.5699211359024048 |
| iou_continuity | 0.9200869798660278 |
| confidence_mean | 0.27993112802505493 |
| confidence_variance | 0.03254939988255501 |
| detection_count | 106 |
| detected_frame_count | 84 |
| restoration_total_time_sec | 275.2298807 |
| fps | 24.986 |
| inference_time_ms | 1100.9195228 |
| note | Stage 2 metrics are computed when source files are available. Missing values are left as None. |
| degraded_frame_count | 250 |
| degraded_fps | 24.986 |
| degraded_duration_sec | 10.005603137757143 |
| restored_frame_count | 250 |
| restored_fps | 24.986 |
| restored_duration_sec | 10.005603137757143 |
| gt_frame_count | 250 |
| gt_fps | 24.986 |
| gt_duration_sec | 10.005603137757143 |
| warnings | None |
| degraded_video_path | G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\inputs\high_blur_640_10s.mp4 |
| gt_video_path | G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\inputs\GT_640_10s.mp4 |
| restored_video_path | G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\outputs\high_blur_640_10s_vsrnet_restored.mp4 |
| detection_csv_path | G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\outputs\detection_results.csv |
| metrics_json_path | G:\4周学习计划\第三周\VSRNet_Engineering_Assistant\outputs\metrics.json |
| metrics_source | reused |

## Retrieved VSRNet Knowledge

### Context 1: vsrnet_summary.txt (chunk 0, score 0.5602)

VSRNet is a lightweight video restoration network designed for vibration-degraded monitoring videos.

The project focuses on restoring video affected by physical vibration, rolling-shutter-like artifacts, and motion blur. It is intended for monitoring scenarios where stable visual evidence matters for later perception tasks.

Stage 1 of this assistant did not run real VSRNet inference. Stage 2 attempts to wrap the existing local VSRNet inference script when the input video and checkpoint are ava

### Context 2: method_notes.txt (chunk 0, score 0.5048)

VSRNet uses a practical multi-frame restoration setting. The original repository includes sliding-window inference for frame sequences and full videos.

The engineering workflow should treat restoration as one part of a larger closed loop: restore degraded video, evaluate visual quality, inspect downstream detection stability, and summarize what the metrics imply.

Vibration intensity in the project is controlled by exposure proportion and displacement amplitudes. Larger beta values usually mean

### Context 3: method_notes.txt (chunk 1, score 0.3896)

exposure proportion and displacement amplitudes. Larger beta values usually mean longer motion integration, while larger amplitudes mean stronger visible vibration.

## Rule-Based Engineering Analysis

This section is deterministic and is the source of truth.

### Direct Answer
For the question 'How does VSRNet perform under high vibration?', the current result is most promising on temporal detection stability, while restoration quality is moderate and runtime remains a major weakness.

### Visual Restoration Quality
PSNR 19.389 dB and SSIM 0.699 suggest moderate restoration quality, not near-perfect restoration.

### Detection Stability
IoU continuity 0.92 suggests strong temporal consistency of detection boxes. Jitter 1.57 pixels suggests low frame-to-frame center displacement, but this depends on object scale and detection quality. Confidence mean 0.28 suggests detections are relatively low-confidence, so detection reliability still needs improvement. Confidence variance 0.033 indicates measurable fluctuation in detection confidence. Detection count 106 over 84 detected frames means objects were detected in part of the 250-frame clip, not every frame.

### Efficiency
Restoration took 275.23 seconds total and 1100.92 ms/frame. This indicates the current implementation is slow and not real-time.

### Engineering Conclusion
The Stage 2 pipeline is useful for engineering analysis because it connects real restoration, detection, metrics, RAG context, and reporting. The current metrics do not support claims of excellent restoration or deployment-ready speed. The strongest signal is detection-box continuity; the weakest signals are low confidence and slow inference.

### Cautions
Since this appears to be a 640-width about 10 seconds test clip, conclusions are preliminary. The rule-based analysis is deterministic and should be treated as the reliable source of truth. Do not compare against traditional methods or claim unseen improvements unless baseline metrics are added.

## LLM Additional Analysis

This section is a supplementary interpretation layer. It does not replace, edit, or override the rule-based analysis above.

LLM Additional Analysis was sanitized by validation before being added.

### Additional Interpretation
The current results indicate that VSRNet performs moderately well in restoring video quality under high vibration conditions, with PSNR at 19.389 dB and SSIM at 0.699 suggesting a moderate level of restoration quality.

The temporal detection stability is strong as evidenced by IoU continuity at 0.92, but the low confidence mean (0.28) and variance (0.033) suggest room for improvement in detection reliability.

Given that the restoration process took 275.23 seconds total and 1100.92 ms/frame, it is clear that the current implementation is not real-time.

### Practical Implications
Given these findings, it's crucial to focus on improving detection confidence and efficiency.

The current low confidence levels indicate that while objects are detected in most frames (with a count of 106 over 84 detected frames), their reliability needs enhancement.

The current runtime indicates the implementation is slow and not real-time.

### Suggested Next Checks
To improve VSRNet's performance under high vibration conditions, consider the following next steps:
- **Enhance Detection Confidence**: Investigate methods to increase detection confidence without compromising object localization accuracy.

- **Optimize Inference Time**: Explore techniques such as model pruning, quantization, or using more efficient inference frameworks to reduce processing time.


## Recommendations

- Place the VSRNet checkpoint at the path passed through `--vsrnet-checkpoint`.
- Provide an input video through `--input` when testing real restoration.
- Install Ultralytics only when YOLO detection or tracking is needed.
- Add more project notes, experiment logs, and evaluation observations to `documents/` to improve RAG retrieval.
- Treat the raw metrics and rule-based analysis as more reliable than any LLM commentary.

## Warnings / Limitations

- None

## Next Steps

1. Confirm the restored output video visually.
2. Inspect `detection_results.csv` for detection quality and tracking IDs.
3. Compare GT and restored video with aligned frames when PSNR/SSIM are needed.
4. Add real experiment notes to the document corpus.
5. Upgrade the report writer later if an LLM-based explanation layer is required.
