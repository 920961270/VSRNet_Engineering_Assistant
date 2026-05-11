"""Rule-based and optional LLM commentary for VSRNet reports."""

import json
import os
import re
import socket
import urllib.error
import urllib.request
from pathlib import Path


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
FORBIDDEN_PHRASES = [
    "near-perfect",
    "near perfect",
    "noise-free",
    "excellent",
    "exceptional",
    "highly efficient",
    "real-time",
    "deployment-ready",
    "significant speed improvement",
    "significant improvement in processing speed",
]
ENGLISH_NEGATIONS = [
    "not",
    "no",
    "never",
    "cannot",
    "can't",
    "does not",
    "do not",
    "did not",
    "without",
    "should not",
]
CHINESE_NEGATIONS = [
    "\u4e0d",
    "\u4e0d\u662f",
    "\u4e0d\u80fd",
    "\u4e0d\u5e94",
    "\u6ca1\u6709",
    "\u5e76\u975e",
    "涓?",
    "涓嶆槸",
    "涓嶈兘",
    "涓嶅簲",
    "娌℃湁",
    "骞堕潪",
]


def _clean_metric_value(value):
    """Convert values to JSON-friendly forms without hiding missing values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_clean_metric_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_metric_value(item) for key, item in value.items()}
    return str(value)


def _compact_metrics(metrics):
    """Keep all metrics while converting unusual values to JSON-friendly text."""
    return {str(key): _clean_metric_value(value) for key, value in metrics.items()}


def _compact_contexts(retrieved_contexts, max_chars=900):
    """Trim retrieved contexts so local and API LLM calls stay lightweight."""
    compacted = []
    for context in retrieved_contexts:
        text = (context.get("text") or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        compacted.append(
            {
                "source": context.get("source"),
                "chunk_id": context.get("chunk_id"),
                "score": context.get("score"),
                "text": text,
            }
        )
    return compacted


def _pipeline_status(restoration_outputs, detection_outputs):
    """Summarize pipeline status for prompt context."""
    return {
        "restoration": {
            "used_placeholder": bool(
                restoration_outputs and restoration_outputs.get("used_placeholder")
            ),
            "warning": restoration_outputs.get("warning") if restoration_outputs else None,
            "restored_video": restoration_outputs.get("restored_video")
            if restoration_outputs
            else None,
            "input_video": restoration_outputs.get("input_video")
            if restoration_outputs
            else None,
        },
        "detection": {
            "used_placeholder": bool(
                detection_outputs and detection_outputs.get("used_placeholder")
            ),
            "warning": detection_outputs.get("warning") if detection_outputs else None,
            "detection_csv": detection_outputs.get("detection_csv")
            if detection_outputs
            else None,
        },
    }


def _collect_warnings(metrics, restoration_outputs, detection_outputs):
    """Collect warnings and limitations."""
    warnings = []
    if restoration_outputs and restoration_outputs.get("warning"):
        warnings.append(restoration_outputs["warning"])
    if detection_outputs and detection_outputs.get("warning"):
        warnings.append(detection_outputs["warning"])
    metric_warnings = metrics.get("warnings") or []
    if isinstance(metric_warnings, list):
        warnings.extend(metric_warnings)
    else:
        warnings.append(str(metric_warnings))
    return warnings


def _detect_clip_caution(metrics, pipeline_status):
    """Create a caution sentence for recognizable short test clips."""
    input_video = pipeline_status.get("restoration", {}).get("input_video")
    input_name = Path(input_video).name if input_video else ""
    duration = metrics.get("degraded_duration_sec") or metrics.get("restored_duration_sec")
    try:
        duration_value = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_value = None

    hints = []
    lower_name = input_name.lower()
    if "640" in lower_name:
        hints.append("640-width")
    if "10s" in lower_name or (duration_value is not None and 8 <= duration_value <= 12):
        hints.append("about 10 seconds")

    if hints:
        return (
            "Since this appears to be a "
            + " ".join(hints)
            + " test clip, conclusions are preliminary."
        )
    return "Conclusions are preliminary and depend on the available test clip and metrics."


def _metric_text(metrics, key, unavailable="unavailable"):
    """Format a metric value for readable deterministic text."""
    value = metrics.get(key)
    if value is None:
        return unavailable
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def generate_rule_based_analysis(
    question,
    metrics,
    restoration_outputs=None,
    detection_outputs=None,
):
    """Generate the deterministic source-of-truth engineering analysis."""
    pipeline_status = _pipeline_status(restoration_outputs, detection_outputs)
    psnr = metrics.get("psnr")
    ssim = metrics.get("ssim")
    jitter = metrics.get("jitter")
    iou = metrics.get("iou_continuity")
    confidence_mean = metrics.get("confidence_mean")
    confidence_variance = metrics.get("confidence_variance")
    detection_count = metrics.get("detection_count")
    detected_frame_count = metrics.get("detected_frame_count")
    restored_frame_count = metrics.get("restored_frame_count")
    total_time = metrics.get("restoration_total_time_sec")
    inference_ms = metrics.get("inference_time_ms")

    if psnr is not None and ssim is not None:
        visual_line = (
            f"PSNR {_metric_text(metrics, 'psnr')} dB and SSIM "
            f"{_metric_text(metrics, 'ssim')} suggest moderate restoration quality, "
            "not near-perfect restoration."
        )
    else:
        visual_line = "PSNR or SSIM is unavailable, so visual restoration quality cannot be fully judged."

    stability_parts = []
    if iou is not None:
        stability_parts.append(
            f"IoU continuity {_metric_text(metrics, 'iou_continuity')} suggests strong temporal consistency of detection boxes."
        )
    else:
        stability_parts.append("IoU continuity is unavailable.")
    if jitter is not None:
        stability_parts.append(
            f"Jitter {_metric_text(metrics, 'jitter')} pixels suggests low frame-to-frame center displacement, but this depends on object scale and detection quality."
        )
    else:
        stability_parts.append("Jitter is unavailable.")
    if confidence_mean is not None:
        stability_parts.append(
            f"Confidence mean {_metric_text(metrics, 'confidence_mean')} suggests detections are relatively low-confidence, so detection reliability still needs improvement."
        )
    else:
        stability_parts.append("Confidence mean is unavailable.")
    if confidence_variance is not None:
        stability_parts.append(
            f"Confidence variance {_metric_text(metrics, 'confidence_variance')} indicates measurable fluctuation in detection confidence."
        )
    if detection_count is not None and detected_frame_count is not None:
        if restored_frame_count is not None:
            stability_parts.append(
                f"Detection count {detection_count} over {detected_frame_count} detected frames means objects were detected in part of the {restored_frame_count}-frame clip, not every frame."
            )
        else:
            stability_parts.append(
                f"Detection count {detection_count} over {detected_frame_count} detected frames means objects were not detected in every available frame."
            )

    if total_time is not None and inference_ms is not None:
        efficiency_line = (
            f"Restoration took {_metric_text(metrics, 'restoration_total_time_sec')} seconds total "
            f"and {_metric_text(metrics, 'inference_time_ms')} ms/frame. "
            "This indicates the current implementation is slow and not real-time."
        )
    else:
        efficiency_line = "Restoration runtime or per-frame inference time is unavailable."

    caution_line = _detect_clip_caution(metrics, pipeline_status)

    return (
        "### Direct Answer\n"
        f"For the question '{question}', the current result is most promising on temporal detection stability, "
        "while restoration quality is moderate and runtime remains a major weakness.\n\n"
        "### Visual Restoration Quality\n"
        f"{visual_line}\n\n"
        "### Detection Stability\n"
        f"{' '.join(stability_parts)}\n\n"
        "### Efficiency\n"
        f"{efficiency_line}\n\n"
        "### Engineering Conclusion\n"
        "The Stage 2 pipeline is useful for engineering analysis because it connects real restoration, detection, "
        "metrics, RAG context, and reporting. The current metrics do not support claims of excellent restoration "
        "or deployment-ready speed. The strongest signal is detection-box continuity; the weakest signals are "
        "low confidence and slow inference.\n\n"
        "### Cautions\n"
        f"{caution_line} The rule-based analysis is deterministic and should be treated as the reliable source of truth. "
        "Do not compare against traditional methods or claim unseen improvements unless baseline metrics are added."
    )


def _build_additional_prompt(
    question,
    metrics,
    retrieved_contexts,
    pipeline_status,
    warnings,
    rule_based_analysis,
):
    """Build a prompt for short supplementary LLM commentary."""
    prompt_data = {
        "user_question": question,
        "metrics": _compact_metrics(metrics),
        "retrieved_contexts": _compact_contexts(retrieved_contexts),
        "pipeline_status": pipeline_status,
        "warnings_or_limitations": warnings,
        "rule_based_analysis": rule_based_analysis,
    }

    return (
        "You are writing a supplementary commentary. Do not modify or override "
        "the rule-based conclusions.\n\n"
        "Write under exactly these headings:\n"
        "### Additional Interpretation\n"
        "### Practical Implications\n"
        "### Suggested Next Checks\n\n"
        "Rules:\n"
        "- Do not repeat the Rule-Based Engineering Analysis.\n"
        "- Do not rewrite all metrics.\n"
        "- Do not invent numbers.\n"
        "- Do not override the rule-based conclusions.\n"
        "- Keep the output under 200 words.\n"
        "- Focus on what the metric pattern implies and what to check next.\n"
        "- Avoid exaggerated claims such as excellent, near-perfect, noise-free, real-time, or highly efficient.\n"
        "- Do not use numbered lists, because extra numbers can be confused with metrics.\n\n"
        "Input data:\n"
        f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"
    )


def _post_json(url, payload, headers=None, timeout=180):
    """POST JSON and return the parsed JSON response."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def _is_timeout_exception(exc):
    """Return True when an urllib/socket exception is a timeout."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


def _generate_with_openai(prompt, model, llm_timeout):
    """Generate commentary with OpenAI Chat Completions."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY is not set."

    selected_model = model or "gpt-4o-mini"
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful supplementary engineering commentator.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        response = _post_json(
            OPENAI_CHAT_COMPLETIONS_URL,
            payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=llm_timeout,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, OSError) as exc:
        if _is_timeout_exception(exc):
            return None, f"LLM call timed out after {llm_timeout} seconds."
        return None, f"OpenAI LLM report failed: {exc}."

    choices = response.get("choices") or []
    if not choices:
        return None, "OpenAI returned no choices."

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        return None, "OpenAI returned an empty analysis."
    return content, None


def _generate_with_ollama(prompt, model, llm_timeout):
    """Generate commentary with a local Ollama model."""
    selected_model = model or "llama3.1"
    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        response = _post_json(OLLAMA_GENERATE_URL, payload, timeout=llm_timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, OSError) as exc:
        if _is_timeout_exception(exc):
            return None, f"LLM call timed out after {llm_timeout} seconds."
        return None, f"Ollama LLM report failed: {exc}."

    content = (response.get("response") or "").strip()
    if not content:
        return None, "Ollama returned an empty analysis."
    return content, None


def _metric_number_strings(value):
    """Return acceptable text forms for numeric metric values."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        forms = {str(value)}
        for digits in range(0, 5):
            text = f"{float(value):.{digits}f}"
            forms.add(text)
            forms.add(text.rstrip("0").rstrip("."))
        return {form for form in forms if form not in {"", "-0"}}
    if isinstance(value, list):
        forms = set()
        for item in value:
            forms.update(_metric_number_strings(item))
        return forms
    if isinstance(value, dict):
        forms = set()
        for item in value.values():
            forms.update(_metric_number_strings(item))
        return forms
    return set()


def _allowed_numbers(metrics, rule_based_analysis):
    """Collect numeric strings allowed in supplementary LLM commentary."""
    allowed = set()
    for value in metrics.values():
        allowed.update(_metric_number_strings(value))
    allowed.update(re.findall(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?", rule_based_analysis))
    return allowed


def _extract_numbers(text):
    """Extract simple numeric literals from generated text."""
    return re.findall(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?", text)


def _split_sentences(text):
    """Split text into short validation units while preserving readability."""
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _previous_words(text_before_phrase):
    """Return the previous few English-like words before a phrase."""
    words = re.findall(r"[A-Za-z']+", text_before_phrase.lower())
    return words[-5:]


def contains_unsupported_exaggeration(sentence):
    """Return True when a sentence contains a positive unsupported hype claim."""
    lower_sentence = sentence.lower()
    for phrase in FORBIDDEN_PHRASES:
        start = 0
        while True:
            index = lower_sentence.find(phrase, start)
            if index == -1:
                break
            before = lower_sentence[max(0, index - 80) : index]
            previous_words = _previous_words(before)
            previous_text = " ".join(previous_words)
            if any(negation in previous_text for negation in ENGLISH_NEGATIONS):
                start = index + len(phrase)
                continue
            nearby_chars = sentence[max(0, index - 20) : index]
            if any(negation in nearby_chars for negation in CHINESE_NEGATIONS):
                start = index + len(phrase)
                continue
            return True
            start = index + len(phrase)
    return False


def _find_phrase(sentence):
    """Find the first forbidden phrase in a sentence."""
    lower_sentence = sentence.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower_sentence:
            return phrase
    return None


def _replacement_for_phrase(phrase):
    """Return deterministic replacement text for unsafe wording."""
    if phrase in {"near-perfect", "near perfect"}:
        return "The restoration quality should be interpreted as moderate based on PSNR and SSIM."
    if phrase == "noise-free":
        return "The results do not support a noise-free restoration claim."
    if phrase in {"excellent", "exceptional"}:
        return "The current metrics support a cautious engineering interpretation rather than an excellent-performance claim."
    if phrase in {
        "highly efficient",
        "real-time",
        "significant speed improvement",
        "significant improvement in processing speed",
    }:
        return "The current runtime indicates the implementation is slow and not real-time."
    if phrase == "deployment-ready":
        return "The current evidence does not support a deployment-ready claim."
    return "The current metrics support a cautious engineering interpretation."


def _sanitize_llm_output(raw_text):
    """Sanitize unsafe wording before validation."""
    sanitized_sentences = []
    log_entries = []
    changed = False

    for sentence in _split_sentences(raw_text):
        phrase = _find_phrase(sentence)
        if phrase is None:
            sanitized_sentences.append(sentence)
            continue

        unsupported = contains_unsupported_exaggeration(sentence)
        if unsupported:
            replacement = _replacement_for_phrase(phrase)
            sanitized_sentences.append(replacement)
            changed = True
            log_entries.append(
                {
                    "matched_phrase": phrase,
                    "sentence": sentence,
                    "action": "sanitized",
                    "replacement": replacement,
                }
            )
        else:
            sanitized_sentences.append(sentence)
            log_entries.append(
                {
                    "matched_phrase": phrase,
                    "sentence": sentence,
                    "action": "allowed",
                    "replacement": None,
                }
            )

    return "\n\n".join(sanitized_sentences).strip(), log_entries, changed


def _contradicts_rule_based(text, rule_based_analysis):
    """Reject common contradictions against deterministic conclusions."""
    lower_text = text.lower()
    lower_rule = rule_based_analysis.lower()

    if "not real-time" in lower_rule and contains_unsupported_exaggeration(lower_text):
        for phrase in ["real-time", "highly efficient", "significant speed improvement"]:
            if phrase in lower_text:
                return f"LLM contradicted slow-runtime conclusion with '{phrase}'."
    if "moderate restoration quality" in lower_rule:
        for phrase in ["excellent", "exceptional", "near-perfect", "near perfect", "noise-free"]:
            if phrase in lower_text and contains_unsupported_exaggeration(lower_text):
                return f"LLM contradicted moderate restoration quality with '{phrase}'."
    if "relatively low-confidence" in lower_rule:
        for phrase in ["high confidence", "strong confidence", "reliable detections"]:
            if phrase in lower_text:
                return f"LLM contradicted low-confidence detection judgment with '{phrase}'."

    return None


def _validate_sanitized_output(text, metrics, rule_based_analysis):
    """Validate supplementary LLM commentary after sanitization."""
    if not text.strip() or len(text.strip()) < 40:
        return "LLM output is empty or lacks useful supplementary commentary."

    allowed_numbers = _allowed_numbers(metrics, rule_based_analysis)
    unknown_numbers = [
        number for number in _extract_numbers(text) if number not in allowed_numbers
    ]
    if unknown_numbers:
        return (
            "LLM additional analysis mentioned numeric values not present in metrics "
            "or rule-based analysis: "
            + ", ".join(sorted(set(unknown_numbers))[:8])
        )

    contradiction = _contradicts_rule_based(text, rule_based_analysis)
    if contradiction:
        return contradiction

    return None


def _write_debug_file(output_dir, file_name, content):
    """Write an LLM debug artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    path.write_text(content, encoding="utf-8")
    return path


def _write_validation_log(output_dir, log_entries, final_result, final_reason=None):
    """Write validation details to outputs/llm_validation_log.txt."""
    lines = []
    for entry in log_entries:
        lines.append(f"matched phrase: {entry['matched_phrase']}")
        lines.append(f"sentence: {entry['sentence']}")
        lines.append(f"action: {entry['action']}")
        if entry.get("replacement"):
            lines.append(f"replacement: {entry['replacement']}")
        lines.append("")
    lines.append(f"final validation result: {final_result}")
    if final_reason:
        lines.append(f"final reason: {final_reason}")
    return _write_debug_file(output_dir, "llm_validation_log.txt", "\n".join(lines))


def _rejected_additional_analysis(reason):
    """Return the required rejection note for failed LLM commentary."""
    if reason:
        print(f"LLM additional analysis rejected reason: {reason}")
    return (
        "LLM Additional Analysis was rejected by validation. "
        "See outputs/llm_validation_log.txt for details."
    )


def _timeout_additional_analysis(llm_timeout):
    """Return the required timeout note."""
    return (
        f"LLM Additional Analysis timed out after {llm_timeout} seconds. "
        "The rule-based analysis above remains the reliable interpretation."
    )


def generate_llm_additional_analysis(
    question,
    metrics,
    retrieved_contexts,
    rule_based_analysis,
    restoration_outputs=None,
    detection_outputs=None,
    provider="none",
    model=None,
    output_dir="outputs",
    llm_timeout=180,
):
    """Generate optional supplementary LLM commentary."""
    provider = (provider or "none").lower()
    print("LLM Additional Analysis enabled: True")
    print(f"LLM provider: {provider}")
    print(f"LLM model: {model or 'default'}")
    print(f"LLM timeout seconds: {llm_timeout}")

    pipeline_status = _pipeline_status(restoration_outputs, detection_outputs)
    warnings = _collect_warnings(metrics, restoration_outputs, detection_outputs)

    if provider == "none":
        reason = "LLM provider is none."
        _write_debug_file(output_dir, "llm_raw_output.md", "")
        _write_debug_file(output_dir, "llm_sanitized_output.md", "")
        _write_validation_log(output_dir, [], "rejected", reason)
        return _rejected_additional_analysis(reason), reason

    prompt = _build_additional_prompt(
        question=question,
        metrics=metrics,
        retrieved_contexts=retrieved_contexts,
        pipeline_status=pipeline_status,
        warnings=warnings,
        rule_based_analysis=rule_based_analysis,
    )

    if provider == "openai":
        raw_analysis, warning = _generate_with_openai(prompt, model, llm_timeout)
    elif provider == "ollama":
        raw_analysis, warning = _generate_with_ollama(prompt, model, llm_timeout)
    else:
        raw_analysis = None
        warning = f"Unknown LLM provider '{provider}'."

    if not raw_analysis:
        if warning and warning.startswith("LLM call timed out"):
            print(warning)
            _write_debug_file(output_dir, "llm_raw_output.md", "")
            _write_debug_file(output_dir, "llm_sanitized_output.md", "")
            _write_validation_log(output_dir, [], "timeout", warning)
            return _timeout_additional_analysis(llm_timeout), warning

        _write_debug_file(output_dir, "llm_raw_output.md", "")
        _write_debug_file(output_dir, "llm_sanitized_output.md", "")
        _write_validation_log(output_dir, [], "rejected", warning)
        return _rejected_additional_analysis(warning), warning

    raw_path = _write_debug_file(output_dir, "llm_raw_output.md", raw_analysis)
    sanitized_analysis, log_entries, sanitized = _sanitize_llm_output(raw_analysis)
    sanitized_path = _write_debug_file(
        output_dir,
        "llm_sanitized_output.md",
        sanitized_analysis,
    )
    print(f"Saved raw LLM output to {raw_path}")
    print(f"Saved sanitized LLM output to {sanitized_path}")

    validation_warning = _validate_sanitized_output(
        sanitized_analysis,
        metrics,
        rule_based_analysis,
    )
    if validation_warning:
        _write_validation_log(output_dir, log_entries, "rejected", validation_warning)
        return _rejected_additional_analysis(validation_warning), validation_warning

    _write_validation_log(output_dir, log_entries, "accepted", None)
    if sanitized:
        return (
            "LLM Additional Analysis was sanitized by validation before being added.\n\n"
            + sanitized_analysis,
            None,
        )
    return sanitized_analysis, None

