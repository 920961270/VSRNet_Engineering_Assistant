"""Template-based markdown report generation."""

from pathlib import Path


def _format_metric_value(value):
    """Return a readable metric value for the report."""
    if value is None:
        return "Not available"
    if isinstance(value, list):
        return "<br>".join(str(item) for item in value) if value else "None"
    return str(value)


def _format_contexts(retrieved_contexts):
    """Format retrieved RAG contexts as markdown."""
    if not retrieved_contexts:
        return "No project context was retrieved."

    lines = []
    for index, context in enumerate(retrieved_contexts, start=1):
        lines.append(
            f"### Context {index}: {context['source']} "
            f"(chunk {context['chunk_id']}, score {context['score']:.4f})"
        )
        lines.append("")
        lines.append(context["text"].strip())
        lines.append("")

    return "\n".join(lines).strip()


def _status_line(name, outputs):
    """Describe whether a pipeline stage used the real implementation."""
    if not outputs:
        return f"- {name}: Unknown"

    if outputs.get("skipped") and outputs.get("reused"):
        if outputs.get("restored_video"):
            return f"- {name}: Skipped, reused existing restored video: {outputs['restored_video']}"
        if outputs.get("detection_csv"):
            return f"- {name}: Skipped, reused existing detection CSV: {outputs['detection_csv']}"
        if outputs.get("metrics_json"):
            return f"- {name}: Skipped, loaded existing metrics JSON: {outputs['metrics_json']}"
        return f"- {name}: Skipped, reused existing output"

    if outputs.get("used_placeholder"):
        warning = outputs.get("warning") or "No warning message was provided."
        return f"- {name}: Placeholder used. Warning: {warning}"

    return f"- {name}: Real pipeline used."


def generate_markdown_report(
    question,
    metrics,
    retrieved_contexts,
    output_dir,
    restoration_outputs=None,
    detection_outputs=None,
    metrics_outputs=None,
    rule_based_analysis=None,
    llm_additional_analysis=None,
):
    """Generate a Markdown engineering report without calling an LLM API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "engineering_report.md"

    context_block = _format_contexts(retrieved_contexts)
    metric_rows = "\n".join(
        f"| {key} | {_format_metric_value(value)} |" for key, value in metrics.items()
    )

    warnings = []
    if restoration_outputs and restoration_outputs.get("warning"):
        warnings.append(restoration_outputs["warning"])
    if detection_outputs and detection_outputs.get("warning"):
        warnings.append(detection_outputs["warning"])
    if metrics_outputs and metrics_outputs.get("warning"):
        warnings.append(metrics_outputs["warning"])
    metric_warnings = metrics.get("warnings") or []
    if isinstance(metric_warnings, list):
        warnings.extend(metric_warnings)
    else:
        warnings.append(str(metric_warnings))
    warnings_block = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None"
    rule_based_section = rule_based_analysis or (
        "### Direct Answer\n"
        "Rule-based analysis was not generated because metrics were unavailable."
    )
    llm_section = ""
    if llm_additional_analysis:
        llm_section = (
            "## LLM Additional Analysis\n\n"
            "This section is a supplementary interpretation layer. It does not replace, edit, "
            "or override the rule-based analysis above.\n\n"
            f"{llm_additional_analysis}\n"
        )

    report = f"""# VSRNet Engineering Analysis Report

## User Question

{question}

## Pipeline Status

{_status_line("VSRNet restoration", restoration_outputs)}
{_status_line("YOLO detection/tracking", detection_outputs)}
{_status_line("Metrics", metrics_outputs)}
- RAG retrieval: Real pipeline used.
- Report generation: Real pipeline used.

## Pipeline Summary

This Stage 2 pipeline attempts to run real VSRNet restoration through the existing local repository script, then attempts YOLO detection or tracking with Ultralytics if it is installed. If a required heavy component is missing, the assistant records a warning, writes placeholder outputs, and continues to metrics, RAG retrieval, and report generation.

## Metric Summary

| Metric | Value |
| --- | --- |
{metric_rows}

## Retrieved VSRNet Knowledge

{context_block}

## Rule-Based Engineering Analysis

This section is deterministic and is the source of truth.

{rule_based_section}

{llm_section}

## Recommendations

- Place the VSRNet checkpoint at the path passed through `--vsrnet-checkpoint`.
- Provide an input video through `--input` when testing real restoration.
- Install Ultralytics only when YOLO detection or tracking is needed.
- Add more project notes, experiment logs, and evaluation observations to `documents/` to improve RAG retrieval.
- Treat the raw metrics and rule-based analysis as more reliable than any LLM commentary.

## Warnings / Limitations

{warnings_block}

## Next Steps

1. Confirm the restored output video visually.
2. Inspect `detection_results.csv` for detection quality and tracking IDs.
3. Compare GT and restored video with aligned frames when PSNR/SSIM are needed.
4. Add real experiment notes to the document corpus.
5. Upgrade the report writer later if an LLM-based explanation layer is required.
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Markdown report saved to {report_path}")

    return report_path
