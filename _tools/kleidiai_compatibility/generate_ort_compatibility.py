#!/usr/bin/env python3
"""Generate a GitHub-renderable KleidiAI/ONNX Runtime compatibility report."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path

import generate_kernel_status as report


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(
    os.environ.get(
        "KLEIDIAI_MARKDOWN_STANDALONE_OUTPUT",
        ROOT / "kleidiai_onnxruntime_compatibility.md",
    )
).expanduser().resolve()
DOCS_OUTPUT = Path(
    os.environ.get(
        "KLEIDIAI_MARKDOWN_OUTPUT",
        ROOT / "onnxruntime" / "docs" / "performance" / "kleidiai" / "README.md",
    )
).expanduser().resolve()


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def integration_status(variant: report.Variant) -> tuple[str, str]:
    result = variant.ort
    if result.active:
        return "integrated", "✅ Integrated"
    if result.references:
        return "referenced", "🔗 Referenced only"
    if result.pr_candidates:
        return "pr", "🟣 PR available"
    return "not-integrated", "— Not integrated"


def evidence_links(variant: report.Variant, ort_revision: str) -> str:
    result = variant.ort
    links = []
    for item in (result.active or result.references)[:2]:
        url = f"{report.ORT_GITHUB}/blob/{ort_revision}/{item.path}#L{item.line}"
        links.append(f"[code:{item.line}]({url} \"{markdown_cell(item.path)}:{item.line}\")")
    for pull_request in report.unique_prs(result.pr_candidates):
        links.append(f"[PR #{pull_request.number}]({pull_request.url})")
    return "<br>".join(links) or "—"


def generate() -> tuple[str, dict[str, int]]:
    variants = report.inventory()
    ort_pin = report.discover_ort_pin()
    report.scan_framework(
        variants,
        report.ORT_ROOT,
        [report.ORT_ROOT / "onnxruntime" / "core" / "mlas" / "lib"],
        "ort",
    )
    pr_audited_at = report.apply_pr_candidates(variants)
    for variant in variants:
        variant.ort.available_primary = variant.relative_c in ort_pin.files

    kai_revision = report.run_git(report.KAI_ROOT, "rev-parse", "HEAD")
    kai_describe = report.describe_revision(report.KAI_ROOT)
    ort_revision = report.run_git(report.ORT_ROOT, "rev-parse", "HEAD")
    ort_release, ort_release_label = report.latest_stable_release(report.ORT_ROOT)

    statuses = Counter(integration_status(variant)[0] for variant in variants)
    counts = {
        "microkernels": len(variants),
        "integrated": statuses["integrated"],
        "referenced": statuses["referenced"],
        "pr": statuses["pr"],
        "not_integrated": statuses["not-integrated"],
        "unavailable_at_pin": sum(not variant.ort.available_primary for variant in variants),
    }

    lines = [
        "# KleidiAI compatibility in ONNX Runtime",
        "",
        (
            "GitHub-native snapshot of public KleidiAI micro-kernels and their exact "
            "ONNX Runtime MLAS integration status."
        ),
        "",
        f"- KleidiAI: [`{kai_describe}`]({report.KAI_GITHUB}/commit/{kai_revision})",
        f"- ONNX Runtime: [`{ort_release_label}`]({report.ORT_GITHUB}/releases/tag/{ort_release})",
        f"- ONNX Runtime KleidiAI pin: [`{ort_pin.label}`]({report.KAI_GITHUB}/tree/{ort_pin.revision})",
        f"- Open pull requests audited: `{pr_audited_at}`",
        "",
        "## Summary",
        "",
        "| Classification | Micro-kernels |",
        "| --- | ---: |",
        f"| ✅ Integrated | {counts['integrated']} |",
        f"| 🔗 Referenced only | {counts['referenced']} |",
        f"| 🟣 PR available | {counts['pr']} |",
        f"| — Not integrated | {counts['not_integrated']} |",
        f"| Unavailable in ONNX Runtime's `{ort_pin.label}` pin | {counts['unavailable_at_pin']} |",
        f"| **Total public KleidiAI micro-kernels** | **{counts['microkernels']}** |",
        "",
        "> Integration and pin availability are separate. A kernel can be absent from the current "
        "ONNX Runtime pin even when a newer integration pull request exists.",
        "",
        "## Kernel compatibility",
        "",
        "Sections are grouped by user-facing operation. Kernel symbols are hidden behind compact "
        "source links; hover a link to see its full symbol.",
        "",
    ]

    by_operation: dict[str, list[report.Variant]] = defaultdict(list)
    for variant in variants:
        by_operation[variant.operation].append(variant)

    for operation in sorted(by_operation):
        operation_variants = sorted(by_operation[operation], key=lambda item: item.raw_name)
        lines.extend(
            [
                f"### {operation} ({len(operation_variants)})",
                "",
                "<details>",
                f"<summary>Show {len(operation_variants)} micro-kernel"
                f"{'s' if len(operation_variants) != 1 else ''}</summary>",
                "",
                "| Kernel | Type signature | ISA / tile | At pin | ONNX Runtime |",
                "| :---: | --- | --- | :---: | --- |",
            ]
        )
        for variant in operation_variants:
            signature = " · ".join(item.raw.upper() for item in variant.descriptors) or "—"
            _status_key, status_label = integration_status(variant)
            kernel_url = f"{report.KAI_GITHUB}/blob/{kai_revision}/{variant.relative_c}"
            kernel = f'[source]({kernel_url} "{variant.raw_name}")'
            isa_tile = variant.isa
            if variant.tile != "—":
                isa_tile += f" · `{variant.tile}`"
            available = "✅" if variant.ort.available_primary else "❌"
            evidence = evidence_links(variant, ort_revision)
            ort_status = status_label if evidence == "—" else f"{status_label}<br>{evidence}"
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(value)
                    for value in (
                        kernel,
                        f"`{signature}`",
                        isa_tile,
                        available,
                        ort_status,
                    )
                )
                + " |"
            )
        lines.extend(["", "</details>", ""])

    return "\n".join(lines), counts


def main() -> None:
    output, counts = generate()
    for destination in (OUTPUT, DOCS_OUTPUT):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
        print(f"Generated {destination}")
    print(counts)


if __name__ == "__main__":
    main()
