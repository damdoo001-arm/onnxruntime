#!/usr/bin/env python3
"""Validate the generated KleidiAI page before publishing it."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: validate_generated_page.py PAGE")
    page = Path(sys.argv[1])
    text = page.read_text(encoding="utf-8")

    expected_front_matter = (
        "---\n"
        "layout: default\n"
        "title: Arm KleidiAI\n"
        "description: Arm KleidiAI micro-kernel compatibility in ONNX Runtime\n"
        "parent: Performance\n"
    )
    if not text.startswith(expected_front_matter):
        fail("page is not registered below Performance > Arm KleidiAI")
    if "<!doctype" in text.lower() or "<html" in text.lower() or "<body" in text.lower():
        fail("Pages output must be a Jekyll content fragment, not a complete document")
    if re.search(r"__[A-Z][A-Z0-9_]*__", text):
        fail("unresolved template placeholder")
    if "<h1>Arm KleidiAI compatibility in ONNX Runtime</h1>" not in text:
        fail("visible title does not identify Arm KleidiAI")
    if (
        '<a href="https://www.arm.com/markets/artificial-intelligence/software/kleidi">'
        "Arm KleidiAI micro-kernel library</a>" not in text
    ):
        fail("introductory Arm KleidiAI library link is missing")
    kai_release = re.search(
        r"KleidiAI tagged release: <a href=\"https://github\.com/ARM-software/"
        r"kleidiai/tree/(?P<tag>v\d+\.\d+\.\d+)\"><code>(?P=tag)</code>",
        text,
    )
    if not kai_release:
        fail("KleidiAI metadata does not use a stable release tag")
    ort_release = re.search(
        r"ONNX Runtime release: <a href=\"https://github\.com/microsoft/"
        r"onnxruntime/tree/(?P<tag>v\d+\.\d+\.\d+)\"><code>(?P=tag)</code>",
        text,
    )
    if not ort_release:
        fail("ONNX Runtime metadata does not use a stable release tag")
    if not re.search(
        r"ONNX Runtime KleidiAI pin: <a href=\"https://github\.com/ARM-software/"
        r"kleidiai/tree/(?P<pin>v\d+\.\d+\.\d+)\"><code>(?P=pin)</code>",
        text,
    ):
        fail("ONNX Runtime KleidiAI pin metadata is missing or malformed")
    if not re.search(
        r'Last updated date: <time datetime="\d{4}-\d{2}-\d{2}">\d{4}-\d{2}-\d{2}</time>',
        text,
    ):
        fail("last-updated metadata is missing or malformed")
    if re.search(
        r"github\.com/(?:ARM-software/kleidiai|microsoft/onnxruntime)/"
        r"(?:blob|commit)/[0-9a-f]{40}",
        text,
    ):
        fail("page contains an intermediate source commit reference")

    match = re.search(
        r'<script type="application/json" id="kernelData">(.*?)</script>', text, re.DOTALL
    )
    if not match:
        fail("kernelData payload is missing")
    records = json.loads(match.group(1))
    if not records:
        fail("kernel inventory is empty")
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)):
        fail("kernel names are not unique")
    if 'type="checkbox"' not in text:
        fail("checkbox filters are missing")
    if "<th>At pin</th>" in text or "Pin availability" in text:
        fail("removed pin-availability field is still present")
    if "<th>ISA" in text or "['isa','ISA']" in text:
        fail("ISA field was not replaced by Extension")
    if '<th>Extension Type</th>' not in text:
        fail("Extension Type column is missing")
    if "['ortOps','ONNX operator']" not in text:
        fail("ONNX operator filter is missing")
    if "class=\"expand-row\"" not in text:
        fail("operation rows are not expandable")
    if not all(record.get("extension") for record in records):
        fail("one or more kernel records are missing extension metadata")
    if any(record.get("extension") == "NEON" for record in records):
        fail("NEON kernels must not be included in the compatibility page")
    if not all(record.get("tileSource") for record in records):
        fail("one or more kernel records are missing tile provenance")
    missing_generic_tiles = [
        record.get("name")
        for record in records
        if record.get("operationKey") != "dwconv" and record.get("tile") == "—"
    ]
    if missing_generic_tiles:
        fail(f"non-depthwise kernels have unresolved tiles: {missing_generic_tiles}")
    invalid_derived_tiles = [
        record.get("name")
        for record in records
        if record.get("tileSource") == "get_m_step/get_n_step"
        and not re.fullmatch(r"\d+vlx\d+vl", str(record.get("tile")))
    ]
    if invalid_derived_tiles:
        fail(f"invalid source-derived tiles: {invalid_derived_tiles}")
    if "<th>Misc</th>" in text or any("misc" in record for record in records):
        fail("removed misc field is still present")
    variable = [record for record in records if record.get("tileKind") == "Variable"]
    elastic_kernel = (
        "kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa"
    )
    if len(variable) != 1 or variable[0]["name"] != elastic_kernel:
        fail("Variable must identify only KleidiAI's elastic GEMM implementation")
    if not all(isinstance(record.get("lhsPacks"), list) for record in records):
        fail("one or more kernel records are missing LHS packer associations")
    if not all(isinstance(record.get("rhsPacks"), list) for record in records):
        fail("one or more kernel records are missing RHS packer associations")
    if not any(record.get("rhsPacks") for record in records):
        fail("no RHS packer associations were discovered")
    packers = [
        packer
        for record in records
        for side in ("lhsPacks", "rhsPacks")
        for packer in record.get(side, [])
    ]
    if not all(isinstance(packer.get("details"), list) for packer in packers):
        fail("one or more packers are missing input/output details")
    detail_values = {detail for packer in packers for detail in packer["details"]}
    for expected in ("Signed", "Unsigned", "Symmetric", "Asymmetric"):
        if not any(expected.lower() in detail.lower() for detail in detail_values):
            fail(f"packer details do not expose {expected.lower()} data")
    rhs_orientations = {
        packer.get("orientation")
        for record in records
        for packer in record.get("rhsPacks", [])
    }
    if not {"K×N", "N×K"}.issubset(rhs_orientations) or "item.orientation" not in text:
        fail("RHS packer links are missing K×N/N×K orientation labels")
    if not any(
        len(record.get("lhsPacks", [])) > 1 or len(record.get("rhsPacks", [])) > 1
        for record in records
    ):
        fail("authoritative many-to-one packer associations are missing")
    if not all(isinstance(record.get("ortOps"), list) for record in records):
        fail("one or more kernel records are missing ONNX operator metadata")
    if not all(record.get("ortOps") for record in records if record.get("status") == "integrated"):
        fail("an integrated kernel has no framework-level ONNX operator mapping")
    integrated_evidence = [
        record.get("evidence", [])
        for record in records
        if record.get("status") == "integrated"
    ]
    if not all(
        len(items) == 1
        and items[0].get("label") == "Invocation"
        and "kai_ukernel_interface.cpp" not in items[0].get("url", "")
        for items in integrated_evidence
    ):
        fail("integrated kernels must link once to runtime invocation, not registration")
    if "integration 2" in text:
        fail("duplicate integration evidence is still present")
    if '<td></td><td colspan=' in text or '<td colspan="6">' not in text:
        fail("expanded rows must span the full table width without indentation")
    depthwise = [record for record in records if record.get("operationKey") == "dwconv"]
    if not all(isinstance(record.get("outputDetails"), list) for record in records):
        fail("one or more kernels are missing output details")
    if not depthwise or not all(
        record.get("convKernelSize") != "—"
        and record.get("stride") != "—"
        and record.get("inputPath") in {"Indirect", "Planar"}
        and record.get("rhsPacks")
        for record in depthwise
    ):
        fail("depthwise kernels are missing geometry, input path, or RHS details")
    depthwise_template = re.search(
        r"if\(isDepthwise\)return `(.*?)`;return `<table", text, re.DOTALL
    )
    if not depthwise_template:
        fail("dedicated depthwise detail table is missing")
    depthwise_html = depthwise_template.group(1)
    if (
        "<th>Conv kernel size</th>" not in depthwise_html
        or "<th>Stride</th>" not in depthwise_html
        or "<th>Input path</th>" not in depthwise_html
        or "<th>Conv Details</th>" not in depthwise_html
        or "<th>RHS Details</th>" not in depthwise_html
        or "<th>Tile</th>" in depthwise_html
    ):
        fail("depthwise detail columns must show geometry, input path, and RHS details")
    if ">Kernel</a>" not in text or ":'Packer'" not in text:
        fail("expanded rows must use capitalized Kernel and Packer link labels")
    if "<th>GEMM Details</th>" not in text or "computeDetails(kernel)" not in text:
        fail("expanded rows must expose compute-kernel output details")
    if "<th>LHS Details</th>" not in text or "<th>RHS Details</th>" not in text:
        fail("expanded matmul rows must expose LHS and RHS details")
    if any("availability" in record for record in records):
        fail("removed pin-availability data is still embedded")
    packing_operations = {
        "Depthwise RHS pack",
        "LHS pack",
        "Packing",
        "RHS pack K×N",
        "RHS pack N×K",
    }
    if any(record.get("operation") in packing_operations for record in records):
        fail("packing-only operations are still embedded")
    if "['tile','Tile']" in text or '<th>ISA / tile</th>' in text:
        fail("tile is still exposed as a top-level search or table field")
    if "${record.tile}" in text:
        fail("tile is still included as a dedicated search term")
    if any(
        "packed" in str(record.get(key, "")).lower()
        for record in records
        for key in ("output", "lhs", "rhs")
    ):
        fail("packed datatypes must use the compact (P) notation")

    print(f"Validated {page}: {len(records)} unique kernels")


if __name__ == "__main__":
    main()
