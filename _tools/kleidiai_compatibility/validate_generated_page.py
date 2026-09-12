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
        "title: KleidiAI\n"
        "description: KleidiAI micro-kernel compatibility in ONNX Runtime\n"
        "parent: Performance\n"
    )
    if not text.startswith(expected_front_matter):
        fail("page is not registered below Performance > KleidiAI")
    if "<!doctype" in text.lower() or "<html" in text.lower() or "<body" in text.lower():
        fail("Pages output must be a Jekyll content fragment, not a complete document")
    if re.search(r"__[A-Z][A-Z0-9_]*__", text):
        fail("unresolved template placeholder")

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
    if "class=\"expand-row\"" not in text:
        fail("operation rows are not expandable")
    if not all(record.get("extension") for record in records):
        fail("one or more kernel records are missing extension metadata")
    if not all(record.get("misc") for record in records):
        fail("one or more kernel records are missing misc metadata")
    if {record.get("tileKind") for record in records} != {"Fixed", "Variable"}:
        fail("tile-size classification must contain Fixed and Variable kernels")
    if not any(record.get("misc") == "Cortex-A55" for record in records):
        fail("Cortex-A55 specialization metadata is missing")
    if not all(isinstance(record.get("lhsPacks"), list) for record in records):
        fail("one or more kernel records are missing LHS packer associations")
    if not all(isinstance(record.get("rhsPacks"), list) for record in records):
        fail("one or more kernel records are missing RHS packer associations")
    if not any(record.get("rhsPacks") for record in records):
        fail("no RHS packer associations were discovered")
    if '<td></td><td colspan=' in text or '<td colspan="7">' not in text:
        fail("expanded rows must span the full table width without indentation")
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
