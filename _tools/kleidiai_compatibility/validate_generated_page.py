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

    print(f"Validated {page}: {len(records)} unique kernels")


if __name__ == "__main__":
    main()
