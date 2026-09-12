#!/usr/bin/env python3
"""Regenerate and validate every KleidiAI compatibility page output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]


def configured_path(argument: str | None, variable: str, default: Path | None) -> Path | None:
    value = argument or os.environ.get(variable)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve() if default is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate and validate the KleidiAI compatibility page."
    )
    parser.add_argument("--kleidiai-root", help="KleidiAI source checkout")
    parser.add_argument("--onnxruntime-root", help="ONNX Runtime source checkout")
    parser.add_argument("--pages-output", help="Jekyll HTML page destination")
    parser.add_argument("--standalone-output", help="Standalone HTML destination")
    parser.add_argument("--markdown-output", help="GitHub Markdown destination")
    parser.add_argument(
        "--markdown-standalone-output", help="Standalone Markdown destination"
    )
    parser.add_argument(
        "--skip-markdown", action="store_true", help="Generate only the HTML page"
    )
    args = parser.parse_args()

    kleidiai_root = configured_path(
        args.kleidiai_root, "KLEIDIAI_SOURCE_ROOT", None
    )
    onnxruntime_root = configured_path(
        args.onnxruntime_root, "ONNXRUNTIME_SOURCE_ROOT", None
    )
    pages_output = configured_path(
        args.pages_output,
        "KLEIDIAI_PAGES_OUTPUT",
        REPOSITORY_ROOT / "docs" / "performance" / "kleidiai" / "index.html",
    )
    standalone_output = configured_path(
        args.standalone_output,
        "KLEIDIAI_STANDALONE_OUTPUT",
        None,
    )
    markdown_output = configured_path(
        args.markdown_output,
        "KLEIDIAI_MARKDOWN_OUTPUT",
        REPOSITORY_ROOT / "docs" / "performance" / "kleidiai" / "README.md",
    )
    markdown_standalone_output = configured_path(
        args.markdown_standalone_output,
        "KLEIDIAI_MARKDOWN_STANDALONE_OUTPUT",
        None,
    )

    if kleidiai_root is None or not (kleidiai_root / "kai" / "ukernels").is_dir():
        parser.error("--kleidiai-root must identify a KleidiAI source checkout")
    if onnxruntime_root is None or not (onnxruntime_root / ".git").exists():
        parser.error("--onnxruntime-root must identify an ONNX Runtime git checkout")
    assert pages_output is not None
    assert markdown_output is not None

    # The standalone reports are useful when explicitly requested, but they
    # should not dirty a checkout during the normal page-update command.
    with tempfile.TemporaryDirectory(prefix="kleidiai-page-") as temporary_directory:
        scratch = Path(temporary_directory)
        standalone_output = standalone_output or scratch / "compatibility.html"
        markdown_standalone_output = (
            markdown_standalone_output or scratch / "compatibility.md"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "KLEIDIAI_SOURCE_ROOT": str(kleidiai_root),
                "ONNXRUNTIME_SOURCE_ROOT": str(onnxruntime_root),
                "KLEIDIAI_PAGES_OUTPUT": str(pages_output),
                "KLEIDIAI_STANDALONE_OUTPUT": str(standalone_output),
                "KLEIDIAI_MARKDOWN_OUTPUT": str(markdown_output),
                "KLEIDIAI_MARKDOWN_STANDALONE_OUTPUT": str(
                    markdown_standalone_output
                ),
            }
        )
        commands = [
            [sys.executable, str(ROOT / "generate_ort_compatibility_html.py")]
        ]
        if not args.skip_markdown:
            commands.append(
                [sys.executable, str(ROOT / "generate_ort_compatibility.py")]
            )
        commands.append(
            [
                sys.executable,
                str(ROOT / "validate_generated_page.py"),
                str(pages_output),
            ]
        )
        for command in commands:
            subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
