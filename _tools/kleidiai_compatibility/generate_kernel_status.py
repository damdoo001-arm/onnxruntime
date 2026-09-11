#!/usr/bin/env python3
"""Generate a self-contained KleidiAI integration status report.

The report inventories public KleidiAI micro-kernels (the same C-file anchored
units used by KleidiAI's name checker), groups compatible variants, and finds
exact production references in local XNNPACK, ONNX Runtime, MNN, and llama.cpp
checkouts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent


def configured_path(variable: str, default: Path) -> Path:
    """Resolve an optional CI path override while preserving local defaults."""

    return Path(os.environ.get(variable, default)).expanduser().resolve()


KAI_ROOT = configured_path("KLEIDIAI_SOURCE_ROOT", ROOT / "kleidiai")
KAI_UKERNELS = KAI_ROOT / "kai" / "ukernels"
XNN_ROOT = configured_path("XNNPACK_SOURCE_ROOT", ROOT / "XNNPACK")
ORT_ROOT = configured_path("ONNXRUNTIME_SOURCE_ROOT", ROOT / "onnxruntime")
MNN_ROOT = configured_path("MNN_SOURCE_ROOT", ROOT / "MNN")
LLAMA_ROOT = configured_path("LLAMA_SOURCE_ROOT", ROOT / "llama.cpp")
KAI_GITHUB = "https://github.com/ARM-software/kleidiai"
XNN_GITHUB = "https://github.com/google/XNNPACK"
ORT_GITHUB = "https://github.com/microsoft/onnxruntime"
MNN_GITHUB = "https://github.com/alibaba/MNN"
LLAMA_GITHUB = "https://github.com/ggml-org/llama.cpp"
PR_CANDIDATES_PATH = configured_path(
    "KLEIDIAI_PR_CANDIDATES", ROOT / "open_kernel_prs.json"
)

TECHS = ("neon", "sve", "sve2", "sve2p1", "sme", "sme2", "sme2p1")
FEATURES = ("dotprod", "i8mm")
INSTRUCTIONS = ("dot", "i8mm", "mla", "mmla", "mopa", "mop4a", "sdot")
UARCHES = ("cortexa55",)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".s", ".inc"}

TYPE_LABELS = {
    "bf16": "BF16",
    "f16": "FP16",
    "f32": "FP32",
    "fp32": "FP32",
    "i2": "INT2",
    "i4": "INT4",
    "i8": "INT8",
    "i16": "INT16",
    "i32": "INT32",
    "u2": "UINT2",
    "u4": "UINT4",
    "u8": "UINT8",
    "u16": "UINT16",
    "u32": "UINT32",
    "x2": "raw 2-bit",
    "x4": "raw 4-bit",
    "x8": "raw 8-bit",
    "x16": "raw 16-bit",
    "x32": "raw 32-bit",
}

AXIS_LABELS = {
    "cx": "per-channel",
    "dx": "per-dimension",
    "c32": "per-block (32)",
    "d32": "per-dimension block (32)",
}

PACK_PREFIXES = (
    ("rhs_dwconv_pack_", "Depthwise RHS pack", "dwconv-rhs-pack"),
    ("rhs_pack_kxn_", "RHS pack K×N", "rhs-pack-kxn"),
    ("rhs_pack_nxk_", "RHS pack N×K", "rhs-pack-nxk"),
    ("lhs_pack_", "LHS pack", "lhs-pack"),
)


def fail(message: str) -> None:
    raise SystemExit(message)


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        fail(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def short_revision(repo: Path) -> str:
    return run_git(repo, "rev-parse", "--short=12", "HEAD")


def describe_revision(repo: Path) -> str:
    described = run_git(repo, "describe", "--tags", "--always", check=False)
    return described or short_revision(repo)


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "item"


def strip_comments(text: str) -> str:
    """Remove C/C++ comments while retaining line numbers."""

    def blank_block(match: re.Match[str]) -> str:
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    text = re.sub(r"/\*.*?\*/", blank_block, text, flags=re.DOTALL)
    return "\n".join(re.sub(r"//.*", "", line) for line in text.splitlines())


@dataclass(frozen=True)
class Descriptor:
    raw: str
    quantization: str | None
    base_type: str
    axis: str | None
    packed: bool

    @property
    def type_label(self) -> str:
        return TYPE_LABELS.get(self.base_type, self.base_type.upper())

    @property
    def compact(self) -> str:
        bits = [self.type_label]
        if self.quantization == "qa":
            bits.append("asymmetric")
        elif self.quantization == "qs":
            bits.append("symmetric")
        if self.axis:
            bits.append(AXIS_LABELS.get(self.axis, self.axis))
        if self.packed:
            bits.append("packed")
        return " · ".join(bits)


DESCRIPTOR_RE = re.compile(
    r"^(?P<quant>qa|qs)?"
    r"(?P<base>bf\d+|fp\d+|f\d+|i\d+|u\d+|x\d+)"
    r"(?P<axis>cx|dx|c\d+|d\d+)?"
    r"(?P<packed>p)?"
)


def parse_descriptor(raw: str) -> Descriptor:
    match = DESCRIPTOR_RE.match(raw)
    if not match:
        return Descriptor(raw, None, raw, None, False)
    return Descriptor(
        raw=raw,
        quantization=match.group("quant"),
        base_type=match.group("base"),
        axis=match.group("axis"),
        packed=bool(match.group("packed")),
    )


def simplify_descriptor(raw: str) -> str:
    match = DESCRIPTOR_RE.match(raw)
    if not match:
        return raw
    return "".join(
        part or ""
        for part in (
            match.group("quant"),
            match.group("base"),
            match.group("axis"),
            match.group("packed"),
        )
    )


@dataclass(frozen=True)
class Evidence:
    path: str
    line: int
    text: str
    kind: str


@dataclass(frozen=True)
class PullRequest:
    framework: str
    repository: str
    number: int
    title: str
    url: str
    draft: bool


@dataclass
class FrameworkResult:
    active: list[Evidence] = field(default_factory=list)
    references: list[Evidence] = field(default_factory=list)
    runtime_selected: bool = False
    available_primary: bool = True
    available_secondary: bool | None = None
    pr_candidates: list[PullRequest] = field(default_factory=list)


@dataclass(frozen=True)
class ReleaseHistory:
    added_commit: str
    added_date: str
    first_stable_release: str | None
    latest_stable_release: str | None
    explicit_deprecation: bool


@dataclass
class Variant:
    identity: str
    raw_name: str
    canonical_name: str
    relative_c: str
    physical_family: str
    canonical_family: str
    operation: str
    operation_key: str
    descriptors: list[Descriptor]
    isa: str
    feature_instruction: str
    tile: str
    layouts: list[str]
    files: list[str]
    helper_files: list[str]
    group_key: tuple[str, ...]
    release_history: ReleaseHistory | None = None
    xnn: FrameworkResult = field(default_factory=FrameworkResult)
    ort: FrameworkResult = field(default_factory=FrameworkResult)
    mnn: FrameworkResult = field(default_factory=FrameworkResult)
    llama: FrameworkResult = field(default_factory=FrameworkResult)

    @property
    def is_compute(self) -> bool:
        return self.operation_key in {"gemm", "gemv", "imatmul", "dwconv"}


@dataclass
class Group:
    key: tuple[str, ...]
    operation: str
    operation_key: str
    family: str
    descriptors: list[Descriptor]
    variants: list[Variant] = field(default_factory=list)


@dataclass(frozen=True)
class Pin:
    label: str
    revision: str
    files: frozenset[str]


def load_naming_exceptions() -> tuple[dict[str, object], dict[str, object]]:
    tools_dir = KAI_ROOT / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        from naming.issues import KNOWN_DIRECTORY_PROBLEMS, KNOWN_UKERNEL_PROBLEMS
    except ImportError as exc:
        fail(f"Could not import KleidiAI naming rules: {exc}")
    return KNOWN_UKERNEL_PROBLEMS, KNOWN_DIRECTORY_PROBLEMS


def operation_and_descriptors(canonical_family: str) -> tuple[str, str, list[Descriptor]]:
    choices = (
        ("imatmul_clamp_", "Indirect GEMM + clamp", "imatmul"),
        ("matmul_clamp_", "Matmul + clamp", "matmul"),
        ("matmul_", "Matmul", "matmul"),
        ("dwconv_", "Depthwise convolution + clamp", "dwconv"),
    )
    for prefix, label, key in choices:
        if canonical_family.startswith(prefix):
            raw_descriptors = canonical_family[len(prefix) :].split("_")
            return label, key, [parse_descriptor(value) for value in raw_descriptors]
    return canonical_family.replace("_", " ").title(), "other", []


def classify_direct_matmul(canonical_family: str, tile: str) -> tuple[str, str]:
    """Classify fixed-single-row matmul kernels as GEMV and the rest as GEMM."""

    is_gemv = tile.startswith("1x")
    operation_key = "gemv" if is_gemv else "gemm"
    operation = "GEMV" if is_gemv else "GEMM"
    if canonical_family.startswith("matmul_clamp_"):
        operation += " + clamp"
    return operation, operation_key


def trim_implementation_suffix(tokens: list[str]) -> list[str]:
    suffixes = set(TECHS) | set(FEATURES) | set(INSTRUCTIONS) | set(UARCHES)
    trimmed = list(tokens)
    while trimmed and trimmed[-1] in suffixes:
        trimmed.pop()
    return trimmed


def parse_pack_group(canonical_name: str) -> tuple[tuple[str, ...], str, str, list[Descriptor]]:
    core = canonical_name.removeprefix("kai_")
    for prefix, label, key in PACK_PREFIXES:
        if core.startswith(prefix):
            raw_descriptors = trim_implementation_suffix(core[len(prefix) :].split("_"))
            simplified = tuple(simplify_descriptor(value) for value in raw_descriptors)
            descriptors = [parse_descriptor(value) for value in simplified]
            return (key, *simplified), label, key, descriptors
    # This should be unreachable after applying KleidiAI's known-name corrections.
    raw = tuple(trim_implementation_suffix(core.split("_")))
    return ("pack", *raw), "Packing", "pack", [parse_descriptor(value) for value in raw]


def extract_impl_metadata(canonical_name: str) -> tuple[str, str, str, list[str]]:
    tokens = canonical_name.split("_")
    isa = next((token.upper() for token in tokens if token in TECHS), "Scalar / generic")
    feature_instruction = ", ".join(
        dict.fromkeys(token.upper() for token in tokens if token in set(FEATURES) | set(INSTRUCTIONS))
    ) or "—"
    tile = "—"
    for index, token in enumerate(tokens):
        if token in TECHS and index:
            candidate = tokens[index - 1]
            if re.fullmatch(r"(?:\d+(?:vl|vs)?|mr|nr)x(?:\d+(?:vl|vs)?|mr|nr)(?:x\d+)?", candidate):
                tile = candidate
            break
    layouts = sorted(
        set(
            re.findall(
                r"p(?:\d+(?:vl|vs)?|mr|nr)x(?:\d+(?:vl|vs)?|mr|nr)(?:s1s0|s4s0|s16s0)?",
                canonical_name,
            )
        )
    )
    return isa, feature_instruction, tile, layouts


def collect_source_bundle(c_path: Path, all_assembly: list[Path]) -> tuple[list[str], list[str]]:
    raw_name = c_path.stem
    candidates = [c_path]
    header = c_path.with_suffix(".h")
    assembly = c_path.with_name(f"{raw_name}_asm.S")
    if header.exists():
        candidates.append(header)
    if assembly.exists():
        candidates.append(assembly)

    helpers: list[Path] = []
    c_text = c_path.read_text(encoding="utf-8", errors="replace")
    exact_assembly = {assembly.resolve()} if assembly.exists() else set()
    for asm_path in all_assembly:
        if asm_path.parent != c_path.parent or asm_path.resolve() in exact_assembly:
            continue
        symbol = asm_path.stem.removesuffix("_asm")
        alternate = (
            "kai_" + symbol.removeprefix("kai_kernel_")
            if symbol.startswith("kai_kernel_")
            else "kai_kernel_" + symbol.removeprefix("kai_")
        )
        if alternate == raw_name:
            # Two U8 bundles use a kai_kernel_* assembly filename for a
            # kai_* public C wrapper. It is the bundle implementation, not a
            # separately callable public micro-kernel.
            candidates.append(asm_path)
            continue
        patterns = (
            re.compile(r"(?<![A-Za-z0-9])" + re.escape(symbol) + r"(?![A-Za-z0-9_])"),
            re.compile(r"(?<![A-Za-z0-9])" + re.escape(alternate) + r"(?![A-Za-z0-9_])"),
        )
        if any(pattern.search(c_text) for pattern in patterns):
            helpers.append(asm_path)

    relative = lambda path: path.relative_to(KAI_ROOT).as_posix()
    return [relative(path) for path in candidates], [relative(path) for path in sorted(helpers)]


def inventory() -> list[Variant]:
    known_kernels, known_directories = load_naming_exceptions()
    all_assembly = sorted(KAI_UKERNELS.rglob("*.S"))
    variants: list[Variant] = []

    for c_path in sorted(KAI_UKERNELS.rglob("*.c")):
        relative = c_path.relative_to(KAI_ROOT).as_posix()
        raw_name = c_path.stem
        kernel_issue = known_kernels.get(raw_name)
        canonical_name = kernel_issue.expected if kernel_issue else raw_name
        physical_family = c_path.parent.name
        directory_issue = known_directories.get(physical_family)
        canonical_family = directory_issue.expected if directory_issue else physical_family

        isa, feature_instruction, tile, layouts = extract_impl_metadata(canonical_name)
        files, helpers = collect_source_bundle(c_path, all_assembly)
        if physical_family == "pack":
            group_key, operation, operation_key, descriptors = parse_pack_group(canonical_name)
            family = " · ".join((operation, *(item.raw for item in descriptors)))
        else:
            operation, operation_key, descriptors = operation_and_descriptors(canonical_family)
            if operation_key == "matmul":
                operation, operation_key = classify_direct_matmul(canonical_family, tile)
            group_key = ("compute", canonical_family, operation_key)
            family = canonical_family

        variants.append(
            Variant(
                identity=relative,
                raw_name=raw_name,
                canonical_name=canonical_name,
                relative_c=relative,
                physical_family=physical_family,
                canonical_family=canonical_family,
                operation=operation,
                operation_key=operation_key,
                descriptors=descriptors,
                isa=isa,
                feature_instruction=feature_instruction,
                tile=tile,
                layouts=layouts,
                files=files,
                helper_files=helpers,
                group_key=group_key,
            )
        )

    if not variants:
        fail("No C-anchored public KleidiAI micro-kernels were found")
    if len({variant.identity for variant in variants}) != len(variants):
        fail("Duplicate micro-kernel identity discovered")
    return variants


def files_at_revision(revision: str) -> frozenset[str]:
    output = run_git(KAI_ROOT, "ls-tree", "-r", "--name-only", revision, "--", "kai/ukernels")
    return frozenset(output.splitlines())


STABLE_TAG_RE = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def stable_release_tags() -> list[str]:
    tags = []
    for tag in run_git(KAI_ROOT, "tag", "--list", "v*").splitlines():
        match = STABLE_TAG_RE.fullmatch(tag)
        if match:
            version = tuple(int(match.group(part)) for part in ("major", "minor", "patch"))
            tags.append((version, tag))
    return [tag for _version, tag in sorted(tags)]


def apply_release_history(variants: list[Variant]) -> None:
    """Attach exact-path provenance without treating pin absence as deprecation."""

    releases = stable_release_tags()
    release_files = {tag: files_at_revision(tag) for tag in releases}
    deprecation_pattern = re.compile(r"\b(?:deprecat(?:e|ed|ion)|obsolete)\b", re.IGNORECASE)

    for variant in variants:
        additions = run_git(
            KAI_ROOT,
            "log",
            "--diff-filter=A",
            "--format=%H%x09%cs",
            "--",
            variant.relative_c,
            check=False,
        ).splitlines()
        added_commit = ""
        added_date = ""
        if additions:
            # git log is newest first. If a path was removed and later restored,
            # the newest addition is the start of the current availability span.
            added_commit, _, added_date = additions[0].partition("\t")

        containing_releases = [tag for tag in releases if variant.relative_c in release_files[tag]]
        bundle_text = "\n".join(
            (KAI_ROOT / path).read_text(encoding="utf-8", errors="replace")
            for path in variant.files
            if (KAI_ROOT / path).is_file()
        )
        variant.release_history = ReleaseHistory(
            added_commit=added_commit,
            added_date=added_date,
            first_stable_release=containing_releases[0] if containing_releases else None,
            latest_stable_release=containing_releases[-1] if containing_releases else None,
            explicit_deprecation=bool(deprecation_pattern.search(bundle_text)),
        )


def discover_pins() -> tuple[Pin, Pin, Pin, Pin, Pin]:
    module_text = (XNN_ROOT / "MODULE.bazel").read_text(encoding="utf-8")
    commit_match = re.search(r'kleidiai-([0-9a-f]{40})', module_text)
    if not commit_match:
        fail("Could not identify XNNPACK's KleidiAI MODULE.bazel pin")
    xnn_commit = commit_match.group(1)
    xnn_label = run_git(KAI_ROOT, "describe", "--tags", "--exact-match", xnn_commit, check=False) or xnn_commit[:12]

    deps_text = (XNN_ROOT / "DEPS").read_text(encoding="utf-8")
    legacy_match = re.search(r"kleidiai@(?P<tag>v[0-9.]+)", deps_text)
    if not legacy_match:
        fail("Could not identify XNNPACK's DEPS KleidiAI pin")
    xnn_legacy = legacy_match.group("tag")

    mnn_cmake = (MNN_ROOT / "cmake" / "KleidiAI.cmake").read_text(encoding="utf-8")
    mnn_match = re.search(r'set\(KLEIDIAI_COMMIT_SHA\s+"?(?P<tag>[0-9.]+)"?\)', mnn_cmake)
    if not mnn_match:
        fail("Could not identify MNN's KleidiAI pin")
    mnn_tag = f"v{mnn_match.group('tag')}"

    llama_cmake = (LLAMA_ROOT / "ggml" / "src" / "ggml-cpu" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    llama_match = re.search(r'set\(KLEIDIAI_COMMIT_TAG\s+"?(?P<tag>v[0-9.]+)"?\)', llama_cmake)
    if not llama_match:
        fail("Could not identify llama.cpp's KleidiAI pin")
    llama_tag = llama_match.group("tag")

    return (
        Pin(xnn_label, xnn_commit, files_at_revision(xnn_commit)),
        Pin(xnn_legacy, xnn_legacy, files_at_revision(xnn_legacy)),
        discover_ort_pin(),
        Pin(mnn_tag, mnn_tag, files_at_revision(mnn_tag)),
        Pin(llama_tag, llama_tag, files_at_revision(llama_tag)),
    )


def discover_ort_pin() -> Pin:
    """Discover only ONNX Runtime's KleidiAI pin for the focused report."""

    deps_text = (ORT_ROOT / "cmake" / "deps.txt").read_text(encoding="utf-8")
    match = re.search(r"kleidiai;[^\n]*/tags/(?P<tag>v[0-9.]+)\.tar\.gz", deps_text)
    if not match:
        fail("Could not identify ONNX Runtime's KleidiAI pin")
    tag = match.group("tag")
    return Pin(tag, tag, files_at_revision(tag))


def source_files(roots: Iterable[Path]) -> list[tuple[Path, str, list[str]]]:
    result: list[tuple[Path, str, list[str]]] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "kai_" not in text:
                continue
            cleaned = strip_comments(text)
            result.append((path, cleaned, cleaned.splitlines()))
    return result


def scan_framework(
    variants: list[Variant],
    repo_root: Path,
    production_roots: list[Path],
    attr: str,
) -> None:
    sources = source_files(production_roots)
    for variant in variants:
        # The core name appears after prefixes such as kai_run_. Requiring a
        # non-alphanumeric predecessor prevents matmul matching inside imatmul.
        core = variant.raw_name.removeprefix("kai_")
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(core) + r"(?![A-Za-z0-9_])")
        active: list[Evidence] = []
        references: list[Evidence] = []
        for path, _text, lines in sources:
            for line_number, line in enumerate(lines, start=1):
                if not pattern.search(line):
                    continue
                kind = "include" if line.lstrip().startswith("#include") else "use"
                evidence = Evidence(
                    path=path.relative_to(repo_root).as_posix(),
                    line=line_number,
                    text=line.strip(),
                    kind=kind,
                )
                (references if kind == "include" else active).append(evidence)
        result: FrameworkResult = getattr(variant, attr)
        result.active = active
        result.references = references


def apply_pr_candidates(variants: list[Variant]) -> str:
    """Attach reviewed open PRs to exact, currently unintegrated bundles."""

    if not PR_CANDIDATES_PATH.is_file():
        fail(f"Open-PR audit file not found: {PR_CANDIDATES_PATH}")
    try:
        payload = json.loads(PR_CANDIDATES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Could not read open-PR audit: {error}")

    audited_at = payload.get("audited_at")
    if not isinstance(audited_at, str):
        fail("Open-PR audit must contain an audited_at date")
    try:
        dt.date.fromisoformat(audited_at)
    except ValueError:
        fail(f"Invalid open-PR audit date: {audited_at}")

    repositories = {"xnn": "google/XNNPACK", "ort": "microsoft/onnxruntime"}
    by_name = {variant.raw_name: variant for variant in variants}
    seen_prs: set[tuple[str, int]] = set()
    records = payload.get("pull_requests")
    if not isinstance(records, list):
        fail("Open-PR audit must contain a pull_requests list")

    for record in records:
        if not isinstance(record, dict):
            fail("Every open-PR record must be an object")
        framework = record.get("framework")
        repository = record.get("repository")
        number = record.get("number")
        title = record.get("title")
        url = record.get("url")
        draft = record.get("draft")
        names = record.get("kernel_names")
        if framework not in repositories or repository != repositories[framework]:
            fail(f"Invalid framework/repository in open-PR record: {framework}/{repository}")
        if not isinstance(number, int) or number <= 0:
            fail(f"Invalid pull-request number: {number}")
        if not isinstance(title, str) or not title.strip():
            fail(f"PR #{number} has no title")
        expected_url = f"https://github.com/{repository}/pull/{number}"
        if url != expected_url:
            fail(f"PR #{number} URL must be {expected_url}")
        if not isinstance(draft, bool):
            fail(f"PR #{number} draft must be true or false")
        if record.get("state") != "open" or record.get("merged") is not False:
            fail(f"PR #{number} is not recorded as open and unmerged")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            fail(f"PR #{number} must map to at least one exact kernel name")
        if len(names) != len(set(names)):
            fail(f"PR #{number} contains duplicate kernel names")
        key = (framework, number)
        if key in seen_prs:
            fail(f"Duplicate open-PR record: {framework} #{number}")
        seen_prs.add(key)

        candidate = PullRequest(framework, repository, number, title, url, draft)
        for name in names:
            variant = by_name.get(name)
            if variant is None:
                fail(f"PR #{number} references unknown current KleidiAI bundle: {name}")
            result: FrameworkResult = getattr(variant, framework)
            if result.active:
                fail(f"PR #{number} maps to already-integrated bundle: {name}")
            result.pr_candidates.append(candidate)

    return audited_at


def mark_xnn_runtime_selection(variants: list[Variant]) -> None:
    configs_root = XNN_ROOT / "src" / "configs"
    config_text = "\n".join(
        strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for path in configs_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    cache: dict[str, str] = {}
    for variant in variants:
        if not variant.xnn.active or not variant.is_compute or variant.operation_key == "dwconv":
            continue
        symbols: set[str] = set()
        for evidence in variant.xnn.active:
            source_path = XNN_ROOT / evidence.path
            if source_path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if evidence.path not in cache:
                cache[evidence.path] = strip_comments(source_path.read_text(encoding="utf-8", errors="replace"))
            symbols.update(re.findall(r"\bxnn_[A-Za-z0-9_]+", cache[evidence.path]))
        variant.xnn.runtime_selected = any(
            re.search(r"\b" + re.escape(symbol) + r"\b", config_text) for symbol in symbols
        )


def apply_availability(
    variants: list[Variant], xnn: Pin, xnn_legacy: Pin, ort: Pin, mnn: Pin, llama: Pin
) -> None:
    for variant in variants:
        variant.xnn.available_primary = variant.relative_c in xnn.files
        variant.xnn.available_secondary = variant.relative_c in xnn_legacy.files
        variant.ort.available_primary = variant.relative_c in ort.files
        variant.mnn.available_primary = variant.relative_c in mnn.files
        variant.llama.available_primary = variant.relative_c in llama.files


def group_variants(variants: list[Variant]) -> list[Group]:
    groups_by_key: dict[tuple[str, ...], Group] = {}
    for variant in variants:
        group = groups_by_key.get(variant.group_key)
        if group is None:
            family = (
                variant.canonical_family
                if variant.group_key[0] == "compute"
                else " · ".join((variant.operation, *(item.compact for item in variant.descriptors)))
            )
            group = Group(
                key=variant.group_key,
                operation=variant.operation,
                operation_key=variant.operation_key,
                family=family,
                descriptors=variant.descriptors,
            )
            groups_by_key[variant.group_key] = group
        group.variants.append(variant)

    operation_order = {
        "gemm": 0,
        "gemv": 1,
        "imatmul": 2,
        "dwconv": 3,
        "lhs-pack": 4,
        "rhs-pack-kxn": 5,
        "rhs-pack-nxk": 6,
        "dwconv-rhs-pack": 7,
        "pack": 8,
    }
    groups = list(groups_by_key.values())
    for group in groups:
        group.variants.sort(key=lambda item: (item.isa, item.tile, item.raw_name))
    groups.sort(key=lambda item: (operation_order.get(item.operation_key, 99), item.family))
    return groups


def quantization_class(group: Group) -> tuple[str, str]:
    descriptors = group.descriptors
    quantized = [item for item in descriptors if item.quantization]
    if "pack" in group.operation_key:
        if not quantized:
            return "Unquantized packing", "unquantized"
        return "Quantization / packed transform", "packing-quantization"
    if not quantized:
        if any(item.base_type.startswith(("i", "u", "x")) for item in descriptors):
            return "Integer, quantization metadata not encoded", "integer"
        return "Unquantized", "unquantized"

    output = descriptors[0] if descriptors else None
    lhs = descriptors[1] if len(descriptors) > 1 else None
    rhs = descriptors[2] if len(descriptors) > 2 else None
    output_is_float = bool(output and output.base_type in {"f16", "f32", "fp32", "bf16"})
    dynamic_lhs = bool(lhs and lhs.quantization and lhs.axis in {"dx", "d32"} and output_is_float)
    weight_quantized = bool(rhs and rhs.quantization)
    if dynamic_lhs:
        label = "Dynamic activation quantization (inferred)"
        if weight_quantized:
            label += " + quantized weights"
        return label, "dynamic"
    if output and output.quantization:
        return "Fully quantized output path", "fully-quantized"
    if output_is_float and weight_quantized:
        return "Weight-only quantization", "weight-only"
    return "Quantized", "quantized"


def quantization_detail(descriptors: list[Descriptor]) -> str:
    roles = ["Output", "LHS", "RHS"]
    parts = []
    for index, descriptor in enumerate(descriptors):
        if not descriptor.quantization:
            continue
        role = roles[index] if index < len(roles) else f"Aux {index - 2}"
        mode = "asymmetric" if descriptor.quantization == "qa" else "symmetric"
        axis = AXIS_LABELS.get(descriptor.axis or "", descriptor.axis or "per-tensor")
        parts.append(f"{role}: {mode}, {axis}")
    return "; ".join(parts) or "No quantization marker"


def packed_format_summary(variants: list[Variant]) -> str:
    """Keep concrete packed-format tokens without repeating operand state."""

    layouts = sorted({layout for variant in variants for layout in variant.layouts})
    if not layouts:
        return "—"
    shown = ", ".join(layouts[:3])
    if len(layouts) > 3:
        shown += f" +{len(layouts) - 3}"
    return shown


def split_operand_descriptors(group: Group) -> tuple[list[Descriptor], list[Descriptor]]:
    """Place packer operands in the same semantic column as their operation."""

    operands = group.descriptors[1:]
    if group.operation_key == "lhs-pack":
        return operands, []
    if group.operation_key in {"rhs-pack-kxn", "rhs-pack-nxk", "dwconv-rhs-pack"}:
        return [], operands
    # Compute families encode output, LHS/input, then RHS/weights. Keep any
    # additional RHS-side metadata alongside the weights descriptor.
    return operands[:1], operands[1:]


def descriptor_column(descriptors: list[Descriptor]) -> str:
    return "<br>".join(h(item.compact) for item in descriptors) if descriptors else "—"


def group_framework_status(group: Group, attr: str) -> tuple[str, str, int, int]:
    results = [getattr(variant, attr) for variant in group.variants]
    active = sum(bool(result.active) for result in results)
    referenced = sum(bool(result.references) and not result.active for result in results)
    total = len(results)
    available = sum(result.available_primary for result in results)
    if active == total:
        return "Integrated", "integrated", active, total
    if active:
        return "Partial", "partial", active, total
    if referenced:
        return "Referenced only", "referenced", active, total
    if not available:
        return "Unavailable at pin", "unavailable", active, total
    return "Not integrated", "not-integrated", active, total


def variant_framework_status(variant: Variant, attr: str) -> tuple[str, str]:
    result: FrameworkResult = getattr(variant, attr)
    if result.active:
        if attr == "xnn" and variant.is_compute:
            if result.runtime_selected:
                return "Direct · runtime", "integrated"
            return "Direct · wrapper/test", "available"
        if "pack" in variant.operation_key:
            return "Direct · packer", "integrated"
        return "Direct · runtime", "integrated"
    if result.references:
        return "Referenced only", "referenced"
    if not result.available_primary:
        return "Unavailable at pin", "unavailable"
    return "Not integrated", "not-integrated"


def evidence_html(repository_url: str, revision: str, result: FrameworkResult) -> str:
    evidence = result.active or result.references
    if not evidence:
        return ""
    items = []
    for item in evidence[:4]:
        href = f"{repository_url}/blob/{revision}/{item.path}#L{item.line}"
        excerpt = item.text
        if len(excerpt) > 150:
            excerpt = excerpt[:147] + "…"
        items.append(
            f'<li><a href="{h(href)}" target="_blank" rel="noopener noreferrer">'
            f'{h(item.path)}:{item.line}</a>'
            f'<code>{h(excerpt)}</code></li>'
        )
    if len(evidence) > 4:
        items.append(f"<li>+ {len(evidence) - 4} more exact occurrences</li>")
    return f'<ul class="evidence-list">{"".join(items)}</ul>'


def status_badge(label: str, key: str, suffix: str = "") -> str:
    suffix_html = f'<span class="status-count">{h(suffix)}</span>' if suffix else ""
    return f'<span class="status status-{h(key)}">{h(label)}{suffix_html}</span>'


def unique_prs(pull_requests: Iterable[PullRequest]) -> list[PullRequest]:
    deduplicated = {
        (pull_request.framework, pull_request.number): pull_request
        for pull_request in pull_requests
    }
    return sorted(deduplicated.values(), key=lambda pull_request: pull_request.number, reverse=True)


def group_pr_candidates(group: Group, attr: str) -> list[PullRequest]:
    return unique_prs(
        pull_request
        for variant in group.variants
        for pull_request in getattr(variant, attr).pr_candidates
    )


def pr_links(pull_requests: Iterable[PullRequest]) -> str:
    links = []
    for pull_request in unique_prs(pull_requests):
        draft = '<span class="pr-draft">Draft</span>' if pull_request.draft else ""
        accessible_title = (
            f'PR available: #{pull_request.number} — {pull_request.title}'
            + (" (draft)" if pull_request.draft else "")
        )
        links.append(
            f'<a class="status status-pr-available" href="{h(pull_request.url)}" '
            f'target="_blank" rel="noopener noreferrer" title="{h(accessible_title)}" '
            f'aria-label="{h(accessible_title)}">PR available · #{pull_request.number}{draft}</a>'
        )
    return f'<div class="pr-list">{"".join(links)}</div>' if links else ""


def kai_public_url(path: str, revision: str) -> str:
    return f"{KAI_GITHUB}/blob/{revision}/{path}"


def source_links(paths: list[str], helpers: list[str], kai_revision: str) -> str:
    links = [
        f'<a href="{h(kai_public_url(path, kai_revision))}" target="_blank" rel="noopener noreferrer">'
        f'{h(Path(path).name)}</a>'
        for path in paths
    ]
    if helpers:
        links.extend(
            f'<a class="helper" href="{h(kai_public_url(path, kai_revision))}" '
            f'target="_blank" rel="noopener noreferrer">{h(Path(path).name)} <small>helper</small></a>'
            for path in helpers
        )
    return '<div class="source-links">' + "".join(links) + "</div>"


def release_history_html(
    variant: Variant, xnn_pin: Pin, ort_pin: Pin, mnn_pin: Pin, llama_pin: Pin
) -> str:
    history = variant.release_history
    if history is None:
        return '<span class="muted">Release history unavailable</span>'

    if history.added_commit:
        commit_link = (
            f'<a href="{KAI_GITHUB}/commit/{h(history.added_commit)}" target="_blank" '
            f'rel="noopener noreferrer"><code>{h(history.added_commit[:12])}</code></a>'
        )
        introduced = f"{commit_link} · {h(history.added_date)}"
    else:
        introduced = "Not resolved from repository history"

    if history.first_stable_release:
        first = (
            f'<a href="{KAI_GITHUB}/tree/{h(history.first_stable_release)}" target="_blank" '
            f'rel="noopener noreferrer">{h(history.first_stable_release)}</a>'
        )
    else:
        first = "Not yet included in a stable tag"

    if history.latest_stable_release:
        latest = (
            f'<a href="{KAI_GITHUB}/tree/{h(history.latest_stable_release)}" target="_blank" '
            f'rel="noopener noreferrer">{h(history.latest_stable_release)}</a>'
        )
    else:
        latest = "—"

    pin_gaps = []
    if not variant.xnn.available_primary:
        pin_gaps.append(f"XNNPACK {xnn_pin.label}")
    if not variant.ort.available_primary:
        pin_gaps.append(f"ONNX Runtime {ort_pin.label}")
    if not variant.mnn.available_primary:
        pin_gaps.append(f"MNN {mnn_pin.label}")
    if not variant.llama.available_primary:
        pin_gaps.append(f"llama.cpp {llama_pin.label}")
    gap = (
        '<div class="pin-gap"><strong>Absent from older pin:</strong> '
        f'{h(", ".join(pin_gaps))}. This is an availability gap, not a deprecation.</div>'
        if pin_gaps
        else ""
    )
    deprecation = (
        '<span class="provenance-warning">Deprecation-like text found; manual review required.</span>'
        if history.explicit_deprecation
        else '<span class="provenance-ok">Present at audited HEAD; no explicit deprecation marker found.</span>'
    )
    return (
        '<div class="release-history">'
        f'<div><strong>Introduced:</strong> {introduced}</div>'
        f'<div><strong>First stable:</strong> {first}</div>'
        f'<div><strong>Latest stable containing path:</strong> {latest}</div>'
        f'{gap}<div>{deprecation}</div></div>'
    )


def render_variant_rows(
    group: Group,
    xnn_pin: Pin,
    xnn_legacy_pin: Pin,
    ort_pin: Pin,
    mnn_pin: Pin,
    llama_pin: Pin,
    kai_revision: str,
    xnn_revision: str,
    ort_revision: str,
    mnn_revision: str,
    llama_revision: str,
) -> str:
    rows = []
    for variant in group.variants:
        xnn_label, xnn_key = variant_framework_status(variant, "xnn")
        ort_label, ort_key = variant_framework_status(variant, "ort")
        mnn_label, mnn_key = variant_framework_status(variant, "mnn")
        llama_label, llama_key = variant_framework_status(variant, "llama")
        xnn_availability = (
            f'<div class="availability">At {h(xnn_pin.label)}: '
            f'{"yes" if variant.xnn.available_primary else "no"}; '
            f'at DEPS {h(xnn_legacy_pin.label)}: '
            f'{"yes" if variant.xnn.available_secondary else "no"}</div>'
        )
        ort_availability = (
            f'<div class="availability">At {h(ort_pin.label)}: '
            f'{"yes" if variant.ort.available_primary else "no"}</div>'
        )
        mnn_availability = (
            f'<div class="availability">At {h(mnn_pin.label)}: '
            f'{"yes" if variant.mnn.available_primary else "no"}</div>'
        )
        llama_availability = (
            f'<div class="availability">At {h(llama_pin.label)}: '
            f'{"yes" if variant.llama.available_primary else "no"}</div>'
        )
        rows.append(
            f'<tr data-isa="{h(variant.isa)}">'
            f'<td><code class="kernel-name">{h(variant.raw_name)}</code></td>'
            f'<td><strong>{h(variant.isa)}</strong><span>{h(variant.feature_instruction)}</span></td>'
            f'<td><code>{h(variant.tile)}</code></td>'
            f'<td>{source_links(variant.files, variant.helper_files, kai_revision)}</td>'
            f'<td>{status_badge(xnn_label, xnn_key)}{pr_links(variant.xnn.pr_candidates)}'
            f'{xnn_availability}{evidence_html(XNN_GITHUB, xnn_revision, variant.xnn)}</td>'
            f'<td>{status_badge(ort_label, ort_key)}{pr_links(variant.ort.pr_candidates)}'
            f'{ort_availability}{evidence_html(ORT_GITHUB, ort_revision, variant.ort)}</td>'
            f'<td>{status_badge(mnn_label, mnn_key)}'
            f'{mnn_availability}{evidence_html(MNN_GITHUB, mnn_revision, variant.mnn)}</td>'
            f'<td>{status_badge(llama_label, llama_key)}'
            f'{llama_availability}{evidence_html(LLAMA_GITHUB, llama_revision, variant.llama)}</td>'
            "</tr>"
        )
    return "".join(rows)


def render_group_rows(
    groups: list[Group],
    xnn_pin: Pin,
    xnn_legacy_pin: Pin,
    ort_pin: Pin,
    mnn_pin: Pin,
    llama_pin: Pin,
    kai_revision: str,
    xnn_revision: str,
    ort_revision: str,
    mnn_revision: str,
    llama_revision: str,
) -> str:
    result = []
    for index, group in enumerate(groups):
        row_id = f"group-{index}-{slug(group.family)[:42]}"
        quant_label, quant_key = quantization_class(group)
        quant_detail = quantization_detail(group.descriptors)
        output = group.descriptors[0].compact if group.descriptors else "—"
        lhs_descriptors, rhs_descriptors = split_operand_descriptors(group)
        lhs_html = descriptor_column(lhs_descriptors)
        rhs_html = descriptor_column(rhs_descriptors)
        packed_format = packed_format_summary(group.variants)
        isas = sorted({variant.isa for variant in group.variants})
        xnn_label, xnn_key, xnn_active, xnn_total = group_framework_status(group, "xnn")
        ort_label, ort_key, ort_active, ort_total = group_framework_status(group, "ort")
        mnn_label, mnn_key, mnn_active, mnn_total = group_framework_status(group, "mnn")
        llama_label, llama_key, llama_active, llama_total = group_framework_status(group, "llama")
        xnn_prs = group_pr_candidates(group, "xnn")
        ort_prs = group_pr_candidates(group, "ort")
        xnn_filter_tokens = [xnn_key, *(["pr-available"] if xnn_prs else [])]
        ort_filter_tokens = [ort_key, *(["pr-available"] if ort_prs else [])]
        search_text = " ".join(
            [
                group.family,
                group.operation,
                output,
                *(item.compact for item in lhs_descriptors),
                *(item.compact for item in rhs_descriptors),
                quant_label,
                quant_detail,
                packed_format,
                *isas,
                *(variant.raw_name for variant in group.variants),
                *(f"#{pull_request.number} {pull_request.title}" for pull_request in xnn_prs),
                *(f"#{pull_request.number} {pull_request.title}" for pull_request in ort_prs),
            ]
        ).lower()
        output_types = sorted({group.descriptors[0].type_label}) if group.descriptors else []
        lhs_types = sorted({item.type_label for item in lhs_descriptors})
        rhs_types = sorted({item.type_label for item in rhs_descriptors})
        family_code = f'<code>{h(group.family)}</code>' if group.key[0] == "compute" else h(group.family)
        result.append(
            f'<tr class="data-row" data-detail="{h(row_id)}" '
            f'data-search="{h(search_text)}" data-operation="{h(group.operation_key)}" '
            f'data-output-types="{h("|".join(output_types))}" '
            f'data-lhs-types="{h("|".join(lhs_types))}" data-rhs-types="{h("|".join(rhs_types))}" '
            f'data-quant="{h(quant_key)}" '
            f'data-isa="{h("|".join(isas))}" '
            f'data-xnn="{h("|".join(xnn_filter_tokens))}" '
            f'data-ort="{h("|".join(ort_filter_tokens))}" '
            f'data-mnn="{h(mnn_key)}" data-llama="{h(llama_key)}" '
            f'data-sort-operation="{h(group.operation.lower())}" '
            f'data-sort-variants="{len(group.variants)}" data-sort-xnn="{xnn_active}" '
            f'data-sort-ort="{ort_active}" data-sort-mnn="{mnn_active}" data-sort-llama="{llama_active}">'
            f'<td class="operation-cell"><button class="expand" type="button" aria-expanded="false" '
            f'aria-controls="{h(row_id)}"><span class="chevron">›</span><span>{h(group.operation)}</span></button>'
            f'<div class="row-meta"><span>{len(group.variants)} variant{"s" if len(group.variants) != 1 else ""}</span></div></td>'
            f'<td>{h(output)}</td>'
            f'<td>{lhs_html}</td>'
            f'<td>{rhs_html}</td>'
            f'<td><code>{h(packed_format)}</code></td>'
            f'<td><div class="chip-list">{"".join(f"<span class=\"mini-chip\">{h(isa)}</span>" for isa in isas)}</div></td>'
            f'<td>{status_badge(xnn_label, xnn_key, f"{xnn_active}/{xnn_total}")}{pr_links(xnn_prs)}</td>'
            f'<td>{status_badge(ort_label, ort_key, f"{ort_active}/{ort_total}")}{pr_links(ort_prs)}</td>'
            f'<td>{status_badge(mnn_label, mnn_key, f"{mnn_active}/{mnn_total}")}</td>'
            f'<td>{status_badge(llama_label, llama_key, f"{llama_active}/{llama_total}")}</td>'
            "</tr>"
        )
        result.append(
            f'<tr class="detail-row" id="{h(row_id)}" hidden><td colspan="10">'
            '<div class="detail-panel"><div class="detail-heading">'
            f'<div><h3>{family_code}</h3><p><strong>{h(group.operation)}</strong> · '
            'Exact public implementation bundles and framework evidence.</p></div>'
            f'<span class="variant-total">{len(group.variants)} variant{"s" if len(group.variants) != 1 else ""}</span></div>'
            '<div class="variant-scroll"><table class="variant-table"><thead><tr>'
            '<th>Micro-kernel</th><th>ISA / instruction</th><th>Tile</th><th>Bundle files</th>'
            '<th>XNNPACK evidence</th><th>ONNX Runtime evidence</th>'
            '<th>MNN evidence</th><th>llama.cpp evidence</th>'
            '</tr></thead><tbody>'
            f'{render_variant_rows(group, xnn_pin, xnn_legacy_pin, ort_pin, mnn_pin, llama_pin, kai_revision, xnn_revision, ort_revision, mnn_revision, llama_revision)}'
            '</tbody></table></div></div></td></tr>'
        )
    return "".join(result)


def checkbox_filter(
    filter_id: str,
    label: str,
    values: Iterable[tuple[str, str]],
) -> str:
    options = "".join(
        '<label class="check-option">'
        f'<input type="checkbox" value="{h(value)}" data-label="{h(option_label)}">'
        f'<span>{h(option_label)}</span></label>'
        for value, option_label in values
    )
    return (
        '<div class="control multi-control">'
        f'<span class="control-label" id="{h(filter_id)}Label">{h(label)}</span>'
        f'<details class="multi-filter" id="{h(filter_id)}" '
        f'aria-labelledby="{h(filter_id)}Label">'
        f'<summary aria-labelledby="{h(filter_id)}Label {h(filter_id)}Summary">'
        f'<span class="filter-summary" id="{h(filter_id)}Summary">All</span></summary>'
        f'<div class="checklist" role="group" aria-label="{h(label)}">{options}</div>'
        '</details></div>'
    )


def render_html(
    variants: list[Variant],
    groups: list[Group],
    xnn_pin: Pin,
    xnn_legacy_pin: Pin,
    ort_pin: Pin,
    mnn_pin: Pin,
    llama_pin: Pin,
    pr_audited_at: str,
) -> str:
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    kai_revision_full = run_git(KAI_ROOT, "rev-parse", "HEAD")
    kai_describe = describe_revision(KAI_ROOT)
    xnn_revision_full = run_git(XNN_ROOT, "rev-parse", "HEAD")
    ort_revision_full = run_git(ORT_ROOT, "rev-parse", "HEAD")
    mnn_revision_full = run_git(MNN_ROOT, "rev-parse", "HEAD")
    llama_revision_full = run_git(LLAMA_ROOT, "rev-parse", "HEAD")
    xnn_revision = short_revision(XNN_ROOT)
    ort_revision = short_revision(ORT_ROOT)
    mnn_revision = short_revision(MNN_ROOT)
    llama_revision = short_revision(LLAMA_ROOT)
    xnn_active = sum(bool(variant.xnn.active) for variant in variants)
    ort_active = sum(bool(variant.ort.active) for variant in variants)
    mnn_active = sum(bool(variant.mnn.active) for variant in variants)
    llama_active = sum(bool(variant.llama.active) for variant in variants)
    xnn_prs = unique_prs(
        pull_request for variant in variants for pull_request in variant.xnn.pr_candidates
    )
    ort_prs = unique_prs(
        pull_request for variant in variants for pull_request in variant.ort.pr_candidates
    )
    output_type_values = sorted(
        {group.descriptors[0].type_label for group in groups if group.descriptors}
    )
    lhs_type_values = sorted(
        {
            descriptor.type_label
            for group in groups
            for descriptor in split_operand_descriptors(group)[0]
        }
    )
    rhs_type_values = sorted(
        {
            descriptor.type_label
            for group in groups
            for descriptor in split_operand_descriptors(group)[1]
        }
    )
    isa_values = sorted({variant.isa for variant in variants})
    quant_labels_by_key: dict[str, str] = {}
    for group in groups:
        quant_label, quant_key = quantization_class(group)
        quant_labels_by_key.setdefault(quant_key, quant_label)
    if "unquantized" in quant_labels_by_key:
        quant_labels_by_key["unquantized"] = "Unquantized (compute or packing)"
    quant_options = sorted(quant_labels_by_key.items(), key=lambda item: item[1])
    operation_filter_labels = {
        "gemm": "GEMM",
        "gemv": "GEMV",
        "imatmul": "Indirect GEMM",
        "dwconv": "Depthwise convolution",
    }
    operations_by_key = {group.operation_key: group.operation for group in groups}
    operation_options = sorted(
        [
            (key, operation_filter_labels.get(key, label))
            for key, label in operations_by_key.items()
        ],
        key=lambda item: item[1],
    )
    integration_options = (
        ("integrated", "Integrated"),
        ("partial", "Partial"),
        ("referenced", "Referenced only"),
        ("not-integrated", "Not integrated"),
        ("unavailable", "Unavailable at pin"),
    )
    xnn_integration_options = integration_options + (
        (("pr-available", "PR available"),) if xnn_prs else ()
    )
    ort_integration_options = integration_options + (
        (("pr-available", "PR available"),) if ort_prs else ()
    )

    data_snapshot = {
        "generated": now.isoformat(timespec="seconds"),
        "kleidiai": {"revision": kai_revision_full, "describe": kai_describe, "public_microkernels": len(variants)},
        "xnnpack": {
            "revision": xnn_revision_full,
            "short_revision": xnn_revision,
            "pin": xnn_pin.label,
            "direct": xnn_active,
        },
        "onnxruntime": {
            "revision": ort_revision_full,
            "short_revision": ort_revision,
            "pin": ort_pin.label,
            "direct": ort_active,
        },
        "mnn": {
            "revision": mnn_revision_full,
            "short_revision": mnn_revision,
            "pin": mnn_pin.label,
            "direct": mnn_active,
        },
        "llama_cpp": {
            "revision": llama_revision_full,
            "short_revision": llama_revision,
            "pin": llama_pin.label,
            "direct": llama_active,
        },
        "open_pull_requests": {
            "audited_at": pr_audited_at,
            "xnnpack": len(xnn_prs),
            "onnxruntime": len(ort_prs),
        },
    }

    return f'''<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="KleidiAI micro-kernel inventory and exact XNNPACK, ONNX Runtime, MNN, and llama.cpp integration status">
  <title>KleidiAI integration status</title>
  <style>
    :root {{ color-scheme: dark; --bg:#09111f; --surface:#101b2d; --surface-2:#15243a; --surface-3:#1a2c46; --text:#e7eef9; --muted:#9fb0c8; --line:#29405f; --accent:#58a6ff; --accent-2:#8b5cf6; --green:#35d399; --amber:#f7b955; --red:#fb7185; --blue:#67b7ff; --shadow:0 18px 45px rgba(0,0,0,.28); }}
    html[data-theme="light"] {{ color-scheme:light; --bg:#f2f6fb; --surface:#fff; --surface-2:#f8fbff; --surface-3:#eef4fb; --text:#142033; --muted:#5b6b82; --line:#d4deeb; --accent:#1769c2; --accent-2:#6d40c7; --shadow:0 16px 38px rgba(27,48,76,.12); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 10% 0%, rgba(47,116,190,.19), transparent 28rem), var(--bg); color:var(--text); font:14px/1.48 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    button,input {{ font:inherit; }} code {{ font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace; }}
    .shell {{ width:min(1880px, calc(100% - 32px)); margin:0 auto; padding:28px 0 64px; }}
    .hero {{ position:relative; overflow:hidden; padding:32px; border:1px solid var(--line); border-radius:22px; background:linear-gradient(125deg, rgba(41,103,168,.27), rgba(88,43,131,.2)),var(--surface); box-shadow:var(--shadow); }}
    .hero:after {{ content:""; position:absolute; width:300px; height:300px; border-radius:50%; right:-110px; top:-165px; background:radial-gradient(circle,rgba(88,166,255,.31),transparent 68%); }}
    h1 {{ max-width:900px; margin:8px 0 8px; font-size:clamp(30px,4vw,48px); line-height:1.08; letter-spacing:-.035em; }}
    .lede {{ max-width:900px; margin:0; color:var(--muted); font-size:16px; }}
    .hero-actions {{ position:absolute; z-index:2; right:24px; top:24px; display:flex; gap:8px; }}
    .icon-button {{ border:1px solid var(--line); border-radius:10px; color:var(--text); background:var(--surface-2); padding:8px 11px; cursor:pointer; }}
    .revision-strip {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:22px; color:var(--muted); font-size:12px; }}
    .revision-strip code {{ color:var(--text); }}
    .controls {{ position:sticky; z-index:10; top:0; margin-top:18px; padding:14px; border:1px solid var(--line); border-radius:15px; background:color-mix(in srgb,var(--surface) 92%,transparent); backdrop-filter:blur(16px); box-shadow:0 12px 30px rgba(0,0,0,.16); }}
    .control-grid {{ display:grid; grid-template-columns:minmax(240px,2fr) repeat(10,minmax(125px,1fr)) auto; gap:9px; }}
    .control {{ display:flex; min-width:0; flex-direction:column; gap:4px; }}
    .control-label {{ color:var(--muted); font-size:10px; text-transform:uppercase; font-weight:800; letter-spacing:.07em; }}
    input[type="search"] {{ width:100%; min-width:0; border:1px solid var(--line); border-radius:9px; background:var(--surface-2); color:var(--text); padding:9px 10px; outline:none; }}
    input[type="search"]:focus {{ border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent); }}
    .multi-control {{ position:relative; }}
    .multi-filter {{ position:relative; min-width:0; }}
    .multi-filter summary {{ display:flex; align-items:center; justify-content:space-between; gap:7px; min-width:0; height:39px; padding:9px 10px; border:1px solid var(--line); border-radius:9px; background:var(--surface-2); color:var(--text); cursor:pointer; list-style:none; outline:none; }}
    .multi-filter summary::-webkit-details-marker {{ display:none; }}
    .multi-filter summary:focus-visible {{ border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent); }}
    .multi-filter summary::after {{ content:"▾"; flex:0 0 auto; color:var(--muted); font-size:11px; transition:transform .15s ease; }}
    .multi-filter[open] summary {{ border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent); }}
    .multi-filter[open] summary::after {{ transform:rotate(180deg); }}
    .filter-summary {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .checklist {{ position:absolute; z-index:60; top:calc(100% + 6px); left:0; width:max-content; min-width:100%; max-width:min(320px,calc(100vw - 24px)); max-height:310px; overflow:auto; padding:7px; border:1px solid var(--line); border-radius:10px; background:var(--surface-2); box-shadow:var(--shadow); }}
    .multi-filter.viewport-right .checklist {{ right:0; left:auto; }}
    .check-option {{ display:flex; align-items:center; gap:8px; padding:7px 8px; border-radius:7px; color:var(--text); cursor:pointer; font-size:12px; font-weight:600; line-height:1.25; letter-spacing:normal; text-transform:none; }}
    .check-option:hover {{ background:var(--surface-3); }}
    .check-option input {{ flex:0 0 15px; width:15px; height:15px; margin:0; accent-color:var(--accent); }}
    .check-option:has(input:focus-visible) {{ outline:2px solid var(--accent); outline-offset:-2px; }}
    .reset {{ align-self:end; border:1px solid var(--line); border-radius:9px; background:var(--surface-3); color:var(--text); padding:9px 13px; cursor:pointer; }}
    .result-line {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:10px; color:var(--muted); font-size:12px; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .table-wrap {{ margin-top:12px; overflow:auto; border:1px solid var(--line); border-radius:15px; background:var(--surface); box-shadow:var(--shadow); }}
    table {{ width:100%; border-collapse:separate; border-spacing:0; }}
    .main-table {{ min-width:1580px; }}
    th {{ position:sticky; top:0; z-index:3; padding:0; border-bottom:1px solid var(--line); background:var(--surface-3); color:var(--muted); text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.055em; }}
    th button {{ width:100%; border:0; background:transparent; color:inherit; padding:13px 12px; text-align:left; text-transform:inherit; letter-spacing:inherit; font-weight:800; cursor:pointer; }}
    th button:hover {{ color:var(--text); }}
    td {{ vertical-align:top; padding:13px 12px; border-bottom:1px solid var(--line); }}
    .data-row:hover td {{ background:color-mix(in srgb,var(--surface-3) 58%,transparent); }}
    .data-row:last-child td {{ border-bottom:0; }}
    .operation-cell {{ min-width:170px; }}
    .expand {{ display:flex; align-items:flex-start; gap:8px; width:100%; padding:0; border:0; background:transparent; color:var(--text); text-align:left; cursor:pointer; font-weight:750; }}
    .expand code {{ white-space:normal; overflow-wrap:anywhere; }}
    .chevron {{ display:inline-grid; flex:0 0 20px; width:20px; height:20px; place-items:center; border:1px solid var(--line); border-radius:6px; transition:transform .18s ease; color:var(--accent); }}
    .expand[aria-expanded="true"] .chevron {{ transform:rotate(90deg); }}
    .row-meta {{ display:flex; align-items:center; gap:7px; margin:7px 0 0 28px; color:var(--muted); font-size:11px; }}
    td small {{ display:block; margin-top:5px; color:var(--muted); }}
    .chip-list {{ display:flex; flex-wrap:wrap; gap:4px; }}
    .mini-chip {{ display:inline-flex; align-items:center; padding:2px 6px; border:1px solid var(--line); border-radius:999px; background:var(--surface-2); color:var(--muted); font-size:10px; font-weight:750; white-space:nowrap; }}
    .status {{ display:inline-flex; flex-wrap:wrap; align-items:center; gap:5px; max-width:155px; padding:4px 8px; border:1px solid transparent; border-radius:8px; font-size:10px; line-height:1.25; font-weight:850; text-transform:uppercase; letter-spacing:.025em; }}
    .status-count {{ padding-left:5px; border-left:1px solid currentColor; opacity:.8; }}
    .status-integrated {{ color:#6ce8bb; background:rgba(35,166,122,.15); border-color:rgba(61,211,158,.35); }}
    .status-partial,.status-available {{ color:#ffd17d; background:rgba(190,125,25,.15); border-color:rgba(247,185,85,.37); }}
    .status-referenced {{ color:#8ecbff; background:rgba(58,132,197,.16); border-color:rgba(103,183,255,.34); }}
    .status-not-integrated {{ color:var(--muted); background:rgba(125,143,166,.10); border-color:var(--line); }}
    .status-unavailable {{ color:#f5a3b2; background:rgba(187,55,81,.14); border-color:rgba(251,113,133,.3); }}
    .status-pr-available {{ color:#c4b5fd; background:rgba(124,58,237,.15); border-color:rgba(167,139,250,.4); text-decoration:none; }}
    a.status-pr-available:hover {{ color:#ddd6fe; background:rgba(124,58,237,.24); text-decoration:none; }}
    .pr-list {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }}
    .pr-list .status {{ max-width:none; }} .pr-draft {{ padding-left:5px; border-left:1px solid currentColor; opacity:.8; }}
    .detail-row td {{ padding:0; background:var(--surface-2); }}
    .detail-panel {{ padding:20px; border-bottom:1px solid var(--line); box-shadow:inset 4px 0 0 var(--accent); }}
    .detail-heading {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:12px; }}
    .detail-heading h3 {{ margin:0; font-size:16px; }} .detail-heading p {{ margin:3px 0 0; color:var(--muted); }}
    .variant-total {{ color:var(--muted); white-space:nowrap; }} .variant-scroll {{ overflow:auto; border:1px solid var(--line); border-radius:10px; }}
    .variant-table {{ min-width:1780px; background:var(--surface); }} .variant-table th {{ position:static; padding:10px; }} .variant-table td {{ padding:11px 10px; font-size:12px; }}
    .variant-table td:nth-child(1) {{ width:22%; }} .variant-table td:nth-child(2) {{ width:9%; }} .variant-table td:nth-child(3) {{ width:5%; }} .variant-table td:nth-child(4) {{ width:16%; }} .variant-table td:nth-child(n+5) {{ width:12%; }}
    .kernel-name {{ overflow-wrap:anywhere; }}
    .variant-table td:nth-child(2) span {{ display:block; color:var(--muted); }}
    .source-links {{ display:flex; flex-direction:column; gap:4px; }} .source-links a {{ overflow-wrap:anywhere; }} .source-links .helper {{ color:#a99af2; }}
    .availability {{ margin-top:6px; color:var(--muted); font-size:10px; }}
    .evidence-list {{ margin:7px 0 0; padding-left:16px; color:var(--muted); }} .evidence-list li {{ margin:4px 0; }}
    .evidence-list code {{ display:block; margin-top:2px; color:var(--muted); font-size:9px; overflow-wrap:anywhere; }}
    .empty {{ padding:36px; text-align:center; color:var(--muted); }}
    footer {{ margin-top:18px; color:var(--muted); text-align:center; font-size:11px; }}
    @media (max-width:1250px) {{ .control-grid {{ grid-template-columns:repeat(4,1fr); }} .control.search {{ grid-column:span 2; }} }}
    @media (max-width:760px) {{ .shell {{ width:min(100% - 18px,1880px); padding-top:10px; }} .hero {{ padding:24px 18px; }} .hero-actions {{ position:static; margin-bottom:16px; }} .control-grid {{ grid-template-columns:repeat(2,1fr); }} .control.search {{ grid-column:span 2; }} .reset {{ width:100%; }} }}
    @media print {{ .hero-actions,.controls,.expand .chevron {{ display:none; }} .shell {{ width:100%; }} .table-wrap {{ overflow:visible; box-shadow:none; }} .main-table {{ min-width:0; font-size:9px; }} .detail-row {{ display:none !important; }} th {{ position:static; }} }}
  </style>
</head>
<body>
<main class="shell">
  <header class="hero">
    <div class="hero-actions"><button id="themeToggle" class="icon-button" type="button" aria-label="Toggle colour theme">☀︎ / ◐</button></div>
    <h1>KleidiAI integration status</h1>
    <p class="lede">A grouped inventory of every public micro-kernel at an exact public KleidiAI revision, with datatype, quantization, packing, ISA, and exact production-use evidence from XNNPACK, ONNX Runtime, MNN, and llama.cpp.</p>
    <div class="revision-strip">
      <span>KleidiAI <a href="{KAI_GITHUB}/commit/{h(kai_revision_full)}" target="_blank" rel="noopener noreferrer"><code>{h(kai_describe)}</code></a></span>
      <span>XNNPACK <a href="{XNN_GITHUB}/commit/{h(xnn_revision_full)}" target="_blank" rel="noopener noreferrer"><code>{h(xnn_revision)}</code></a></span>
      <span>ONNX Runtime <a href="{ORT_GITHUB}/commit/{h(ort_revision_full)}" target="_blank" rel="noopener noreferrer"><code>{h(ort_revision)}</code></a></span>
      <span>MNN <a href="{MNN_GITHUB}/commit/{h(mnn_revision_full)}" target="_blank" rel="noopener noreferrer"><code>{h(mnn_revision)}</code></a></span>
      <span>llama.cpp <a href="{LLAMA_GITHUB}/commit/{h(llama_revision_full)}" target="_blank" rel="noopener noreferrer"><code>{h(llama_revision)}</code></a></span>
      <span>Generated <code>{h(now.strftime('%Y-%m-%d %H:%M %Z'))}</code></span>
    </div>
  </header>

  <section class="controls" aria-label="Table filters">
    <div class="control-grid">
      <div class="control search"><label class="control-label" for="search">Search</label><input id="search" type="search" placeholder="Family, datatype, ISA, symbol…"></div>
      {checkbox_filter('operationFilter', 'Operation', operation_options)}
      {checkbox_filter('outputTypeFilter', 'Output type', ((value, value) for value in output_type_values))}
      {checkbox_filter('lhsTypeFilter', 'Input / LHS type', ((value, value) for value in lhs_type_values))}
      {checkbox_filter('rhsTypeFilter', 'Weights / RHS type', ((value, value) for value in rhs_type_values))}
      {checkbox_filter('quantFilter', 'Quantization', quant_options)}
      {checkbox_filter('isaFilter', 'ISA', ((value, value) for value in isa_values))}
      {checkbox_filter('xnnFilter', 'XNNPACK', xnn_integration_options)}
      {checkbox_filter('ortFilter', 'ONNX Runtime', ort_integration_options)}
      {checkbox_filter('mnnFilter', 'MNN', integration_options)}
      {checkbox_filter('llamaFilter', 'llama.cpp', integration_options)}
      <button id="reset" class="reset" type="button">Reset</button>
    </div>
    <div class="result-line"><span id="resultCount" aria-live="polite">{len(groups)} of {len(groups)} groups · {len(variants)} micro-kernels</span><div class="legend">{status_badge('Integrated','integrated')}{status_badge('Partial','partial')}{status_badge('Referenced only','referenced')}{status_badge('Not integrated','not-integrated')}{status_badge('Unavailable at pin','unavailable')}{status_badge('PR available','pr-available')}</div></div>
  </section>

  <section class="table-wrap" aria-label="KleidiAI integration table">
    <table class="main-table" id="statusTable">
      <thead><tr>
        <th><button type="button" data-sort="operation">Operation</button></th>
        <th><button type="button" data-sort="output">Output</button></th>
        <th><button type="button" data-sort="lhs">Input / LHS</button></th>
        <th><button type="button" data-sort="rhs">Weights / RHS</button></th>
        <th><button type="button" data-sort="format">Packed format</button></th>
        <th><button type="button" data-sort="isa">ISA</button></th>
        <th><button type="button" data-sort="xnn">XNNPACK</button></th>
        <th><button type="button" data-sort="ort">ONNX Runtime</button></th>
        <th><button type="button" data-sort="mnn">MNN</button></th>
        <th><button type="button" data-sort="llama">llama.cpp</button></th>
      </tr></thead>
      <tbody id="tableBody">{render_group_rows(groups, xnn_pin, xnn_legacy_pin, ort_pin, mnn_pin, llama_pin, kai_revision_full, xnn_revision_full, ort_revision_full, mnn_revision_full, llama_revision_full)}</tbody>
    </table>
    <div id="emptyState" class="empty" hidden>No groups match the current filters.</div>
  </section>
  <footer>Generated from audited repository snapshots; source and evidence links target exact public GitHub revisions. Open-PR state audited <code>{h(pr_audited_at)}</code>. Embedded audit metadata: <code id="snapshotData"></code></footer>
</main>
<script type="application/json" id="auditData">{json.dumps(data_snapshot, separators=(',', ':')).replace('</', '<\\/')}</script>
<script>
(() => {{
  const table = document.getElementById('statusTable');
  const body = document.getElementById('tableBody');
  const rows = Array.from(body.querySelectorAll('.data-row'));
  const search = document.getElementById('search');
  const filterRoots = {{
    operation: document.getElementById('operationFilter'),
    outputType: document.getElementById('outputTypeFilter'), lhsType: document.getElementById('lhsTypeFilter'),
    rhsType: document.getElementById('rhsTypeFilter'), quant: document.getElementById('quantFilter'),
    isa: document.getElementById('isaFilter'), xnn: document.getElementById('xnnFilter'), ort: document.getElementById('ortFilter'),
    mnn: document.getElementById('mnnFilter'), llama: document.getElementById('llamaFilter')
  }};
  const detailFor = row => document.getElementById(row.dataset.detail);
  const variantsFor = row => Array.from(detailFor(row).querySelectorAll('.variant-table tbody > tr'));
  const checkedValues = root => new Set(
    Array.from(root.querySelectorAll('input[type="checkbox"]:checked'), checkbox => checkbox.value)
  );
  const tokensOverlap = (value, selected) =>
    selected.size === 0 || String(value || '').split('|').some(token => selected.has(token));
  const selectedFilters = () => Object.fromEntries(
    Object.entries(filterRoots).map(([key, root]) => [key, checkedValues(root)])
  );
  function updateFilterSummary(root) {{
    const checked = Array.from(root.querySelectorAll('input[type="checkbox"]:checked'));
    const labels = checked.map(checkbox => checkbox.dataset.label);
    const summary = root.querySelector('.filter-summary');
    summary.textContent = checked.length === 0
      ? 'All'
      : checked.length <= 2 ? labels.join(', ') : `${{checked.length}} selected`;
    summary.title = checked.length === 0 ? 'All' : labels.join(', ');
  }}
  function filterDetailVariants(row, selectedIsas) {{
    const variants = variantsFor(row);
    let visibleVariants = 0;
    variants.forEach(variant => {{
      const show = selectedIsas.size === 0 || selectedIsas.has(variant.dataset.isa);
      variant.hidden = !show;
      if (show) visibleVariants++;
    }});
    const counter = detailFor(row).querySelector('.variant-total');
    if (selectedIsas.size) {{
      counter.textContent = `${{visibleVariants}} of ${{variants.length}} variants · ${{Array.from(selectedIsas).join(', ')}}`;
    }} else {{
      counter.textContent = `${{variants.length}} variant${{variants.length === 1 ? '' : 's'}}`;
    }}
    return visibleVariants;
  }}
  function filter() {{
    const query = search.value.trim().toLowerCase();
    const selected = selectedFilters();
    let visible = 0, kernels = 0;
    rows.forEach(row => {{
      const show = (!query || row.dataset.search.includes(query)) &&
        tokensOverlap(row.dataset.operation, selected.operation) &&
        tokensOverlap(row.dataset.outputTypes, selected.outputType) &&
        tokensOverlap(row.dataset.lhsTypes, selected.lhsType) &&
        tokensOverlap(row.dataset.rhsTypes, selected.rhsType) &&
        tokensOverlap(row.dataset.quant, selected.quant) &&
        tokensOverlap(row.dataset.isa, selected.isa) &&
        tokensOverlap(row.dataset.xnn, selected.xnn) &&
        tokensOverlap(row.dataset.ort, selected.ort) &&
        tokensOverlap(row.dataset.mnn, selected.mnn) &&
        tokensOverlap(row.dataset.llama, selected.llama);
      row.hidden = !show;
      const detail = detailFor(row);
      const matchingVariants = filterDetailVariants(row, selected.isa);
      if (!show) {{
        detail.hidden = true;
        row.querySelector('.expand').setAttribute('aria-expanded', 'false');
      }}
      if (show) {{ visible++; kernels += matchingVariants; }}
    }});
    document.getElementById('resultCount').textContent = `${{visible}} of {len(groups)} groups · ${{kernels}} micro-kernels`;
    document.getElementById('emptyState').hidden = visible !== 0;
    table.hidden = visible === 0;
  }}
  search.addEventListener('input', filter);
  Object.values(filterRoots).forEach(root => {{
    updateFilterSummary(root);
    root.addEventListener('change', () => {{ updateFilterSummary(root); filter(); }});
    root.addEventListener('toggle', () => {{
      if (root.open) {{
        Object.values(filterRoots).forEach(other => {{ if (other !== root) other.open = false; }});
        root.classList.remove('viewport-right');
        requestAnimationFrame(() => {{
          if (root.querySelector('.checklist').getBoundingClientRect().right > window.innerWidth - 8) {{
            root.classList.add('viewport-right');
          }}
        }});
      }}
    }});
  }});
  document.addEventListener('click', event => {{
    if (!event.target.closest('.multi-filter')) Object.values(filterRoots).forEach(root => root.open = false);
  }});
  document.addEventListener('keydown', event => {{
    if (event.key === 'Escape') {{
      const openRoot = Object.values(filterRoots).find(root => root.open);
      if (openRoot) {{
        event.preventDefault();
        openRoot.open = false;
        openRoot.querySelector('summary').focus();
      }}
    }}
  }});
  document.getElementById('reset').addEventListener('click', () => {{
    search.value = '';
    Object.values(filterRoots).forEach(root => {{
      root.querySelectorAll('input[type="checkbox"]').forEach(checkbox => checkbox.checked = false);
      root.open = false;
      updateFilterSummary(root);
    }});
    filter();
  }});
  rows.forEach(row => row.querySelector('.expand').addEventListener('click', event => {{
    const button = event.currentTarget, detail = detailFor(row), open = button.getAttribute('aria-expanded') === 'true';
    button.setAttribute('aria-expanded', String(!open)); detail.hidden = open;
  }}));
  let sortKey = 'operation', ascending = true;
  function sortRows(key) {{
    ascending = key === sortKey ? !ascending : true; sortKey = key;
    const pairs = rows.map(row => [row, detailFor(row)]);
    const cellIndex = {{ operation:0, output:1, lhs:2, rhs:3, format:4, isa:5, xnn:6, ort:7, mnn:8, llama:9 }}[key];
    pairs.sort((a,b) => {{
      const special = ['operation','variants','xnn','ort','mnn','llama'].includes(key);
      let av = special ? a[0].dataset['sort' + key[0].toUpperCase() + key.slice(1)] : a[0].cells[cellIndex].innerText;
      let bv = special ? b[0].dataset['sort' + key[0].toUpperCase() + key.slice(1)] : b[0].cells[cellIndex].innerText;
      if (['variants','xnn','ort','mnn','llama'].includes(key)) {{ av = Number(av); bv = Number(bv); }} else {{ av = String(av).toLowerCase(); bv = String(bv).toLowerCase(); }}
      return (av < bv ? -1 : av > bv ? 1 : 0) * (ascending ? 1 : -1);
    }});
    pairs.forEach(([row,detail]) => {{ body.append(row,detail); }});
  }}
  document.querySelectorAll('[data-sort]').forEach(button => button.addEventListener('click', () => sortRows(button.dataset.sort)));
  const root = document.documentElement;
  document.getElementById('themeToggle').addEventListener('click', () => {{ root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark'; }});
  const snapshot = JSON.parse(document.getElementById('auditData').textContent);
  document.getElementById('snapshotData').textContent = `${{snapshot.kleidiai.revision}} / ${{snapshot.xnnpack.revision}} / ${{snapshot.onnxruntime.revision}} / ${{snapshot.mnn.revision}} / ${{snapshot.llama_cpp.revision}}`;
}})();
</script>
</body>
</html>
'''


def validate_report(output: str, variants: list[Variant], groups: list[Group]) -> None:
    for variant in variants:
        if output.count(h(variant.raw_name)) < 1:
            fail(f"Generated report omits {variant.raw_name}")
    if output.count('class="data-row"') != len(groups):
        fail("Generated report row count does not match grouped inventory")
    if output.count('class="detail-row"') != len(groups):
        fail("Generated report detail-row count does not match grouped inventory")
    if output.count('colspan="10"') != len(groups):
        fail("Generated report detail rows do not span the ten-column table")
    for attribute in ("data-output-types", "data-lhs-types", "data-rhs-types"):
        if output.count(f'{attribute}="') != len(groups):
            fail(f"Generated report {attribute} metadata does not match grouped inventory")
    if output.count('<tr data-isa="') != len(variants):
        fail("Generated report variant ISA metadata does not match inventory")
    if output.count('class="multi-filter"') != 10 or "<select" in output:
        fail("Generated report does not contain the expected checkbox filter controls")
    expected_pr_links = sum(
        len(unique_prs(result.pr_candidates))
        for variant in variants
        for result in (variant.xnn, variant.ort)
    ) + sum(
        len(group_pr_candidates(group, attr))
        for group in groups
        for attr in ("xnn", "ort")
    )
    # One non-link badge appears in the legend in addition to all candidate links.
    if output.count('class="status status-pr-available"') != expected_pr_links + 1:
        fail("Generated report open-PR links do not match the audited candidate mappings")
    if expected_pr_links and output.count('value="pr-available"') != 2:
        fail("Generated report does not expose PR-available framework filters")
    if "Family only" in output or "family-only" in output:
        fail("Generated report contains an ambiguous family-only variant status")
    if 'data-sort="quant"' in output or "Quantization status" in output:
        fail("Generated report still contains the removed quantization column")
    if 'data-sort="family"' in output or "Family / group" in output:
        fail("Generated report still contains the removed family/group column")
    if "KleidiAI release provenance" in output or 'class="release-history"' in output:
        fail("Generated report still contains release provenance sections")
    hrefs = re.findall(r'<a\s+[^>]*href="([^"]+)"', output)
    relative_hrefs = [href for href in hrefs if not href.startswith("https://")]
    if relative_hrefs:
        fail(f"Generated report contains non-public links: {relative_hrefs[:3]}")
    for repository_url in (XNN_GITHUB, ORT_GITHUB, MNN_GITHUB, LLAMA_GITHUB):
        if f'href="{repository_url}/blob/' not in output:
            fail(f"Generated report has no revision-pinned evidence links for {repository_url}")
    if "<html" not in output or "</html>" not in output:
        fail("Generated report is not a complete HTML document")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "kleidiai_integration_status.html",
        help="Output HTML path (default: %(default)s)",
    )
    parser.add_argument("--check", action="store_true", help="Validate inputs and report without writing")
    args = parser.parse_args()

    for required in (KAI_ROOT, XNN_ROOT, ORT_ROOT, MNN_ROOT, LLAMA_ROOT):
        if not required.is_dir():
            fail(f"Required checkout not found: {required}")

    variants = inventory()
    xnn_pin, xnn_legacy_pin, ort_pin, mnn_pin, llama_pin = discover_pins()
    scan_framework(variants, XNN_ROOT, [XNN_ROOT / "src"], "xnn")
    scan_framework(variants, ORT_ROOT, [ORT_ROOT / "onnxruntime" / "core" / "mlas" / "lib"], "ort")
    scan_framework(variants, MNN_ROOT, [MNN_ROOT / "source" / "backend" / "cpu" / "kleidiai"], "mnn")
    scan_framework(
        variants,
        LLAMA_ROOT,
        [LLAMA_ROOT / "ggml" / "src" / "ggml-cpu" / "kleidiai"],
        "llama",
    )
    pr_audited_at = apply_pr_candidates(variants)
    mark_xnn_runtime_selection(variants)
    apply_availability(variants, xnn_pin, xnn_legacy_pin, ort_pin, mnn_pin, llama_pin)
    groups = group_variants(variants)
    output = render_html(
        variants,
        groups,
        xnn_pin,
        xnn_legacy_pin,
        ort_pin,
        mnn_pin,
        llama_pin,
        pr_audited_at,
    )
    validate_report(output, variants, groups)

    counts = {
        "public_microkernels": len(variants),
        "groups": len(groups),
        "xnnpack_direct": sum(bool(variant.xnn.active) for variant in variants),
        "xnnpack_runtime_compute": sum(variant.xnn.runtime_selected for variant in variants),
        "onnxruntime_direct": sum(bool(variant.ort.active) for variant in variants),
        "onnxruntime_reference_only": sum(
            bool(variant.ort.references) and not variant.ort.active for variant in variants
        ),
        "mnn_direct": sum(bool(variant.mnn.active) for variant in variants),
        "mnn_reference_only": sum(
            bool(variant.mnn.references) and not variant.mnn.active for variant in variants
        ),
        "llama_cpp_direct": sum(bool(variant.llama.active) for variant in variants),
        "llama_cpp_reference_only": sum(
            bool(variant.llama.references) and not variant.llama.active for variant in variants
        ),
        "xnnpack_open_prs": len(
            unique_prs(
                pull_request
                for variant in variants
                for pull_request in variant.xnn.pr_candidates
            )
        ),
        "onnxruntime_open_prs": len(
            unique_prs(
                pull_request
                for variant in variants
                for pull_request in variant.ort.pr_candidates
            )
        ),
    }
    if args.check:
        print(json.dumps(counts, indent=2))
        return 0
    args.output.write_text(output, encoding="utf-8")
    print(f"Generated {args.output}")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
