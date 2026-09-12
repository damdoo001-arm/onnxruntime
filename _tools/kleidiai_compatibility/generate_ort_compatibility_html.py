#!/usr/bin/env python3
"""Generate a compact, checkbox-filterable ONNX Runtime compatibility page."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import generate_kernel_status as report
from generate_layout_prototypes import BASE_CSS
from generate_ort_compatibility import integration_status


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(
    os.environ.get(
        "KLEIDIAI_STANDALONE_OUTPUT", ROOT / "kleidiai_onnxruntime_compatibility.html"
    )
).expanduser().resolve()
PAGES_OUTPUT = Path(
    os.environ.get(
        "KLEIDIAI_PAGES_OUTPUT",
        ROOT / "onnxruntime" / "docs" / "performance" / "kleidiai" / "index.html",
    )
).expanduser().resolve()

MICROKERNEL_TABLE = report.KAI_ROOT / "docs" / "microkernel_tables.md"
ELASTIC_REGISTRY = (
    report.KAI_ROOT
    / "test"
    / "nextgen"
    / "operators"
    / "matmul"
    / "matmul"
    / "matmul_wrapper_registry.cpp"
)

ORT_OPERATION_SOURCES = {
    "Conv": "onnxruntime/core/providers/cpu/nn/conv.cc",
    "FusedConv": "onnxruntime/contrib_ops/cpu/fused_conv.cc",
    "NhwcFusedConv": "onnxruntime/core/providers/cpu/fp16/fp16_conv.cc",
    "MatMul": "onnxruntime/core/providers/cpu/math/matmul.cc",
    "Gemm": "onnxruntime/core/providers/cpu/math/gemm.cc",
    "MatMulNBits": "onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc",
    "DynamicQuantizeMatMul": (
        "onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_matmul.cc"
    ),
    "MatMulIntegerToFloat": (
        "onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_matmul.cc"
    ),
    "MoE": "onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc",
}


def elastic_kernel_names() -> set[str]:
    """Resolve kernels explicitly exposed as elastic by KleidiAI's operator API."""

    text = ELASTIC_REGISTRY.read_text(encoding="utf-8", errors="replace")
    names = set()
    for match in re.finditer(
        r"create_[A-Za-z0-9_]*elastic[A-Za-z0-9_]*\(\)\s*\{(?P<body>.*?)\n\}",
        text,
        re.DOTALL,
    ):
        kernels = re.findall(r"\b(kai_[A-Za-z0-9_]+)\(\)", match.group("body"))
        if kernels:
            names.add(kernels[0])
    if not names:
        report.fail(f"No elastic GEMM wrapper was found in {ELASTIC_REGISTRY}")
    return names


def tile_kind(variant: report.Variant, elastic_names: set[str]) -> str:
    """Only KleidiAI kernels explicitly wrapped as elastic are variable."""

    return "Variable" if variant.raw_name in elastic_names else "Fixed"


def pack_side(variant: report.Variant) -> str | None:
    if variant.operation_key == "lhs-pack" or "_pack_lhs_" in variant.canonical_name:
        return "lhs"
    if (
        variant.operation_key in {"rhs-pack-kxn", "rhs-pack-nxk", "dwconv-rhs-pack"}
        or "_pack_rhs_" in variant.canonical_name
    ):
        return "rhs"
    return None


def documented_packer_map(
    variants: list[report.Variant], revision: str
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Read the exact compute-to-packer relationships published by KleidiAI."""

    compute_names = {variant.raw_name for variant in variants if variant.is_compute}
    packers = {variant.raw_name: variant for variant in variants if not variant.is_compute}
    mapping = {name: {"lhs": [], "rhs": []} for name in compute_names}
    for line in MICROKERNEL_TABLE.read_text(encoding="utf-8").splitlines():
        names = re.findall(r"`(kai_[A-Za-z0-9_]+)`", line)
        compute = next((name for name in names if name in compute_names), None)
        if compute is None:
            continue
        for name in names:
            if name == compute:
                continue
            packer = packers.get(name)
            if packer is None:
                report.fail(f"Documented packer {name} is absent from the inventory")
            side = pack_side(packer)
            if side is None:
                report.fail(f"Cannot classify documented packer {name} as LHS or RHS")
            item = {
                "name": name,
                "url": f"{report.KAI_GITHUB}/blob/{revision}/{packer.relative_c}",
            }
            if item not in mapping[compute][side]:
                mapping[compute][side].append(item)
    missing = sorted(name for name, sides in mapping.items() if not any(sides.values()))
    # Kernels that require no packing legitimately have an empty association.
    if len(missing) > 1:
        report.fail(
            f"Packer table did not resolve associations for {len(missing)} compute kernels"
        )
    return mapping


def framework_operations(
    variant: report.Variant, ort_revision: str
) -> list[dict[str, str]]:
    """Map an integrated MLAS kernel to directly traceable ONNX Runtime operators."""

    if not variant.ort.active:
        return []
    family = variant.canonical_family
    if variant.operation_key == "imatmul":
        names = ["Conv", "FusedConv", "NhwcFusedConv"]
    elif "qai8dxp_qsi4" in family or "qsi8d32p_qai4" in family:
        names = ["MatMulNBits"]
    elif "qai8dxp_qsi8" in family:
        names = ["DynamicQuantizeMatMul", "MatMulIntegerToFloat"]
    elif "bf16p_bf16p" in family:
        names = ["MatMul"]
    elif family.startswith("matmul_clamp_f16_f16_f16p"):
        names = ["MatMul", "Gemm", "Conv", "FusedConv", "NhwcFusedConv", "MoE"]
    elif family.startswith("matmul_clamp_f32_f32"):
        names = ["MatMul", "Gemm", "MoE"]
    else:
        names = []
    operations = []
    for name in names:
        relative = ORT_OPERATION_SOURCES[name]
        if (
            family.startswith("imatmul_clamp_f16_")
            or family.startswith("matmul_clamp_f16_f16_f16p")
        ) and name in {
            "Conv",
            "FusedConv",
            "NhwcFusedConv",
        }:
            relative = "onnxruntime/core/providers/cpu/fp16/fp16_conv.cc"
        elif variant.operation_key == "imatmul" and name in {
            "FusedConv",
            "NhwcFusedConv",
        }:
            relative = "onnxruntime/contrib_ops/cpu/fused_conv.cc"
        source = report.ORT_ROOT / relative
        source_text = (
            source.read_text(encoding="utf-8", errors="replace")
            if source.is_file()
            else ""
        )
        if name not in source_text:
            report.fail(f"ONNX operator evidence for {name} is missing from {relative}")
        line = source_text[: source_text.index(name)].count("\n") + 1
        operations.append(
            {
                "name": name,
                "url": f"{report.ORT_GITHUB}/blob/{ort_revision}/{relative}#L{line}",
            }
        )
    return operations


def operand_datatypes(variant: report.Variant) -> tuple[str, str, str]:
    """Return compact datatypes, retaining only the packed distinction."""

    def label(descriptor: report.Descriptor) -> str:
        return f"{descriptor.type_label}{' (P)' if descriptor.packed else ''}"

    descriptors = variant.descriptors
    output = label(descriptors[0]) if descriptors else "—"
    operands = descriptors[1:]
    lhs = operands[:1]
    rhs = operands[1:]
    join = lambda values: " · ".join(label(value) for value in values) or "—"
    return output, join(lhs), join(rhs)


def collect() -> tuple[list[dict[str, object]], dict[str, str]]:
    variants = report.inventory()
    report.scan_framework(
        variants,
        report.ORT_ROOT,
        [report.ORT_ROOT / "onnxruntime" / "core" / "mlas" / "lib"],
        "ort",
    )
    audited_at = report.apply_pr_candidates(variants)
    kai_revision = report.run_git(report.KAI_ROOT, "rev-parse", "HEAD")
    ort_revision = report.run_git(report.ORT_ROOT, "rev-parse", "HEAD")
    packer_map = documented_packer_map(variants, kai_revision)
    elastic_names = elastic_kernel_names()
    variants = [variant for variant in variants if variant.is_compute]
    records = []
    for variant in variants:
        output, lhs, rhs = operand_datatypes(variant)
        status_key, status_label = integration_status(variant)
        evidence = []
        ort_evidence = variant.ort.active or variant.ort.references
        evidence_kind = "integration" if variant.ort.active else "reference"
        for index, item in enumerate(ort_evidence[:2], start=1):
            evidence.append(
                {
                    "label": evidence_kind if index == 1 else f"{evidence_kind} {index}",
                    "url": f"{report.ORT_GITHUB}/blob/{ort_revision}/{item.path}#L{item.line}",
                }
            )
        for pull_request in report.unique_prs(variant.ort.pr_candidates):
            evidence.append({"label": f"PR #{pull_request.number}", "url": pull_request.url})
        records.append(
            {
                "name": variant.raw_name,
                "source": f"{report.KAI_GITHUB}/blob/{kai_revision}/{variant.relative_c}",
                "operation": variant.operation,
                "output": output,
                "lhs": lhs,
                "rhs": rhs,
                "signature": " · ".join(item.raw.upper() for item in variant.descriptors) or "—",
                "extension": variant.isa,
                "tile": variant.tile,
                "tileKind": tile_kind(variant, elastic_names),
                "lhsPacks": packer_map[variant.raw_name]["lhs"],
                "rhsPacks": packer_map[variant.raw_name]["rhs"],
                "status": status_key,
                "statusLabel": status_label.removeprefix("✅ ")
                .removeprefix("🔗 ")
                .removeprefix("🟣 ")
                .removeprefix("— "),
                "evidence": evidence,
                "ortOps": framework_operations(variant, ort_revision),
            }
        )
    metadata = {
        "kai_revision": kai_revision,
        "kai_describe": report.describe_revision(report.KAI_ROOT),
        "ort_revision": ort_revision,
        "ort_short": report.short_revision(report.ORT_ROOT),
        "audited_at": audited_at,
    }
    return records, metadata


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="KleidiAI compatibility in ONNX Runtime"><title>KleidiAI compatibility in ONNX Runtime</title><style>__BASE_CSS__
.controls{position:sticky;z-index:10;top:0;margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:13px;background:color-mix(in srgb,var(--surface) 94%,transparent);backdrop-filter:blur(14px)}.search-row{display:grid;grid-template-columns:minmax(240px,1fr) auto;gap:9px}.clear{border:1px solid var(--line);border-radius:9px;background:var(--surface3);padding:8px 13px;cursor:pointer}.dropdown-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:9px}.filter-menu{position:relative;min-width:0}.filter-menu summary{display:flex;min-height:50px;flex-direction:column;justify-content:center;gap:2px;padding:7px 10px;border:1px solid var(--line);border-radius:9px;background:var(--surface2);cursor:pointer;list-style:none}.filter-menu summary::-webkit-details-marker{display:none}.filter-menu summary:after{content:"▾";position:absolute;right:10px;top:16px;color:var(--muted)}.filter-menu[open] summary{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}.filter-menu[open] summary:after{transform:rotate(180deg)}.filter-name{color:var(--muted);font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.filter-value{max-width:calc(100% - 20px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.checklist{position:absolute;z-index:30;top:calc(100% + 5px);left:0;width:max(100%,280px);max-height:330px;overflow:auto;padding:7px;border:1px solid var(--line);border-radius:10px;background:var(--surface2);box-shadow:0 18px 45px rgba(0,0,0,.32)}.filter-menu:nth-child(3n) .checklist{right:0;left:auto}.check-option{display:grid;grid-template-columns:16px minmax(0,1fr) auto;align-items:start;gap:8px;padding:7px;border-radius:7px;cursor:pointer}.check-option:hover{background:var(--surface3)}.check-option input{margin:3px 0 0;accent-color:var(--accent)}.check-option .count{color:var(--muted);font-size:11px}.active{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}.filter-chip{border:1px solid var(--line);border-radius:999px;background:var(--surface2);padding:4px 8px;cursor:pointer;font-size:11px}.results-head{display:flex;align-items:end;justify-content:space-between;gap:12px;margin:0 0 10px}.results-head h2{margin:0;font-size:17px}.main-table{width:100%;min-width:0;table-layout:fixed}.main-table .kai-toggle{width:3rem}.main-table .kai-datatype{width:15%}.main-table .kai-extension{width:14%}.main-table .kai-status{width:26%}.main-table>thead th{overflow-wrap:anywhere}.main-table>tbody>.group-row>td{padding:9px 8px}.main-table td:first-child,.main-table th:first-child{text-align:center}.operation-header th{padding:10px 12px;background:var(--surface3);color:var(--text);text-align:left}.operation-header span{margin-left:6px;color:var(--muted);font-size:11px;font-weight:400}.expand-row{border:0;background:transparent;color:var(--accent);cursor:pointer;font-size:14px}.detail-row>td{padding:0 8px 10px}.kernel-details{padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--surface2)}.kernel-details table{width:100%;margin:0;table-layout:fixed}.kernel-details th,.kernel-details td{padding:7px;text-align:left;overflow-wrap:anywhere}.kernel-details th:nth-child(1){width:18%}.kernel-details th:nth-child(2){width:28%}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.status{font-weight:700}.status-integrated{color:var(--green)}.status-referenced{color:#8ecbff}.status-pr{color:#c4b5fd}.status-not-integrated{color:var(--muted)}.links{display:flex;flex-wrap:wrap;gap:6px;margin-top:3px}@media(max-width:900px){.dropdown-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.filter-menu:nth-child(3n) .checklist{right:auto;left:0}.filter-menu:nth-child(2n) .checklist{right:0;left:auto}}@media(max-width:520px){.shell{width:min(100% - 16px,1600px);padding-top:8px}.controls{position:static}.search-row,.dropdown-grid{grid-template-columns:1fr}.filter-menu:nth-child(2n) .checklist{right:auto;left:0}.checklist{width:100%}}
</style><style>.controls{padding:10px}.dropdown-grid{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}.filter-menu{flex:1 1 145px}.filter-menu summary{min-height:34px;flex-direction:row;align-items:center;justify-content:space-between;gap:6px;padding:5px 24px 5px 8px}.filter-menu summary:after{top:8px}.filter-value{font-size:11px;font-weight:600}.filter-menu .checklist{right:auto;left:0}.filter-menu:last-child .checklist{right:0;left:auto}.main-table{width:100%!important;max-width:100%!important;min-width:0!important;table-layout:fixed!important}.main-table>thead>tr>th:nth-child(1),.main-table>tbody>.group-row>td:nth-child(1){width:4%!important;text-align:center}.main-table>thead>tr>th:nth-child(n+2):nth-child(-n+4),.main-table>tbody>.group-row>td:nth-child(n+2):nth-child(-n+4){width:14%!important}.main-table>thead>tr>th:nth-child(5),.main-table>tbody>.group-row>td:nth-child(5){width:14%!important}.main-table>thead>tr>th:nth-child(6),.main-table>tbody>.group-row>td:nth-child(6){width:40%!important}.main-table>thead th,.main-table>tbody>.group-row>td{overflow-wrap:anywhere;word-break:break-word}.kernel-details table{width:100%!important;min-width:0!important;table-layout:fixed!important}.kernel-details th:nth-child(1){width:9%!important}.kernel-details th:nth-child(2){width:12%!important}.kernel-details th:nth-child(3),.kernel-details th:nth-child(4){width:19%!important}.kernel-details th:nth-child(5){width:41%!important}.kernel-details th,.kernel-details td{overflow-wrap:anywhere;word-break:break-word}.datatype{display:inline-block;padding:2px 5px;border-radius:5px}.type-output{color:#8ecbff;background:rgba(88,166,255,.12)}.type-lhs{color:#6ce8bb;background:rgba(108,232,187,.1)}.type-rhs{color:#c4b5fd;background:rgba(196,181,253,.1)}.extension-badge{color:#8ecbff}.status-partial{color:var(--amber)}.kernel-link{color:#8ecbff}.pack-links{display:flex;flex-wrap:wrap;gap:5px}.lhs-pack{color:#6ce8bb}.rhs-pack{color:#c4b5fd}.operator-links{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin-top:4px;font-size:11px}.operator-links span{color:var(--muted)}.operator-links a{padding:1px 5px;border-radius:999px;background:rgba(88,166,255,.1)}@media(max-width:700px){.table-wrap{overflow-x:auto}.main-table{min-width:46rem!important}}</style></head><body><main class="shell"><header class="hero"><h1>KleidiAI compatibility in ONNX Runtime</h1><p class="lede">A focused compatibility view with grouped operations and multi-select requirement filters.</p><div class="meta"><span>KleidiAI <a href="__KAI_URL__"><code>__KAI_DESCRIBE__</code></a></span> · <span>ONNX Runtime <a href="__ORT_URL__"><code>__ORT_SHORT__</code></a></span> · <span>PR audit __AUDITED_AT__</span></div></header>
<section class="controls" aria-label="Kernel filters"><div class="search-row"><input id="search" class="search" type="search" placeholder="Search operations, datatypes, or kernel symbols…"><button id="clear" class="clear" type="button">Clear</button></div><div id="dropdowns" class="dropdown-grid"></div><div id="activeFilters" class="active"></div></section><section class="results"><div class="results-head"><h2>Matching operations</h2><span id="resultCount" class="muted count" aria-live="polite"></span></div><div class="table-wrap"><table class="main-table"><thead><tr><th aria-label="Expand kernels"></th><th>Output datatype</th><th>LHS datatype</th><th>RHS datatype</th><th>Extension Type</th><th>ONNX Runtime</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No operations match these filters.</div></div></section></main>
<script type="application/json" id="kernelData">__DATA__</script><script>
(()=>{const data=JSON.parse(document.getElementById('kernelData').textContent);const defs=[['operation','Operation'],['output','Output'],['lhs','LHS'],['rhs','RHS'],['extension','Extension'],['tileKind','Tile mode'],['ortOps','ONNX operator'],['status','ORT status']];const labels={integrated:'Integrated',referenced:'Referenced only',pr:'PR available','not-integrated':'Not integrated'};const selected=Object.fromEntries(defs.map(([key])=>[key,new Set()]));const dropdowns=document.getElementById('dropdowns'),search=document.getElementById('search'),body=document.getElementById('results');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fieldValues=(record,key)=>{const value=record[key];if(!Array.isArray(value))return[value];return value.map(item=>typeof item==='object'?item.name:item)};const values=key=>[...new Set(data.flatMap(record=>fieldValues(record,key)).filter(value=>value&&value!=='—'))].sort((a,b)=>String(labels[a]||a).localeCompare(String(labels[b]||b)));
function matches(record,omit=''){const query=search.value.trim().toLowerCase();const operationNames=record.ortOps.map(item=>item.name).join(' ');if(query&&!`${record.name} ${record.operation} ${record.output} ${record.lhs} ${record.rhs} ${record.signature} ${record.extension} ${operationNames}`.toLowerCase().includes(query))return false;return defs.every(([key])=>key===omit||!selected[key].size||fieldValues(record,key).some(value=>selected[key].has(value)))}
function renderDropdowns(openKey=''){dropdowns.innerHTML=defs.map(([key,label])=>{const chosen=[...selected[key]].map(value=>labels[value]||value);const summary=chosen.length===0?'All':chosen.length<=2?chosen.join(', '):`${chosen.length} selected`;return `<details class="filter-menu" data-key="${key}" ${openKey===key?'open':''}><summary><span class="filter-name">${esc(label)}</span><strong class="filter-value" title="${esc(chosen.join(', '))}">${esc(summary)}</strong></summary><div class="checklist" role="group" aria-label="${esc(label)}">${values(key).map(value=>{const count=data.filter(record=>matches(record,key)&&fieldValues(record,key).includes(value)).length;return `<label class="check-option"><input type="checkbox" data-key="${key}" value="${esc(value)}" ${selected[key].has(value)?'checked':''}><span>${esc(labels[value]||value)}</span><span class="count">${count}</span></label>`}).join('')}</div></details>`}).join('');dropdowns.querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{input.checked?selected[input.dataset.key].add(input.value):selected[input.dataset.key].delete(input.value);render(input.dataset.key)}));dropdowns.querySelectorAll('details').forEach(detail=>detail.addEventListener('toggle',()=>{if(detail.open)dropdowns.querySelectorAll('details').forEach(other=>{if(other!==detail)other.open=false})}))}
function renderActive(){const entries=defs.flatMap(([key,label])=>[...selected[key]].map(value=>({key,label,value})));document.getElementById('activeFilters').innerHTML=entries.map(item=>`<button class="filter-chip" type="button" data-key="${item.key}" data-value="${esc(item.value)}">${esc(item.label)}: ${esc(labels[item.value]||item.value)} ×</button>`).join('');document.querySelectorAll('.filter-chip').forEach(button=>button.addEventListener('click',()=>{selected[button.dataset.key].delete(button.dataset.value);render()}))}
function groupRecords(records){const groups=new Map();for(const record of records){const key=[record.operation,record.output,record.lhs,record.rhs,record.extension].join('\u0000');if(!groups.has(key))groups.set(key,{operation:record.operation,output:record.output,lhs:record.lhs,rhs:record.rhs,extension:record.extension,kernels:[]});groups.get(key).kernels.push(record)}return [...groups.values()].sort((a,b)=>a.operation.localeCompare(b.operation)||a.output.localeCompare(b.output)||a.lhs.localeCompare(b.lhs)||a.rhs.localeCompare(b.rhs)||a.extension.localeCompare(b.extension))}
function statusSummary(kernels){const counts=new Map();for(const kernel of kernels)counts.set(kernel.status,(counts.get(kernel.status)||0)+1);if(counts.size===1){const status=kernels[0].status;return `<span class="status status-${esc(status)}">${esc(labels[status]||kernels[0].statusLabel)}</span>`}const covered=kernels.filter(kernel=>kernel.status!=='not-integrated').length;return `<span class="status status-partial">${covered} of ${kernels.length} covered</span>`}
function operationLinks(kernels){const items=new Map();kernels.flatMap(kernel=>kernel.ortOps).forEach(item=>items.set(item.name,item));return items.size?`<div class="operator-links"><span>Eligible ops:</span>${[...items.values()].map(item=>`<a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">${esc(item.name)}</a>`).join('')}</div>`:''}
function packLinks(items,className,label){return items.length?`<div class="pack-links">${items.map((item,index)=>`<a class="${className}" href="${esc(item.url)}" title="${esc(item.name)}" target="_blank" rel="noopener noreferrer">${esc(label)}${items.length>1?` ${index+1}`:''}</a>`).join('')}</div>`:'<span class="muted">—</span>'}
function renderResults(){const shown=data.filter(record=>matches(record));const groups=groupRecords(shown);const operations=new Map();for(const group of groups){if(!operations.has(group.operation))operations.set(group.operation,[]);operations.get(group.operation).push(group)}document.getElementById('resultCount').textContent=`${operations.size} operations · ${groups.length} variants · ${shown.length} kernels`;document.getElementById('empty').hidden=groups.length>0;let index=0;body.innerHTML=[...operations].map(([operation,variants])=>{const operationKernelCount=variants.reduce((count,variant)=>count+variant.kernels.length,0);return `<tr class="operation-header"><th colspan="6">${esc(operation)} <span>${variants.length} variant${variants.length===1?'':'s'} · ${operationKernelCount} kernel${operationKernelCount===1?'':'s'}</span></th></tr>${variants.map(group=>{const detailsId=`kernel-details-${index++}`;const kernels=[...group.kernels].sort((a,b)=>a.tile.localeCompare(b.tile)||a.name.localeCompare(b.name));return `<tr class="group-row"><td><button class="expand-row" type="button" aria-expanded="false" aria-controls="${detailsId}" title="Show ${kernels.length} kernels"><span aria-hidden="true">▶</span><span class="sr-only">Show kernels</span></button></td><td><code class="datatype type-output">${esc(group.output)}</code></td><td><code class="datatype type-lhs">${esc(group.lhs)}</code></td><td><code class="datatype type-rhs">${esc(group.rhs)}</code></td><td><span class="badge extension-badge">${esc(group.extension)}</span></td><td>${statusSummary(kernels)}${operationLinks(kernels)}</td></tr><tr id="${detailsId}" class="detail-row" hidden><td colspan="6"><div class="kernel-details"><table><thead><tr><th>Tile</th><th>Kernel</th><th>LHS packer</th><th>RHS packer</th><th>ONNX Runtime</th></tr></thead><tbody>${kernels.map(kernel=>`<tr><td>${kernel.tile!=='—'?`<span class="badge tile-badge">${esc(kernel.tile)}</span>`:'—'}</td><td><a class="kernel-link" href="${esc(kernel.source)}" title="${esc(kernel.name)}" target="_blank" rel="noopener noreferrer">kernel</a></td><td>${packLinks(kernel.lhsPacks,'pack-link lhs-pack','packer')}</td><td>${packLinks(kernel.rhsPacks,'pack-link rhs-pack','packer')}</td><td><span class="status status-${esc(kernel.status)}">${esc(kernel.statusLabel)}</span>${operationLinks([kernel])}<div class="links">${kernel.evidence.length?kernel.evidence.map(item=>`<a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">${esc(item.label)}</a>`).join(''):'—'}</div></td></tr>`).join('')}</tbody></table></div></td></tr>`}).join('')}`}).join('');body.querySelectorAll('.expand-row').forEach(button=>button.addEventListener('click',()=>{const expanded=button.getAttribute('aria-expanded')==='true';button.setAttribute('aria-expanded',String(!expanded));button.querySelector('[aria-hidden]').textContent=expanded?'▶':'▼';document.getElementById(button.getAttribute('aria-controls')).hidden=expanded}))}
function render(openKey=''){renderResults();renderActive();renderDropdowns(openKey)}search.addEventListener('input',()=>render());document.getElementById('clear').addEventListener('click',()=>{Object.values(selected).forEach(set=>set.clear());search.value='';render()});document.addEventListener('click',event=>{if(!event.target.closest('.filter-menu'))dropdowns.querySelectorAll('details').forEach(detail=>detail.open=false)});document.addEventListener('keydown',event=>{if(event.key==='Escape'){const open=[...dropdowns.querySelectorAll('details')].find(detail=>detail.open);if(open){open.open=false;open.querySelector('summary').focus()}}});render()})();
</script></body></html>'''


PAGES_TEMPLATE = r'''---
layout: default
title: KleidiAI
description: KleidiAI micro-kernel compatibility in ONNX Runtime
parent: Performance
nav_order: 7
toc: false
redirect_from:
  - /docs/reference/kleidiai/
---
<div class="kleidiai-page">
  <h1>KleidiAI compatibility in ONNX Runtime</h1>
  <p class="kai-lede">Public KleidiAI micro-kernels grouped by operation and their exact ONNX Runtime MLAS integration status.</p>
  <p class="kai-meta"><span>KleidiAI <a href="__KAI_URL__"><code>__KAI_DESCRIBE__</code></a></span><span>ONNX Runtime <a href="__ORT_URL__"><code>__ORT_SHORT__</code></a></span><span>PR audit __AUDITED_AT__</span></p>
  <p class="kai-note">Eligible ONNX operators are a conservative set traced directly from integrated MLAS paths. Actual acceleration also depends on datatype, shape, CPU features, packing, and runtime configuration.</p>
  <section class="controls" aria-label="Kernel filters">
    <div class="search-row"><input id="search" class="search" type="search" placeholder="Search operations, datatypes, or kernel symbols…" aria-label="Search kernels"><button id="clear" class="clear" type="button">Clear</button></div>
    <div id="dropdowns" class="dropdown-grid"></div><div id="activeFilters" class="active"></div>
  </section>
  <section class="results"><div class="results-head"><h2>Matching operations</h2><span id="resultCount" class="muted count" aria-live="polite"></span></div><div class="table-wrap"><table class="main-table"><thead><tr><th aria-label="Expand kernels"></th><th>Output datatype</th><th>LHS datatype</th><th>RHS datatype</th><th>Extension Type</th><th>ONNX Runtime</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No operations match these filters.</div></div></section>
</div>
<style>
.kleidiai-page{--kai-border:#eeebee;--kai-surface:#f5f6fa;--kai-hover:#ebedf5;--kai-text:#5c5962;--kai-heading:#27262b;--kai-muted:#716f75;--kai-link:#226aca;color:var(--kai-text)}
.kleidiai-page .kai-lede{font-size:1.05rem}.kleidiai-page .kai-meta{display:flex;flex-wrap:wrap;gap:.35rem 1rem;color:var(--kai-muted);font-size:.8rem}
.kleidiai-page button,.kleidiai-page input{font:inherit}.kleidiai-page .controls{position:relative;z-index:5;margin:1.25rem 0;padding:.9rem;border:1px solid var(--kai-border);border-radius:.35rem;background:var(--kai-surface)}.kleidiai-page .search-row{display:grid;grid-template-columns:minmax(15rem,1fr) auto;gap:.55rem}.kleidiai-page .search{width:100%;min-width:0;padding:.6rem .7rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;color:var(--kai-text);outline:none}.kleidiai-page .search:focus{border-color:var(--kai-link);box-shadow:0 0 0 2px rgba(34,106,202,.18)}.kleidiai-page .clear{padding:.55rem .8rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;color:var(--kai-link);cursor:pointer}.kleidiai-page .clear:hover{background:var(--kai-hover)}
.kleidiai-page .dropdown-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.5rem;margin-top:.55rem}.kleidiai-page .filter-menu{position:relative;min-width:0;margin:0}.kleidiai-page .filter-menu summary{position:relative;display:flex;min-height:3rem;flex-direction:column;justify-content:center;gap:.1rem;padding:.45rem 1.6rem .45rem .6rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;cursor:pointer;list-style:none}.kleidiai-page .filter-menu summary::-webkit-details-marker{display:none}.kleidiai-page .filter-menu summary:after{content:"▾";position:absolute;right:.6rem;top:.85rem;color:var(--kai-muted)}.kleidiai-page .filter-menu[open] summary{border-color:var(--kai-link);box-shadow:0 0 0 2px rgba(34,106,202,.18)}.kleidiai-page .filter-menu[open] summary:after{transform:rotate(180deg)}.kleidiai-page .filter-name{color:var(--kai-muted);font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}.kleidiai-page .filter-value{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--kai-heading);font-size:.78rem}
.kleidiai-page .checklist{position:absolute;z-index:30;top:calc(100% + .3rem);left:0;width:max(100%,17rem);max-height:20rem;overflow:auto;padding:.4rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;box-shadow:0 .75rem 2rem rgba(39,38,43,.18)}.kleidiai-page .filter-menu:nth-child(3n) .checklist{right:0;left:auto}.kleidiai-page .check-option{display:grid;grid-template-columns:1rem minmax(0,1fr) auto;align-items:start;gap:.45rem;padding:.4rem;border-radius:.2rem;cursor:pointer;font-size:.78rem}.kleidiai-page .check-option:hover{background:var(--kai-hover)}.kleidiai-page .check-option input{margin:.15rem 0 0;accent-color:var(--kai-link)}.kleidiai-page .check-option .count,.kleidiai-page .muted{color:var(--kai-muted)}
.kleidiai-page .active{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.55rem}.kleidiai-page .filter-chip{padding:.25rem .5rem;border:1px solid #d7d4d7;border-radius:999px;background:#fff;color:var(--kai-text);cursor:pointer;font-size:.7rem}.kleidiai-page .filter-chip:hover{border-color:var(--kai-link);color:var(--kai-link)}
.kleidiai-page .results-head{display:flex;align-items:end;justify-content:space-between;gap:.75rem;margin:1.25rem 0 .55rem}.kleidiai-page .results-head h2{margin:0}.kleidiai-page .count{font-size:.8rem;font-variant-numeric:tabular-nums}.kleidiai-page .table-wrap{overflow-x:auto;border:1px solid var(--kai-border);border-radius:.35rem}.kleidiai-page .main-table{display:table;width:100%;min-width:0;margin:0;border-collapse:collapse;table-layout:fixed}.kleidiai-page .main-table .kai-toggle{width:3rem}.kleidiai-page .main-table .kai-datatype{width:15%}.kleidiai-page .main-table .kai-extension{width:14%}.kleidiai-page .main-table .kai-status{width:26%}.kleidiai-page .main-table th,.kleidiai-page .main-table td{padding:.55rem .5rem;border-right:0;border-bottom:1px solid var(--kai-border);vertical-align:top;text-align:left}.kleidiai-page .main-table>thead th{overflow-wrap:anywhere}.kleidiai-page .main-table th{background:var(--kai-surface);color:var(--kai-muted);font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}.kleidiai-page .main-table td{background:#fff;font-size:.78rem}.kleidiai-page .main-table tbody .group-row:hover td{background:#fafbfc}.kleidiai-page .main-table td:first-child,.kleidiai-page .main-table th:first-child{text-align:center}.kleidiai-page .operation-header th{padding:.65rem .7rem;background:var(--kai-hover);color:var(--kai-heading);font-size:.8rem;text-align:left;text-transform:none;letter-spacing:0}.kleidiai-page .operation-header span{margin-left:.35rem;color:var(--kai-muted);font-size:.7rem;font-weight:400}.kleidiai-page .badge{display:inline-flex;margin:.1rem .15rem .1rem 0;padding:.1rem .4rem;border:1px solid var(--kai-border);border-radius:999px;background:var(--kai-surface);color:var(--kai-muted);font-size:.65rem;white-space:nowrap}.kleidiai-page .expand-row{padding:.15rem;border:0;background:transparent;color:var(--kai-link);cursor:pointer;font-size:.8rem}.kleidiai-page .detail-row>td{padding:0 .5rem .5rem;background:#fff}.kleidiai-page .kernel-details{padding:.5rem;border:1px solid var(--kai-border);border-radius:.25rem;background:var(--kai-surface)}.kleidiai-page .kernel-details table{display:table;width:100%;min-width:0;margin:0;border-collapse:collapse;table-layout:fixed}.kleidiai-page .kernel-details th,.kleidiai-page .kernel-details td{padding:.4rem;border-bottom:1px solid var(--kai-border);background:transparent;text-align:left;overflow-wrap:anywhere}.kleidiai-page .kernel-details th:nth-child(1){width:18%}.kleidiai-page .kernel-details th:nth-child(2){width:28%}.kleidiai-page .kernel-details tr:last-child td{border-bottom:0}.kleidiai-page .sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.kleidiai-page .status{font-weight:600}.kleidiai-page .status-integrated{color:#13795b}.kleidiai-page .status-referenced{color:#1e5fb4}.kleidiai-page .status-pr{color:#6f42c1}.kleidiai-page .status-not-integrated{color:var(--kai-muted)}.kleidiai-page .links{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.15rem}.kleidiai-page .empty{padding:2rem;text-align:center;color:var(--kai-muted)}
@media(max-width:50rem){.kleidiai-page .dropdown-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.kleidiai-page .filter-menu:nth-child(3n) .checklist{right:auto;left:0}.kleidiai-page .filter-menu:nth-child(2n) .checklist{right:0;left:auto}}@media(max-width:31.25rem){.kleidiai-page .search-row,.kleidiai-page .dropdown-grid{grid-template-columns:1fr}.kleidiai-page .filter-menu:nth-child(2n) .checklist{right:auto;left:0}.kleidiai-page .checklist{width:100%}}
</style>
<style>
.kleidiai-page .kai-note{margin:.75rem 0;padding:.55rem .7rem;border-left:.18rem solid var(--kai-link);background:var(--kai-surface);color:var(--kai-muted);font-size:.75rem}.kleidiai-page .controls{padding:.65rem}.kleidiai-page .dropdown-grid{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.4rem}.kleidiai-page .filter-menu{flex:1 1 8.5rem}.kleidiai-page .filter-menu summary{min-height:2.15rem;flex-direction:row;align-items:center;justify-content:space-between;gap:.35rem;padding:.28rem 1.35rem .28rem .45rem}.kleidiai-page .filter-menu summary:after{right:.45rem;top:.48rem}.kleidiai-page .filter-name{font-size:.58rem}.kleidiai-page .filter-value{max-width:55%;font-size:.68rem}.kleidiai-page .filter-menu .checklist{right:auto;left:0}.kleidiai-page .filter-menu:last-child .checklist{right:0;left:auto}.kleidiai-page .table-wrap{overflow-x:hidden}.kleidiai-page .main-table{width:100%!important;max-width:100%!important;min-width:0!important;table-layout:fixed!important}.kleidiai-page .main-table>thead>tr>th:nth-child(1),.kleidiai-page .main-table>tbody>.group-row>td:nth-child(1){width:4%!important;text-align:center}.kleidiai-page .main-table>thead>tr>th:nth-child(n+2):nth-child(-n+4),.kleidiai-page .main-table>tbody>.group-row>td:nth-child(n+2):nth-child(-n+4){width:14%!important}.kleidiai-page .main-table>thead>tr>th:nth-child(5),.kleidiai-page .main-table>tbody>.group-row>td:nth-child(5){width:14%!important}.kleidiai-page .main-table>thead>tr>th:nth-child(6),.kleidiai-page .main-table>tbody>.group-row>td:nth-child(6){width:40%!important}.kleidiai-page .main-table>thead th,.kleidiai-page .main-table>tbody>.group-row>td{overflow-wrap:anywhere;word-break:break-word}.kleidiai-page .kernel-details table{width:100%!important;min-width:0!important;table-layout:fixed!important}.kleidiai-page .kernel-details th:nth-child(1){width:9%!important}.kleidiai-page .kernel-details th:nth-child(2){width:12%!important}.kleidiai-page .kernel-details th:nth-child(3),.kleidiai-page .kernel-details th:nth-child(4){width:19%!important}.kleidiai-page .kernel-details th:nth-child(5){width:41%!important}.kleidiai-page .kernel-details th,.kleidiai-page .kernel-details td{overflow-wrap:anywhere;word-break:break-word}.kleidiai-page .datatype{display:inline-block;padding:.08rem .28rem;border-radius:.25rem}.kleidiai-page .type-output{color:#1e5fb4;background:#edf5ff}.kleidiai-page .type-lhs{color:#13795b;background:#edf9f5}.kleidiai-page .type-rhs{color:#6f42c1;background:#f5f0ff}.kleidiai-page .extension-badge{color:#1e5fb4;background:#edf5ff}.kleidiai-page .status-partial{color:#8a5b00;background:#fff8e6}.kleidiai-page .kernel-link{color:#1e5fb4}.kleidiai-page .pack-links{display:flex;flex-wrap:wrap;gap:.3rem}.kleidiai-page .lhs-pack{color:#13795b}.kleidiai-page .rhs-pack{color:#6f42c1}.kleidiai-page .operator-links{display:flex;flex-wrap:wrap;align-items:center;gap:.25rem;margin-top:.2rem;font-size:.65rem}.kleidiai-page .operator-links span{color:var(--kai-muted)}.kleidiai-page .operator-links a{padding:.05rem .3rem;border-radius:999px;background:#edf5ff}@media(max-width:700px){.kleidiai-page .table-wrap{overflow-x:auto}.kleidiai-page .main-table{min-width:46rem!important}}
</style>
__RUNTIME__
'''


def main() -> None:
    records, metadata = collect()
    payload = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    output = (
        TEMPLATE.replace("__BASE_CSS__", BASE_CSS)
        .replace("__DATA__", payload)
        .replace("__KAI_URL__", f"{report.KAI_GITHUB}/commit/{metadata['kai_revision']}")
        .replace("__KAI_DESCRIBE__", metadata["kai_describe"])
        .replace("__ORT_URL__", f"{report.ORT_GITHUB}/commit/{metadata['ort_revision']}")
        .replace("__ORT_SHORT__", metadata["ort_short"])
        .replace("__AUDITED_AT__", metadata["audited_at"])
    )
    runtime = output[output.index('<script type="application/json" id="kernelData">') : output.rindex("</script>") + 9]
    pages_output = (
        PAGES_TEMPLATE.replace("__RUNTIME__", runtime)
        .replace("__KAI_URL__", f"{report.KAI_GITHUB}/commit/{metadata['kai_revision']}")
        .replace("__KAI_DESCRIBE__", metadata["kai_describe"])
        .replace("__ORT_URL__", f"{report.ORT_GITHUB}/commit/{metadata['ort_revision']}")
        .replace("__ORT_SHORT__", metadata["ort_short"])
        .replace("__AUDITED_AT__", metadata["audited_at"])
    )
    for destination, contents in ((OUTPUT, output), (PAGES_OUTPUT, pages_output)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
        print(f"Generated {destination}")
    print({"microkernels": len(records), "kleidiai": metadata["kai_describe"]})


if __name__ == "__main__":
    main()
