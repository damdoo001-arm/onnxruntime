#!/usr/bin/env python3
"""Generate alternative KleidiAI kernel-discovery layout prototypes."""

from __future__ import annotations

import json
from pathlib import Path

import generate_kernel_status as report


ROOT = Path(__file__).resolve().parent
FRAMEWORKS = (
    ("xnn", "XNNPACK"),
    ("ort", "ONNX Runtime"),
    ("mnn", "MNN"),
    ("llama", "llama.cpp"),
)


def operand_types(variant: report.Variant) -> tuple[str, str, str]:
    descriptors = variant.descriptors
    output = descriptors[0].compact if descriptors else "—"
    operands = descriptors[1:]
    if variant.operation_key == "lhs-pack":
        lhs, rhs = operands, []
    elif variant.operation_key in {"rhs-pack-kxn", "rhs-pack-nxk", "dwconv-rhs-pack"}:
        lhs, rhs = [], operands
    else:
        lhs, rhs = operands[:1], operands[1:]
    join = lambda values: " · ".join(value.compact for value in values) or "—"
    return output, join(lhs), join(rhs)


def collect_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    variants = report.inventory()
    xnn_pin, xnn_legacy_pin, ort_pin, mnn_pin, llama_pin = report.discover_pins()
    report.scan_framework(variants, report.XNN_ROOT, [report.XNN_ROOT / "src"], "xnn")
    report.scan_framework(
        variants,
        report.ORT_ROOT,
        [report.ORT_ROOT / "onnxruntime" / "core" / "mlas" / "lib"],
        "ort",
    )
    report.scan_framework(
        variants,
        report.MNN_ROOT,
        [report.MNN_ROOT / "source" / "backend" / "cpu" / "kleidiai"],
        "mnn",
    )
    report.scan_framework(
        variants,
        report.LLAMA_ROOT,
        [report.LLAMA_ROOT / "ggml" / "src" / "ggml-cpu" / "kleidiai"],
        "llama",
    )
    report.apply_availability(
        variants, xnn_pin, xnn_legacy_pin, ort_pin, mnn_pin, llama_pin
    )
    revision = report.run_git(report.KAI_ROOT, "rev-parse", "HEAD")
    records = []
    for variant in variants:
        output, lhs, rhs = operand_types(variant)
        quantization, _quant_key = report.quantization_class(
            report.Group(
                key=variant.group_key,
                operation=variant.operation,
                operation_key=variant.operation_key,
                family=variant.canonical_family,
                descriptors=variant.descriptors,
                variants=[variant],
            )
        )
        framework_status = {}
        used_by = []
        for attr, label in FRAMEWORKS:
            status_label, status_key = report.variant_framework_status(variant, attr)
            framework_status[attr] = {"label": status_label, "key": status_key}
            if getattr(variant, attr).active:
                used_by.append(label)
        relative_folder = str(Path(variant.relative_c).parent).removeprefix("kai/ukernels/")
        records.append(
            {
                "name": variant.raw_name,
                "family": variant.canonical_family,
                "operation": variant.operation,
                "operationKey": variant.operation_key,
                "output": output,
                "lhs": lhs,
                "rhs": rhs,
                "quantization": quantization,
                "isa": variant.isa,
                "instruction": variant.feature_instruction,
                "tile": variant.tile,
                "folder": relative_folder,
                "source": f"{report.KAI_GITHUB}/blob/{revision}/{variant.relative_c}",
                "frameworks": used_by,
                "frameworkStatus": framework_status,
            }
        )
    metadata = {
        "revision": revision,
        "describe": report.describe_revision(report.KAI_ROOT),
        "count": len(records),
    }
    return records, metadata


BASE_CSS = r"""
:root{color-scheme:dark;--bg:#09111f;--surface:#101b2d;--surface2:#15243a;--surface3:#1a2c46;--text:#e7eef9;--muted:#9fb0c8;--line:#29405f;--accent:#58a6ff;--green:#6ce8bb;--amber:#ffd17d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}button,input{font:inherit}button{color:inherit}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}code{font-family:"SFMono-Regular",Consolas,monospace}.shell{width:min(1600px,calc(100% - 28px));margin:auto;padding:24px 0 56px}.hero{padding:24px 26px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(125deg,rgba(41,103,168,.25),rgba(88,43,131,.16)),var(--surface)}h1{margin:0 0 5px;font-size:clamp(26px,4vw,40px);line-height:1.1}.lede,.meta,.muted{color:var(--muted)}.meta{margin-top:12px;font-size:12px}.search{width:100%;border:1px solid var(--line);border-radius:9px;background:var(--surface2);color:var(--text);padding:10px 12px;outline:none}.search:focus{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}.count{font-variant-numeric:tabular-nums}.badge{display:inline-flex;align-items:center;margin:2px 3px 2px 0;padding:2px 7px;border-radius:999px;background:var(--surface3);color:var(--muted);font-size:10px;font-weight:700;white-space:nowrap}.badge.used{color:var(--green);background:rgba(35,166,122,.15)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--surface)}table{width:100%;border-collapse:collapse}th,td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{position:sticky;top:0;z-index:2;background:var(--surface3);color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}tbody tr:hover td{background:color-mix(in srgb,var(--surface3) 56%,transparent)}td code{overflow-wrap:anywhere}.type-stack{min-width:190px}.type-stack span{display:block}.type-stack small{color:var(--muted)}.empty{padding:36px;text-align:center;color:var(--muted)}
"""


def page(template: str, records: list[dict[str, object]], metadata: dict[str, object]) -> str:
    payload = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    return (
        template.replace("__BASE_CSS__", BASE_CSS)
        .replace("__DATA__", payload)
        .replace("__REVISION__", str(metadata["revision"]))
        .replace("__DESCRIBE__", str(metadata["describe"]))
        .replace("__COUNT__", str(metadata["count"]))
    )


FACETED_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KleidiAI needs-first search</title><style>__BASE_CSS__
.workspace{display:grid;grid-template-columns:280px minmax(0,1fr);gap:14px;margin-top:14px}.facets{align-self:start;position:sticky;top:10px;max-height:calc(100vh - 20px);overflow:auto;padding:15px;border:1px solid var(--line);border-radius:13px;background:var(--surface)}.facets h2{margin:0 0 12px;font-size:16px}.facet{margin:0;padding:12px 0;border:0;border-top:1px solid var(--line)}.facet legend{padding:0 0 7px;color:var(--muted);font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.option{display:grid;grid-template-columns:16px minmax(0,1fr) auto;align-items:start;gap:7px;padding:4px 0;cursor:pointer}.option input{margin:3px 0 0;accent-color:var(--accent)}.option .count{color:var(--muted);font-size:11px}.results-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 10px}.results-head h2{margin:0;font-size:17px}.active{display:flex;flex-wrap:wrap;gap:5px;margin:10px 0}.filter-chip{border:1px solid var(--line);border-radius:999px;background:var(--surface2);padding:4px 8px;cursor:pointer;font-size:11px}.clear{border:0;background:transparent;color:var(--accent);padding:5px;cursor:pointer}.main-table{min-width:1030px}@media(max-width:850px){.workspace{grid-template-columns:1fr}.facets{position:static;max-height:none}.facet-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 16px}}@media(max-width:520px){.facet-grid{grid-template-columns:1fr}.shell{width:min(100% - 16px,1600px);padding-top:8px}}
</style></head><body><main class="shell"><header class="hero"><h1>Find a kernel by requirement</h1><p class="lede">Choose operation, data types, quantization, ISA, or a framework that already uses the kernel. Facets remain visible and combine across categories.</p><div class="meta"><code>__DESCRIBE__</code> · <span>__COUNT__ public micro-kernels</span></div></header>
<div class="workspace"><aside class="facets" aria-label="Kernel filters"><h2>Requirements</h2><input id="search" class="search" type="search" placeholder="Search symbol, family, instruction…"><div id="facetRoot" class="facet-grid"></div><div id="activeFilters" class="active"></div><button id="clear" class="clear" type="button">Clear all filters</button></aside>
<section><div class="results-head"><h2>Matching kernels</h2><span id="resultCount" class="muted count" aria-live="polite"></span></div><div class="table-wrap"><table class="main-table"><thead><tr><th>Micro-kernel</th><th>Operation</th><th>Output / inputs</th><th>ISA</th><th>Used by</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No kernels match these requirements.</div></div></section></div></main>
<script type="application/json" id="kernelData">__DATA__</script><script>
(()=>{const data=JSON.parse(document.getElementById('kernelData').textContent);const defs=[['operation','Operation'],['output','Output'],['lhs','Input / LHS'],['rhs','Weights / RHS'],['quantization','Quantization'],['isa','ISA'],['frameworks','Used by framework']];const selected=Object.fromEntries(defs.map(([key])=>[key,new Set()]));const facetRoot=document.getElementById('facetRoot'),results=document.getElementById('results'),search=document.getElementById('search');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const values=(key)=>[...new Set(data.flatMap(r=>key==='frameworks'?r[key]:[r[key]]).filter(v=>v&&v!=='—'))].sort((a,b)=>a.localeCompare(b));
function matches(r,omit=''){const q=search.value.trim().toLowerCase();if(q&&!`${r.name} ${r.family} ${r.operation} ${r.output} ${r.lhs} ${r.rhs} ${r.quantization} ${r.isa} ${r.instruction} ${r.folder}`.toLowerCase().includes(q))return false;return defs.every(([key])=>{if(key===omit||!selected[key].size)return true;const actual=key==='frameworks'?r[key]:[r[key]];return actual.some(v=>selected[key].has(v))})}
function renderFacets(){facetRoot.innerHTML=defs.map(([key,label])=>`<fieldset class="facet"><legend>${esc(label)}</legend>${values(key).map(value=>{const count=data.filter(r=>matches(r,key)&&(key==='frameworks'?r[key].includes(value):r[key]===value)).length;return `<label class="option"><input type="checkbox" data-key="${key}" value="${esc(value)}" ${selected[key].has(value)?'checked':''}><span>${esc(value)}</span><span class="count">${count}</span></label>`}).join('')}</fieldset>`).join('');facetRoot.querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{input.checked?selected[input.dataset.key].add(input.value):selected[input.dataset.key].delete(input.value);render()}))}
function renderActive(){const entries=defs.flatMap(([key,label])=>[...selected[key]].map(value=>({key,label,value})));document.getElementById('activeFilters').innerHTML=entries.map(x=>`<button class="filter-chip" type="button" data-key="${x.key}" data-value="${esc(x.value)}">${esc(x.label)}: ${esc(x.value)} ×</button>`).join('');document.querySelectorAll('.filter-chip').forEach(button=>button.addEventListener('click',()=>{selected[button.dataset.key].delete(button.dataset.value);render()}))}
function renderResults(){const shown=data.filter(r=>matches(r)).sort((a,b)=>a.operation.localeCompare(b.operation)||a.name.localeCompare(b.name));document.getElementById('resultCount').textContent=`${shown.length} of ${data.length}`;document.getElementById('empty').hidden=shown.length>0;results.innerHTML=shown.map(r=>`<tr><td><a href="${esc(r.source)}" target="_blank" rel="noopener noreferrer"><code>${esc(r.name)}</code></a><div class="muted"><small>${esc(r.folder)}</small></div></td><td>${esc(r.operation)}</td><td class="type-stack"><span>${esc(r.output)} <small>output</small></span><span>${esc(r.lhs)} <small>LHS</small></span><span>${esc(r.rhs)} <small>RHS</small></span></td><td><span class="badge">${esc(r.isa)}</span>${r.instruction!=='—'?`<span class="badge">${esc(r.instruction)}</span>`:''}</td><td>${r.frameworks.length?r.frameworks.map(x=>`<span class="badge used">${esc(x)}</span>`).join(''):'<span class="muted">—</span>'}</td></tr>`).join('')}
function render(){renderResults();renderActive();renderFacets()}search.addEventListener('input',render);document.getElementById('clear').addEventListener('click',()=>{Object.values(selected).forEach(set=>set.clear());search.value='';render()});render()})();
</script></body></html>'''


RADIO_DROPDOWN_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KleidiAI radio-dropdown search</title><style>__BASE_CSS__
.controls{position:sticky;top:0;z-index:10;margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:13px;background:color-mix(in srgb,var(--surface) 94%,transparent);backdrop-filter:blur(14px)}.search-row{display:grid;grid-template-columns:minmax(240px,1fr) auto;gap:9px}.clear{border:1px solid var(--line);border-radius:9px;background:var(--surface3);padding:8px 13px;cursor:pointer}.dropdown-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:9px}.filter-menu{position:relative;min-width:0}.filter-menu summary{display:flex;min-height:52px;flex-direction:column;justify-content:center;gap:2px;padding:7px 10px;border:1px solid var(--line);border-radius:9px;background:var(--surface2);cursor:pointer;list-style:none}.filter-menu summary::-webkit-details-marker{display:none}.filter-menu summary:after{content:"▾";position:absolute;right:10px;top:17px;color:var(--muted)}.filter-menu[open] summary{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}.filter-menu[open] summary:after{transform:rotate(180deg)}.filter-name{color:var(--muted);font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.filter-value{max-width:calc(100% - 20px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.menu{position:absolute;z-index:30;top:calc(100% + 5px);left:0;width:max(100%,290px);max-height:340px;overflow:auto;padding:7px;border:1px solid var(--line);border-radius:10px;background:var(--surface2);box-shadow:0 18px 45px rgba(0,0,0,.32)}.filter-menu:nth-child(4n) .menu{right:0;left:auto}.radio-option{display:grid;grid-template-columns:16px minmax(0,1fr) auto;align-items:start;gap:8px;padding:7px;border-radius:7px;cursor:pointer}.radio-option:hover{background:var(--surface3)}.radio-option input{margin:3px 0 0;accent-color:var(--accent)}.radio-option .count{color:var(--muted);font-size:11px}.active{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}.filter-chip{border:1px solid var(--line);border-radius:999px;background:var(--surface2);padding:4px 8px;cursor:pointer;font-size:11px}.results-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 10px}.results-head h2{margin:0;font-size:17px}.main-table{min-width:1030px}@media(max-width:900px){.dropdown-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.filter-menu:nth-child(4n) .menu{right:auto;left:0}.filter-menu:nth-child(2n) .menu{right:0;left:auto}}@media(max-width:520px){.shell{width:min(100% - 16px,1600px);padding-top:8px}.controls{position:static}.search-row,.dropdown-grid{grid-template-columns:1fr}.filter-menu:nth-child(2n) .menu{right:auto;left:0}.menu{width:100%}}
</style></head><body><main class="shell"><header class="hero"><h1>Find a kernel by requirement</h1><p class="lede">Select one value per requirement from compact radio dropdowns. Choose “Any” to leave a requirement unrestricted.</p><div class="meta"><code>__DESCRIBE__</code> · <span>__COUNT__ public micro-kernels</span></div></header>
<section class="controls" aria-label="Kernel filters"><div class="search-row"><input id="search" class="search" type="search" placeholder="Search symbol, family, instruction…"><button id="clear" class="clear" type="button">Clear filters</button></div><div id="dropdowns" class="dropdown-grid"></div><div id="activeFilters" class="active"></div></section>
<section><div class="results-head"><h2>Matching kernels</h2><span id="resultCount" class="muted count" aria-live="polite"></span></div><div class="table-wrap"><table class="main-table"><thead><tr><th>Micro-kernel</th><th>Operation</th><th>Output / inputs</th><th>ISA</th><th>Used by</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No kernels match these requirements.</div></div></section></main>
<script type="application/json" id="kernelData">__DATA__</script><script>
(()=>{const data=JSON.parse(document.getElementById('kernelData').textContent);const defs=[['operation','Operation'],['output','Output'],['lhs','Input / LHS'],['rhs','Weights / RHS'],['quantization','Quantization'],['isa','ISA'],['frameworks','Used by framework']];const selected=Object.fromEntries(defs.map(([key])=>[key,'']));const dropdowns=document.getElementById('dropdowns'),results=document.getElementById('results'),search=document.getElementById('search');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const values=key=>[...new Set(data.flatMap(r=>key==='frameworks'?r[key]:[r[key]]).filter(v=>v&&v!=='—'))].sort((a,b)=>a.localeCompare(b));
function matches(r,omit=''){const q=search.value.trim().toLowerCase();if(q&&!`${r.name} ${r.family} ${r.operation} ${r.output} ${r.lhs} ${r.rhs} ${r.quantization} ${r.isa} ${r.instruction} ${r.folder}`.toLowerCase().includes(q))return false;return defs.every(([key])=>key===omit||!selected[key]||(key==='frameworks'?r[key].includes(selected[key]):r[key]===selected[key]))}
function optionCount(key,value){return data.filter(r=>matches(r,key)&&(!value||(key==='frameworks'?r[key].includes(value):r[key]===value))).length}
function renderDropdowns(){dropdowns.innerHTML=defs.map(([key,label])=>`<details class="filter-menu"><summary><span class="filter-name">${esc(label)}</span><strong class="filter-value">${esc(selected[key]||'Any')}</strong></summary><div class="menu" role="radiogroup" aria-label="${esc(label)}"><label class="radio-option"><input type="radio" name="radio-${key}" data-key="${key}" value="" ${selected[key]===''?'checked':''}><span>Any</span><span class="count">${optionCount(key,'')}</span></label>${values(key).map(value=>`<label class="radio-option"><input type="radio" name="radio-${key}" data-key="${key}" value="${esc(value)}" ${selected[key]===value?'checked':''}><span>${esc(value)}</span><span class="count">${optionCount(key,value)}</span></label>`).join('')}</div></details>`).join('');dropdowns.querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{selected[input.dataset.key]=input.value;input.closest('details').open=false;render()}));dropdowns.querySelectorAll('details').forEach(detail=>detail.addEventListener('toggle',()=>{if(detail.open)dropdowns.querySelectorAll('details').forEach(other=>{if(other!==detail)other.open=false})}))}
function renderActive(){const entries=defs.filter(([key])=>selected[key]).map(([key,label])=>({key,label,value:selected[key]}));document.getElementById('activeFilters').innerHTML=entries.map(x=>`<button class="filter-chip" type="button" data-key="${x.key}">${esc(x.label)}: ${esc(x.value)} ×</button>`).join('');document.querySelectorAll('.filter-chip').forEach(button=>button.addEventListener('click',()=>{selected[button.dataset.key]='';render()}))}
function renderResults(){const shown=data.filter(r=>matches(r)).sort((a,b)=>a.operation.localeCompare(b.operation)||a.name.localeCompare(b.name));document.getElementById('resultCount').textContent=`${shown.length} of ${data.length}`;document.getElementById('empty').hidden=shown.length>0;results.innerHTML=shown.map(r=>`<tr><td><a href="${esc(r.source)}" target="_blank" rel="noopener noreferrer"><code>${esc(r.name)}</code></a><div class="muted"><small>${esc(r.folder)}</small></div></td><td>${esc(r.operation)}</td><td class="type-stack"><span>${esc(r.output)} <small>output</small></span><span>${esc(r.lhs)} <small>LHS</small></span><span>${esc(r.rhs)} <small>RHS</small></span></td><td><span class="badge">${esc(r.isa)}</span>${r.instruction!=='—'?`<span class="badge">${esc(r.instruction)}</span>`:''}</td><td>${r.frameworks.length?r.frameworks.map(x=>`<span class="badge used">${esc(x)}</span>`).join(''):'<span class="muted">—</span>'}</td></tr>`).join('')}
function render(){renderResults();renderActive();renderDropdowns()}search.addEventListener('input',render);document.getElementById('clear').addEventListener('click',()=>{Object.keys(selected).forEach(key=>selected[key]='');search.value='';render()});document.addEventListener('click',event=>{if(!event.target.closest('.filter-menu'))dropdowns.querySelectorAll('details').forEach(detail=>detail.open=false)});document.addEventListener('keydown',event=>{if(event.key==='Escape'){const open=[...dropdowns.querySelectorAll('details')].find(detail=>detail.open);if(open){open.open=false;open.querySelector('summary').focus()}}});render()})();
</script></body></html>'''


FOLDER_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KleidiAI source-tree browser</title><style>__BASE_CSS__
.workspace{display:grid;grid-template-columns:330px minmax(0,1fr);gap:14px;margin-top:14px}.browser{align-self:start;position:sticky;top:10px;max-height:calc(100vh - 20px);overflow:auto;padding:15px;border:1px solid var(--line);border-radius:13px;background:var(--surface)}.browser h2,.results h2{margin:0;font-size:17px}.browser .search{margin:12px 0}.tree details{margin-left:10px}.tree>details{margin-left:0}.tree summary{padding:5px 3px;cursor:pointer;color:var(--muted);font-weight:700}.tree button{display:flex;width:100%;align-items:center;justify-content:space-between;gap:8px;border:0;border-radius:7px;background:transparent;padding:6px 8px;text-align:left;cursor:pointer}.tree button:hover,.tree button.active{background:var(--surface3);color:var(--accent)}.tree .count{color:var(--muted);font-size:11px}.results-head{display:flex;align-items:end;justify-content:space-between;gap:12px;margin-bottom:10px}.crumbs{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin-top:5px}.crumbs button{border:0;background:transparent;color:var(--accent);padding:0;cursor:pointer}.framework-bar{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 10px;padding:10px 12px;border:1px solid var(--line);border-radius:11px;background:var(--surface)}.framework-bar label{display:flex;align-items:center;gap:5px;color:var(--muted);cursor:pointer}.framework-bar input{accent-color:var(--accent)}.main-table{min-width:900px}@media(max-width:850px){.workspace{grid-template-columns:1fr}.browser{position:static;max-height:440px}}@media(max-width:520px){.shell{width:min(100% - 16px,1600px);padding-top:8px}}
</style></head><body><main class="shell"><header class="hero"><h1>Browse the KleidiAI source tree</h1><p class="lede">Navigate the repository hierarchy first, then narrow the selected folder by symbol or downstream framework usage.</p><div class="meta"><code>__DESCRIBE__</code> · <span>__COUNT__ public micro-kernels</span></div></header>
<div class="workspace"><aside class="browser"><h2><code>kai/ukernels</code></h2><input id="search" class="search" type="search" placeholder="Search within selected folder…"><div id="tree" class="tree"></div></aside><section class="results"><div class="results-head"><div><h2>Folder contents</h2><nav id="crumbs" class="crumbs" aria-label="Selected folder"></nav></div><span id="resultCount" class="muted count" aria-live="polite"></span></div><div id="frameworkBar" class="framework-bar" aria-label="Framework filters"></div><div class="table-wrap"><table class="main-table"><thead><tr><th>Micro-kernel</th><th>Family</th><th>Types</th><th>ISA / tile</th><th>Used by</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No kernels match in this folder.</div></div></section></div></main>
<script type="application/json" id="kernelData">__DATA__</script><script>
(()=>{const data=JSON.parse(document.getElementById('kernelData').textContent),treeRoot=document.getElementById('tree'),results=document.getElementById('results'),search=document.getElementById('search');let selectedPath='';const selectedFrameworks=new Set();const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const node={name:'ukernels',path:'',children:new Map(),count:data.length};for(const record of data){let current=node,path=[];for(const part of record.folder.split('/')){path.push(part);if(!current.children.has(part))current.children.set(part,{name:part,path:path.join('/'),children:new Map(),count:0});current=current.children.get(part);current.count++}}
function treeHtml(current,depth=0){return [...current.children.values()].sort((a,b)=>a.name.localeCompare(b.name)).map(child=>child.children.size?`<details ${depth<1?'open':''}><summary>${esc(child.name)} <span class="count">${child.count}</span></summary><button type="button" data-path="${esc(child.path)}" class="${selectedPath===child.path?'active':''}"><span>All in this folder</span><span class="count">${child.count}</span></button>${treeHtml(child,depth+1)}</details>`:`<button type="button" data-path="${esc(child.path)}" class="${selectedPath===child.path?'active':''}"><code>${esc(child.name)}</code><span class="count">${child.count}</span></button>`).join('')}
function renderTree(){treeRoot.innerHTML=`<button type="button" data-path="" class="${selectedPath===''?'active':''}"><span>All kernels</span><span class="count">${data.length}</span></button>${treeHtml(node)}`;treeRoot.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{selectedPath=button.dataset.path;render()}))}
function renderCrumbs(){const parts=selectedPath?selectedPath.split('/'):[];let path='';document.getElementById('crumbs').innerHTML=`<button type="button" data-path="">ukernels</button>${parts.map(part=>{path=path?`${path}/${part}`:part;return `<span>/</span><button type="button" data-path="${esc(path)}">${esc(part)}</button>`}).join('')}`;document.querySelectorAll('#crumbs button').forEach(button=>button.addEventListener('click',()=>{selectedPath=button.dataset.path;render()}))}
function renderFrameworks(){const names=['XNNPACK','ONNX Runtime','MNN','llama.cpp'];document.getElementById('frameworkBar').innerHTML=`<span class="muted">Used by:</span>${names.map((name,i)=>`<label><input type="checkbox" value="${esc(name)}" ${selectedFrameworks.has(name)?'checked':''}>${esc(name)}</label>`).join('')}`;document.querySelectorAll('#frameworkBar input').forEach(input=>input.addEventListener('change',()=>{input.checked?selectedFrameworks.add(input.value):selectedFrameworks.delete(input.value);renderResults()}))}
function renderResults(){const q=search.value.trim().toLowerCase();const shown=data.filter(r=>(!selectedPath||(r.folder===selectedPath||r.folder.startsWith(`${selectedPath}/`)))&&(!q||`${r.name} ${r.family} ${r.operation} ${r.output} ${r.lhs} ${r.rhs} ${r.isa} ${r.instruction}`.toLowerCase().includes(q))&&(!selectedFrameworks.size||r.frameworks.some(x=>selectedFrameworks.has(x)))).sort((a,b)=>a.folder.localeCompare(b.folder)||a.name.localeCompare(b.name));document.getElementById('resultCount').textContent=`${shown.length} kernels`;document.getElementById('empty').hidden=shown.length>0;results.innerHTML=shown.map(r=>`<tr><td><a href="${esc(r.source)}" target="_blank" rel="noopener noreferrer"><code>${esc(r.name)}</code></a><div class="muted"><small>${esc(r.folder)}</small></div></td><td><code>${esc(r.family)}</code><div class="muted"><small>${esc(r.operation)}</small></div></td><td class="type-stack"><span>${esc(r.output)} <small>output</small></span><span>${esc(r.lhs)} <small>LHS</small></span><span>${esc(r.rhs)} <small>RHS</small></span></td><td><span class="badge">${esc(r.isa)}</span><span class="badge">${esc(r.tile)}</span></td><td>${r.frameworks.length?r.frameworks.map(x=>`<span class="badge used">${esc(x)}</span>`).join(''):'<span class="muted">—</span>'}</td></tr>`).join('')}
function render(){renderTree();renderCrumbs();renderFrameworks();renderResults()}search.addEventListener('input',renderResults);render()})();
</script></body></html>'''


def main() -> None:
    records, metadata = collect_records()
    outputs = {
        ROOT / "kleidiai_faceted_search.html": FACETED_TEMPLATE,
        ROOT / "kleidiai_radio_dropdown_search.html": RADIO_DROPDOWN_TEMPLATE,
        ROOT / "kleidiai_folder_browser.html": FOLDER_TEMPLATE,
    }
    for path, template in outputs.items():
        path.write_text(page(template, records, metadata), encoding="utf-8")
        print(f"Generated {path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
