#!/usr/bin/env python3
"""Generate a compact, checkbox-filterable ONNX Runtime compatibility page."""

from __future__ import annotations

import json
import os
from pathlib import Path

import generate_kernel_status as report
from generate_layout_prototypes import BASE_CSS, operand_types
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


def collect() -> tuple[list[dict[str, object]], dict[str, str]]:
    variants = report.inventory()
    ort_pin = report.discover_ort_pin()
    report.scan_framework(
        variants,
        report.ORT_ROOT,
        [report.ORT_ROOT / "onnxruntime" / "core" / "mlas" / "lib"],
        "ort",
    )
    audited_at = report.apply_pr_candidates(variants)
    kai_revision = report.run_git(report.KAI_ROOT, "rev-parse", "HEAD")
    ort_revision = report.run_git(report.ORT_ROOT, "rev-parse", "HEAD")
    records = []
    for variant in variants:
        variant.ort.available_primary = variant.relative_c in ort_pin.files
        output, lhs, rhs = operand_types(variant)
        status_key, status_label = integration_status(variant)
        evidence = []
        for item in (variant.ort.active or variant.ort.references)[:2]:
            evidence.append(
                {
                    "label": f"code:{item.line}",
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
                "isa": variant.isa,
                "tile": variant.tile,
                "status": status_key,
                "statusLabel": status_label.removeprefix("✅ ").removeprefix("🔗 ").removeprefix("🟣 ").removeprefix("— "),
                "availability": "available" if variant.ort.available_primary else "unavailable",
                "evidence": evidence,
            }
        )
    metadata = {
        "kai_revision": kai_revision,
        "kai_describe": report.describe_revision(report.KAI_ROOT),
        "ort_revision": ort_revision,
        "ort_short": report.short_revision(report.ORT_ROOT),
        "ort_pin": ort_pin.label,
        "audited_at": audited_at,
    }
    return records, metadata


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="KleidiAI compatibility in ONNX Runtime"><title>KleidiAI compatibility in ONNX Runtime</title><style>__BASE_CSS__
.controls{position:sticky;z-index:10;top:0;margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:13px;background:color-mix(in srgb,var(--surface) 94%,transparent);backdrop-filter:blur(14px)}.search-row{display:grid;grid-template-columns:minmax(240px,1fr) auto;gap:9px}.clear{border:1px solid var(--line);border-radius:9px;background:var(--surface3);padding:8px 13px;cursor:pointer}.dropdown-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:9px}.filter-menu{position:relative;min-width:0}.filter-menu summary{display:flex;min-height:50px;flex-direction:column;justify-content:center;gap:2px;padding:7px 10px;border:1px solid var(--line);border-radius:9px;background:var(--surface2);cursor:pointer;list-style:none}.filter-menu summary::-webkit-details-marker{display:none}.filter-menu summary:after{content:"▾";position:absolute;right:10px;top:16px;color:var(--muted)}.filter-menu[open] summary{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent)}.filter-menu[open] summary:after{transform:rotate(180deg)}.filter-name{color:var(--muted);font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.filter-value{max-width:calc(100% - 20px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.checklist{position:absolute;z-index:30;top:calc(100% + 5px);left:0;width:max(100%,280px);max-height:330px;overflow:auto;padding:7px;border:1px solid var(--line);border-radius:10px;background:var(--surface2);box-shadow:0 18px 45px rgba(0,0,0,.32)}.filter-menu:nth-child(4n) .checklist{right:0;left:auto}.check-option{display:grid;grid-template-columns:16px minmax(0,1fr) auto;align-items:start;gap:8px;padding:7px;border-radius:7px;cursor:pointer}.check-option:hover{background:var(--surface3)}.check-option input{margin:3px 0 0;accent-color:var(--accent)}.check-option .count{color:var(--muted);font-size:11px}.active{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}.filter-chip{border:1px solid var(--line);border-radius:999px;background:var(--surface2);padding:4px 8px;cursor:pointer;font-size:11px}.results-head{display:flex;align-items:end;justify-content:space-between;gap:12px;margin:0 0 10px}.results-head h2{margin:0;font-size:17px}.main-table{min-width:760px}.main-table td:first-child,.main-table th:first-child{text-align:center}.signature{overflow-wrap:anywhere}.status{font-weight:700}.status-integrated{color:var(--green)}.status-referenced{color:#8ecbff}.status-pr{color:#c4b5fd}.status-not-integrated{color:var(--muted)}.links{display:flex;flex-wrap:wrap;gap:6px;margin-top:3px}.pin-yes{color:var(--green)}.pin-no{color:var(--muted)}@media(max-width:900px){.dropdown-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.filter-menu:nth-child(4n) .checklist{right:auto;left:0}.filter-menu:nth-child(2n) .checklist{right:0;left:auto}}@media(max-width:520px){.shell{width:min(100% - 16px,1600px);padding-top:8px}.controls{position:static}.search-row,.dropdown-grid{grid-template-columns:1fr}.filter-menu:nth-child(2n) .checklist{right:auto;left:0}.checklist{width:100%}}
</style></head><body><main class="shell"><header class="hero"><h1>KleidiAI compatibility in ONNX Runtime</h1><p class="lede">A focused compatibility view with compact source links and multi-select requirement filters.</p><div class="meta"><span>KleidiAI <a href="__KAI_URL__"><code>__KAI_DESCRIBE__</code></a></span> · <span>ONNX Runtime <a href="__ORT_URL__"><code>__ORT_SHORT__</code></a></span> · <span>pin <code>__ORT_PIN__</code></span> · <span>PR audit __AUDITED_AT__</span></div></header>
<section class="controls" aria-label="Kernel filters"><div class="search-row"><input id="search" class="search" type="search" placeholder="Search full kernel symbol…"><button id="clear" class="clear" type="button">Clear filters</button></div><div id="dropdowns" class="dropdown-grid"></div><div id="activeFilters" class="active"></div></section><section class="results"><div class="results-head"><h2>Matching kernels</h2><span id="resultCount" class="muted count" aria-live="polite"></span></div><div class="table-wrap"><table class="main-table"><thead><tr><th>Kernel</th><th>Type signature</th><th>ISA / tile</th><th>At pin</th><th>ONNX Runtime</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No kernels match these filters.</div></div></section></main>
<script type="application/json" id="kernelData">__DATA__</script><script>
(()=>{const data=JSON.parse(document.getElementById('kernelData').textContent);const defs=[['operation','Operation'],['output','Output'],['lhs','Input / LHS'],['rhs','Weights / RHS'],['isa','ISA'],['availability','Pin availability'],['status','ORT status']];const labels={available:'Available',unavailable:'Unavailable',integrated:'Integrated',referenced:'Referenced only',pr:'PR available','not-integrated':'Not integrated'};const selected=Object.fromEntries(defs.map(([key])=>[key,new Set()]));const dropdowns=document.getElementById('dropdowns'),search=document.getElementById('search'),body=document.getElementById('results');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const values=key=>[...new Set(data.map(r=>r[key]).filter(value=>value&&value!=='—'))].sort((a,b)=>String(labels[a]||a).localeCompare(String(labels[b]||b)));
function matches(record,omit=''){const query=search.value.trim().toLowerCase();if(query&&!`${record.name} ${record.operation} ${record.output} ${record.lhs} ${record.rhs} ${record.signature} ${record.isa} ${record.tile}`.toLowerCase().includes(query))return false;return defs.every(([key])=>key===omit||!selected[key].size||selected[key].has(record[key]))}
function renderDropdowns(openKey=''){dropdowns.innerHTML=defs.map(([key,label])=>{const chosen=[...selected[key]].map(value=>labels[value]||value);const summary=chosen.length===0?'All':chosen.length<=2?chosen.join(', '):`${chosen.length} selected`;return `<details class="filter-menu" data-key="${key}" ${openKey===key?'open':''}><summary><span class="filter-name">${esc(label)}</span><strong class="filter-value" title="${esc(chosen.join(', '))}">${esc(summary)}</strong></summary><div class="checklist" role="group" aria-label="${esc(label)}">${values(key).map(value=>{const count=data.filter(record=>matches(record,key)&&record[key]===value).length;return `<label class="check-option"><input type="checkbox" data-key="${key}" value="${esc(value)}" ${selected[key].has(value)?'checked':''}><span>${esc(labels[value]||value)}</span><span class="count">${count}</span></label>`}).join('')}</div></details>`}).join('');dropdowns.querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{input.checked?selected[input.dataset.key].add(input.value):selected[input.dataset.key].delete(input.value);render(input.dataset.key)}));dropdowns.querySelectorAll('details').forEach(detail=>detail.addEventListener('toggle',()=>{if(detail.open)dropdowns.querySelectorAll('details').forEach(other=>{if(other!==detail)other.open=false})}))}
function renderActive(){const entries=defs.flatMap(([key,label])=>[...selected[key]].map(value=>({key,label,value})));document.getElementById('activeFilters').innerHTML=entries.map(item=>`<button class="filter-chip" type="button" data-key="${item.key}" data-value="${esc(item.value)}">${esc(item.label)}: ${esc(labels[item.value]||item.value)} ×</button>`).join('');document.querySelectorAll('.filter-chip').forEach(button=>button.addEventListener('click',()=>{selected[button.dataset.key].delete(button.dataset.value);render()}))}
function renderResults(){const shown=data.filter(record=>matches(record)).sort((a,b)=>a.operation.localeCompare(b.operation)||a.name.localeCompare(b.name));document.getElementById('resultCount').textContent=`${shown.length} of ${data.length}`;document.getElementById('empty').hidden=shown.length>0;body.innerHTML=shown.map(record=>`<tr><td><a href="${esc(record.source)}" title="${esc(record.name)}" aria-label="KleidiAI source for ${esc(record.name)}" target="_blank" rel="noopener noreferrer">source</a></td><td><code class="signature">${esc(record.signature)}</code><div class="muted"><small>${esc(record.operation)}</small></div></td><td><span class="badge">${esc(record.isa)}</span>${record.tile!=='—'?`<span class="badge">${esc(record.tile)}</span>`:''}</td><td class="${record.availability==='available'?'pin-yes':'pin-no'}">${record.availability==='available'?'✓':'—'}</td><td><span class="status status-${esc(record.status)}">${esc(record.statusLabel)}</span><div class="links">${record.evidence.map(item=>`<a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">${esc(item.label)}</a>`).join('')}</div></td></tr>`).join('')}
function render(openKey=''){renderResults();renderActive();renderDropdowns(openKey)}search.addEventListener('input',()=>render());document.getElementById('clear').addEventListener('click',()=>{Object.values(selected).forEach(set=>set.clear());search.value='';render()});document.addEventListener('click',event=>{if(!event.target.closest('.filter-menu'))dropdowns.querySelectorAll('details').forEach(detail=>detail.open=false)});document.addEventListener('keydown',event=>{if(event.key==='Escape'){const open=[...dropdowns.querySelectorAll('details')].find(detail=>detail.open);if(open){open.open=false;open.querySelector('summary').focus()}}});render()})();
</script></body></html>'''


PAGES_TEMPLATE = r'''---
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
  <p class="kai-lede">Public KleidiAI micro-kernels and their exact ONNX Runtime MLAS integration status.</p>
  <p class="kai-meta"><span>KleidiAI <a href="__KAI_URL__"><code>__KAI_DESCRIBE__</code></a></span><span>ONNX Runtime <a href="__ORT_URL__"><code>__ORT_SHORT__</code></a></span><span>pin <code>__ORT_PIN__</code></span><span>PR audit __AUDITED_AT__</span></p>
  <p class="kai-note">Integration and pin availability are separate. A kernel can be absent from the current ONNX Runtime pin even when a newer integration pull request exists.</p>
  <section class="controls" aria-label="Kernel filters">
    <div class="search-row"><input id="search" class="search" type="search" placeholder="Search full kernel symbol…" aria-label="Search kernels"><button id="clear" class="clear" type="button">Clear filters</button></div>
    <div id="dropdowns" class="dropdown-grid"></div><div id="activeFilters" class="active"></div>
  </section>
  <section class="results"><div class="results-head"><h2>Matching kernels</h2><span id="resultCount" class="muted count" aria-live="polite"></span></div><div class="table-wrap"><table class="main-table"><thead><tr><th>Kernel</th><th>Type signature</th><th>ISA / tile</th><th>At pin</th><th>ONNX Runtime</th></tr></thead><tbody id="results"></tbody></table><div id="empty" class="empty" hidden>No kernels match these filters.</div></div></section>
</div>
<style>
.kleidiai-page{--kai-border:#eeebee;--kai-surface:#f5f6fa;--kai-hover:#ebedf5;--kai-text:#5c5962;--kai-heading:#27262b;--kai-muted:#716f75;--kai-link:#226aca;color:var(--kai-text)}
.kleidiai-page .kai-lede{font-size:1.05rem}.kleidiai-page .kai-meta{display:flex;flex-wrap:wrap;gap:.35rem 1rem;color:var(--kai-muted);font-size:.8rem}.kleidiai-page .kai-note{margin:1rem 0;padding:.75rem 1rem;border-left:3px solid var(--kai-link);background:var(--kai-surface);font-size:.85rem}
.kleidiai-page button,.kleidiai-page input{font:inherit}.kleidiai-page .controls{position:relative;z-index:5;margin:1.25rem 0;padding:.9rem;border:1px solid var(--kai-border);border-radius:.35rem;background:var(--kai-surface)}.kleidiai-page .search-row{display:grid;grid-template-columns:minmax(15rem,1fr) auto;gap:.55rem}.kleidiai-page .search{width:100%;min-width:0;padding:.6rem .7rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;color:var(--kai-text);outline:none}.kleidiai-page .search:focus{border-color:var(--kai-link);box-shadow:0 0 0 2px rgba(34,106,202,.18)}.kleidiai-page .clear{padding:.55rem .8rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;color:var(--kai-link);cursor:pointer}.kleidiai-page .clear:hover{background:var(--kai-hover)}
.kleidiai-page .dropdown-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.5rem;margin-top:.55rem}.kleidiai-page .filter-menu{position:relative;min-width:0;margin:0}.kleidiai-page .filter-menu summary{position:relative;display:flex;min-height:3rem;flex-direction:column;justify-content:center;gap:.1rem;padding:.45rem 1.6rem .45rem .6rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;cursor:pointer;list-style:none}.kleidiai-page .filter-menu summary::-webkit-details-marker{display:none}.kleidiai-page .filter-menu summary:after{content:"▾";position:absolute;right:.6rem;top:.85rem;color:var(--kai-muted)}.kleidiai-page .filter-menu[open] summary{border-color:var(--kai-link);box-shadow:0 0 0 2px rgba(34,106,202,.18)}.kleidiai-page .filter-menu[open] summary:after{transform:rotate(180deg)}.kleidiai-page .filter-name{color:var(--kai-muted);font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}.kleidiai-page .filter-value{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--kai-heading);font-size:.78rem}
.kleidiai-page .checklist{position:absolute;z-index:30;top:calc(100% + .3rem);left:0;width:max(100%,17rem);max-height:20rem;overflow:auto;padding:.4rem;border:1px solid #d7d4d7;border-radius:.25rem;background:#fff;box-shadow:0 .75rem 2rem rgba(39,38,43,.18)}.kleidiai-page .filter-menu:nth-child(4n) .checklist{right:0;left:auto}.kleidiai-page .check-option{display:grid;grid-template-columns:1rem minmax(0,1fr) auto;align-items:start;gap:.45rem;padding:.4rem;border-radius:.2rem;cursor:pointer;font-size:.78rem}.kleidiai-page .check-option:hover{background:var(--kai-hover)}.kleidiai-page .check-option input{margin:.15rem 0 0;accent-color:var(--kai-link)}.kleidiai-page .check-option .count,.kleidiai-page .muted{color:var(--kai-muted)}
.kleidiai-page .active{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.55rem}.kleidiai-page .filter-chip{padding:.25rem .5rem;border:1px solid #d7d4d7;border-radius:999px;background:#fff;color:var(--kai-text);cursor:pointer;font-size:.7rem}.kleidiai-page .filter-chip:hover{border-color:var(--kai-link);color:var(--kai-link)}
.kleidiai-page .results-head{display:flex;align-items:end;justify-content:space-between;gap:.75rem;margin:1.25rem 0 .55rem}.kleidiai-page .results-head h2{margin:0}.kleidiai-page .count{font-size:.8rem;font-variant-numeric:tabular-nums}.kleidiai-page .table-wrap{overflow:auto;border:1px solid var(--kai-border);border-radius:.35rem}.kleidiai-page .main-table{display:table;width:100%;min-width:47rem;margin:0;border-collapse:collapse}.kleidiai-page .main-table th,.kleidiai-page .main-table td{padding:.65rem .7rem;border-right:0;border-bottom:1px solid var(--kai-border);vertical-align:top;text-align:left}.kleidiai-page .main-table th{background:var(--kai-surface);color:var(--kai-muted);font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}.kleidiai-page .main-table td{background:#fff;font-size:.78rem}.kleidiai-page .main-table tbody tr:hover td{background:#fafbfc}.kleidiai-page .main-table td:first-child,.kleidiai-page .main-table th:first-child{text-align:center}.kleidiai-page .signature{overflow-wrap:anywhere}.kleidiai-page .badge{display:inline-flex;margin:.1rem .15rem .1rem 0;padding:.1rem .4rem;border:1px solid var(--kai-border);border-radius:999px;background:var(--kai-surface);color:var(--kai-muted);font-size:.65rem;white-space:nowrap}.kleidiai-page .status{font-weight:600}.kleidiai-page .status-integrated,.kleidiai-page .pin-yes{color:#13795b}.kleidiai-page .status-referenced{color:#1e5fb4}.kleidiai-page .status-pr{color:#6f42c1}.kleidiai-page .status-not-integrated,.kleidiai-page .pin-no{color:var(--kai-muted)}.kleidiai-page .links{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.15rem}.kleidiai-page .empty{padding:2rem;text-align:center;color:var(--kai-muted)}
@media(max-width:50rem){.kleidiai-page .dropdown-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.kleidiai-page .filter-menu:nth-child(4n) .checklist{right:auto;left:0}.kleidiai-page .filter-menu:nth-child(2n) .checklist{right:0;left:auto}}@media(max-width:31.25rem){.kleidiai-page .search-row,.kleidiai-page .dropdown-grid{grid-template-columns:1fr}.kleidiai-page .filter-menu:nth-child(2n) .checklist{right:auto;left:0}.kleidiai-page .checklist{width:100%}}
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
        .replace("__ORT_PIN__", metadata["ort_pin"])
        .replace("__AUDITED_AT__", metadata["audited_at"])
    )
    runtime = output[output.index('<script type="application/json" id="kernelData">') : output.rindex("</script>") + 9]
    pages_output = (
        PAGES_TEMPLATE.replace("__RUNTIME__", runtime)
        .replace("__KAI_URL__", f"{report.KAI_GITHUB}/commit/{metadata['kai_revision']}")
        .replace("__KAI_DESCRIBE__", metadata["kai_describe"])
        .replace("__ORT_URL__", f"{report.ORT_GITHUB}/commit/{metadata['ort_revision']}")
        .replace("__ORT_SHORT__", metadata["ort_short"])
        .replace("__ORT_PIN__", metadata["ort_pin"])
        .replace("__AUDITED_AT__", metadata["audited_at"])
    )
    for destination, contents in ((OUTPUT, output), (PAGES_OUTPUT, pages_output)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
        print(f"Generated {destination}")
    print({"microkernels": len(records), "kleidiai": metadata["kai_describe"]})


if __name__ == "__main__":
    main()
