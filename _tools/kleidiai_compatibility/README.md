# KleidiAI compatibility page generator

This directory contains the self-contained Python generator used by the fork's
KleidiAI Pages preview. The workflow checks out ONNX Runtime and KleidiAI into
temporary source directories, regenerates `docs/performance/kleidiai/index.html`,
validates the result, and builds it with the same Jekyll and Svelte stages used
by the ONNX Runtime `gh-pages` workflow.

The reviewed open-pull-request mappings are deliberately stored in
`open_kernel_prs.json`; updating source revisions does not assert that newly
opened pull requests have been manually reviewed.

Environment variables accepted by the generator:

- `KLEIDIAI_SOURCE_ROOT`: KleidiAI checkout.
- `ONNXRUNTIME_SOURCE_ROOT`: ONNX Runtime source checkout.
- `KLEIDIAI_PR_CANDIDATES`: reviewed pull-request mapping JSON.
- `KLEIDIAI_PAGES_OUTPUT`: generated Jekyll page destination.
- `KLEIDIAI_STANDALONE_OUTPUT`: optional standalone report destination.
- `KLEIDIAI_MARKDOWN_OUTPUT`: generated GitHub fallback destination.
- `KLEIDIAI_MARKDOWN_STANDALONE_OUTPUT`: optional standalone Markdown destination.
