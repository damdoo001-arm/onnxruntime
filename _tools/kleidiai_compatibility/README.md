# KleidiAI compatibility page generator

This directory contains the self-contained Python generator used by the fork's
Pages preview. The workflow checks out `microsoft/onnxruntime@gh-pages` as an
immutable website baseline, scans the candidate commit from this fork plus a
selected KleidiAI revision, and writes the generated page into the runner's
temporary baseline tree. It then uses the same Jekyll and Svelte stages as the
ONNX Runtime publishing workflow and deploys only to this fork's Pages site.

The workflow has a hard guard for `damdoo001-arm/onnxruntime`, uses read-only
source checkouts without persisted credentials, and has no repository-content
write permission. A push to the fork's `main` scans that exact commit. Runs on
the preview branch and manual runs scan the selected `candidate_ref`, which
defaults to the fork's `main`.

The hand-authored documentation and landing site require no extra secret. To
also reproduce the generated C, C#, Java, Python, Objective-C, and JavaScript
API snapshots, configure a fork secret named `UPSTREAM_ARTIFACT_TOKEN` with
read access to public workflow artifacts in `microsoft/onnxruntime`.

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
