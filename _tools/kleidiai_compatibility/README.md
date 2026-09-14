# KleidiAI compatibility page generator

This directory contains the self-contained Python generator used by the fork's
Pages preview. The workflow checks out `microsoft/onnxruntime@gh-pages` as an
immutable website baseline, analyses the newest stable `vX.Y.Z` KleidiAI and
ONNX Runtime tags, and writes the generated page into the runner's temporary
baseline tree. It then uses the same Jekyll and Svelte stages as the ONNX
Runtime publishing workflow and deploys only to this fork's Pages site.

The workflow has a hard guard for `damdoo001-arm/onnxruntime`, uses read-only
source checkouts without persisted credentials, and has no repository-content
write permission. Push and manual runs both resolve stable releases from full
source checkouts; development commits and release candidates are not used.

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

Use the orchestration entry point for both local and CI regeneration:

```sh
python3 _tools/kleidiai_compatibility/update_page.py \
  --kleidiai-root /path/to/kleidiai \
  --onnxruntime-root /path/to/onnxruntime-source
```

It generates the HTML and Markdown outputs and validates the Jekyll page. Both
source checkouts must have official release tags available, for example via
`git fetch --tags upstream`.
