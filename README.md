---
title: Local Agent Trace Privacy Scrubber
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
tags:
  - gradio
  - privacy
  - agent-traces
  - codex
  - claude-code
  - nemotron
  - local-ai
  - build-small
  - backyard-ai
  - best-use-of-codex
  - nvidia-nemotron
# TODO: verify exact Build Small hackathon tag slugs against the latest field guide.
---

# Local Agent Trace Privacy Scrubber

A local-first Gradio app for sanitizing raw JSONL agent traces before uploading them to Hugging Face Datasets, Storage Buckets, or any public trace viewer.

Agent traces from Codex, Claude Code, and Pi Agent can include prompts, tool inputs, command output, private paths, screenshots references, secrets, private code, and personal data. This app helps you discover those logs, select the files to process, run deterministic secret redaction and optional OpenMed/Nemotron PII redaction, then download sanitized JSONL files as a zip archive with a redaction report.

**For real private logs, run this locally. Do not upload sensitive traces to a public Space.**

## Why Local Processing Matters

The default current-runtime backend runs redaction inside the Python process where Gradio is running. On your Mac, that means local compute. On Hugging Face ZeroGPU, that means the Space runtime. The selected OpenMed model may download weights from Hugging Face the first time it runs, but trace contents are not sent to Hugging Face Inference API, OpenAI API, Anthropic API, analytics, telemetry, or any external LLM service.

The optional Modal backend is different: deterministic regex redaction runs first in the app process, then model-enabled string values are sent to your deployed Modal app for CUDA GPU inference. Use it only when you intentionally trust and control that Modal deployment.

Do not launch with `share=True` for private logs.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal.

For Apple Silicon / MLX:

```bash
pip install -U "openmed[mlx]"
```

## Modal Cloud GPU Backend

Modal is an opt-in remote compute backend for CUDA GPU model inference. Deploy the worker once:

```bash
modal token new
modal deploy modal_app.py
```

Then select **Modal cloud GPU** in the app settings. Modal currently supports `OpenMed/privacy-filter-nemotron` in this project. Use the current-runtime backend for Apple MLX models.

## Common Trace Locations

- Codex sessions: `~/.codex/sessions`
- Claude Code projects: `~/.claude/projects`
- Pi Agent sessions: `~/.pi/agent/sessions`

When the app runs locally, the local path scanner reads your own machine. When deployed as a remote Hugging Face Space, local path mode reads the Space container, not your computer. Use remote Spaces only with sample or non-sensitive uploaded traces.

## How To Use

1. Choose a source: known local source, custom local path, uploaded files, uploaded directory, or bundled sample logs.
2. Click **Scan logs**.
3. Review discovered JSONL/NDJSON/JSON files and choose all or individual files.
4. Pick a compute backend, model, and redaction mode:
   - `mask`: placeholders such as `<REDACTED:email>`.
   - `remove`: delete detected sensitive spans.
   - `hash`: stable placeholders such as `<HASHED:email:abc123>`.
   - `replace`: deterministic OpenMed/Faker replacement when supported.
5. Click **Process selected logs**.
6. Download the zip archive containing sanitized traces, `redaction_report.json`, and `README_FIRST.txt`.
7. Review sanitized outputs manually before publishing.

Original files are never overwritten.

## Redaction Pipeline

For each selected file, the processor reads UTF-8 text with safe replacement for decoding errors and streams JSONL line by line.

- Valid JSON lines are parsed, and only string values are recursively redacted. Object keys, arrays, numbers, booleans, nulls, and line boundaries are preserved.
- Invalid JSON lines are sanitized as raw text and recorded in the report.
- Deterministic regex sweeps run before and after model redaction.
- Optional OpenMed PII redaction uses the selected backend.
- Long strings are chunked for model inference.
- Outputs preserve relative paths under a temp output directory.

The deterministic sweep includes rules for OpenAI-style keys, Anthropic keys, Hugging Face tokens, GitHub tokens, Slack tokens, JWTs, AWS access keys and secret assignments, private keys, `.env`-style secret assignments, bearer tokens, credentialed URLs, and obvious token-bearing query parameters.

## Models

Default model:

- `OpenMed/privacy-filter-nemotron`

Apple Silicon options:

- `OpenMed/privacy-filter-nemotron-mlx`

OpenMed exposes `extract_pii()` and `deidentify()` APIs for local PII detection and de-identification. This app uses local extraction spans so it can count labels and preserve reports without including raw matched values.

## Hugging Face Space Notes

This project is demo-able as a Space using the bundled fake logs in `sample_logs/`. A public Space is not an appropriate place to upload real private traces unless you fully understand and control the deployment.

This MVP intentionally does not upload sanitized traces to Hugging Face. After local review, upload with the `hf` CLI if desired.

## Limitations

- PII detection is not perfect.
- OpenMed privacy-filter-nemotron is English-focused and trained on synthetic Nemotron-PII data; real agent traces may differ from the model's training distribution.
- Regex rules catch common secret shapes, not every proprietary credential format.
- Redaction reports contain counts and categories, not raw matched values.
- This is not legal, compliance, or security certification advice.

## Development

Run tests:

```bash
pytest
```

Project structure:

```text
app.py
src/trace_scrubber/
tests/
sample_logs/
requirements.txt
```
