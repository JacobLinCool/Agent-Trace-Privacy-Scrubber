---
title: Agent Trace Privacy Scrubber
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: true
models:
  - OpenMed/privacy-filter-nemotron
  - openai/privacy-filter
datasets:
  - nvidia/Nemotron-PII
tags:
  - gradio
  - privacy
  - agent-traces
  - trace-redaction
  - local-ai
  - build-small
  - track:backyard
  - sponsor:openai
  - sponsor:nvidia
  - sponsor:modal
  - achievement:offbrand
  - achievement:fieldnotes
---

# Agent Trace Privacy Scrubber

Agent Trace Privacy Scrubber is a local-first Gradio app for cleaning Codex,
Claude Code, Pi Agent, and other agent session logs before they are shared.

The project started from a very specific Build Small Hackathon tension:
Sharing is Caring encourages builders to publish agent traces so others can see
how a project was actually made, but raw traces are often full of local paths,
private repository snippets, shell output, prompts, emails, API keys, tokens,
customer context, and accidental personal data. A trace is useful precisely
because it is real. That is also why it can be unsafe to publish unchanged.

This app is the local review step before a trace leaves your machine. It scans
agent log folders, lets you choose the files to process, redacts common secrets
with deterministic rules, optionally runs a small privacy-filter model for PII,
and packages sanitized JSONL plus a redaction report for manual review.

**For real private logs, run this locally. Do not upload sensitive traces to a
public Space.**

## Hackathon Links

- Live Space: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
- GitHub repository: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber
- Agent trace dataset: https://huggingface.co/datasets/build-small-hackathon/agent-trace-privacy-scrubber-codex-traces
- Demo video: https://youtu.be/Go2AcwcE72M
- Demo video file: [docs/demo-agent-trace-privacy-scrubber.mp4](docs/demo-agent-trace-privacy-scrubber.mp4)
- Published article: https://huggingface.co/blog/build-small-hackathon/agent-trace-privacy
- Article source: [docs/article.md](docs/article.md)
- Social post: https://x.com/JacobLinCool/status/2066455131189850204
- Social post draft: [docs/social-post.md](docs/social-post.md)
- Submission notes and quest checklist: [docs/submission-notes.md](docs/submission-notes.md)

## Submission Checklist

- [x] Tags: `track:backyard`, sponsor tags, and achievement badge tags are in
  the YAML block at the top of this README.
- [x] Demo video: https://youtu.be/Go2AcwcE72M
- [x] Social link: https://x.com/JacobLinCool/status/2066455131189850204
- [x] Team HF usernames: `jacoblincool`
- [x] Agent trace dataset:
  https://huggingface.co/datasets/build-small-hackathon/agent-trace-privacy-scrubber-codex-traces

## Why This Should Be Local

Cloud frontier models are strong, fast, and convenient. For many hackathon apps,
using a small local model can feel like a constraint: you are doing something a
larger hosted model could probably do better.

Trace scrubbing is different. The sensitive data should not be sent to a remote
API in order to decide whether it is safe to send out. The privacy boundary has
to exist before upload, before dataset publication, and before a public bug
report. In this workflow, local inference is not just a cheaper replacement for
cloud inference; it is part of the product guarantee.

That is the Build Small fit: a small model is valuable here because it can run
where the data already is, under the builder's control.

## Who It Is For

- Hackathon builders preparing sanitized traces for Sharing is Caring-style
  datasets.
- Builders who want to publish useful Codex, Claude Code, or Pi Agent traces
  without leaking secrets.
- Open-source maintainers who need to attach agent logs to public issues.
- Researchers studying agent workflows from JSONL traces.
- Small teams adopting agentic coding tools and wanting a repeatable local
  privacy pass.

## What It Does

1. Finds JSONL, NDJSON, and JSON traces from known local folders, custom paths,
   uploads, uploaded directories, or bundled fake sample logs.
2. Supports known local sources for Codex (`~/.codex/sessions`), Claude Code
   (`~/.claude/projects`), and Pi Agent (`~/.pi/agent/sessions`).
3. Lets the user select exactly which files to process.
4. Runs deterministic secret redaction for API keys, bearer tokens, JWTs, Slack
   tokens, GitHub tokens, Hugging Face tokens, AWS-looking credentials,
   private-key blocks, credentialed URLs, `.env`-style assignments, and
   token-bearing query strings.
5. Optionally runs model PII redaction with `OpenMed/privacy-filter-nemotron` or
   the Apple Silicon MLX sibling.
6. Preserves JSONL structure where possible and records invalid JSON lines
   instead of silently dropping them.
7. Shows progress while processing: current phase, file index, current file,
   line count, total progress, ETA, and redaction counts.
8. Packages sanitized outputs, `redaction_report.json`, and `README_FIRST.txt`
   into a downloadable zip.
9. Keeps Modal cloud GPU inference opt-in and visible in the UI.

Original files are never overwritten.

## How To Use

1. Choose a source: known local source, custom local path, uploaded files,
   uploaded directory, or bundled sample logs.
2. Click **Scan logs**.
3. Review discovered files and select the traces to process.
4. Pick a backend, model, and redaction mode:
   - `mask`: placeholders such as `<REDACTED:email>`.
   - `remove`: delete detected sensitive spans.
   - `hash`: stable placeholders such as `<HASHED:email:abc123>`.
   - `replace`: deterministic OpenMed/Faker replacement when supported.
5. Click **Process selected logs**.
6. Download the sanitized zip archive.
7. Manually review the sanitized output before publishing.

## Runtime Boundaries

The default current-runtime backend runs redaction inside the Python process
where Gradio is running. On your Mac, that means local compute. On Hugging Face
ZeroGPU, that means the Space runtime. The selected OpenMed model may download
weights from Hugging Face the first time it runs, but trace contents are not
sent to Hugging Face Inference API, OpenAI API, Anthropic API, analytics,
telemetry, or any external LLM service.

The Modal backend is deliberately different: deterministic regex redaction runs
first in the app process, then model-enabled string values are sent to the
deployed Modal function for CUDA GPU inference. Use Modal only when you
intentionally trust and control that deployment.

Do not launch with `share=True` for private logs.

## Models

Default model:

- `OpenMed/privacy-filter-nemotron`

Apple Silicon option:

- `OpenMed/privacy-filter-nemotron-mlx`

The OpenMed privacy-filter Nemotron model is a token-classification model for
PII extraction. Its model card describes it as a fine-tune of
`openai/privacy-filter` on the `nvidia/Nemotron-PII` dataset with 55
fine-grained PII categories. Hugging Face lists the PyTorch checkpoint at 1B
parameters, which is under the Build Small 32B cap and below the Tiny Titan 4B
threshold. This submission is positioned around privacy workflow quality, so it
does not claim Tiny Titan as its main story.

## Technical Architecture

```text
Gradio UI (app.py)
  -> discovery.py: source scanning and table rows
  -> jsonl_processor.py: streaming JSONL/NDJSON/JSON processing
  -> redactors.py: deterministic secrets + OpenMed PII redaction
  -> modal_backend.py: optional Modal function client
  -> reporting.py: report preview rows
  -> zipper.py: sanitized archive packaging

modal_app.py
  -> Modal RTX PRO 6000 (H100 fallback) function for OpenMed CUDA inference
```

The data path is intentionally simple. JSON values are parsed with Python's JSON
tooling, string leaves are redacted recursively, non-string values are preserved,
and invalid lines are sanitized as raw text with report warnings.

The redaction report contains counts, categories, timings, invalid-line counts,
and warnings. It does not include the raw matched secret or PII values, because
the report must not become a second leak.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open the local Gradio URL printed in the terminal.

For Apple Silicon / MLX:

```bash
pip install -U "openmed[mlx]"
```

## Modal Cloud GPU Backend

Modal is an opt-in remote compute backend for CUDA GPU model inference. Deploy
the worker once:

```bash
modal token new
modal deploy modal_app.py
```

Then select **Modal cloud GPU** in the app settings. Modal currently supports
`OpenMed/privacy-filter-nemotron` in this project. Use the current-runtime
backend for Apple MLX models.

The worker defaults to an **RTX PRO 6000** GPU (H100 fallback) with **model batch
size 64** — see the benchmark below for why.

Current deployment used for the hackathon preparation:

- https://modal.com/apps/jacoblincool/main/deployed/agent-trace-privacy-scrubber

### Hardware benchmark (how the default GPU was chosen)

We benchmarked all seven Modal GPU types on a redaction workload of 1000 chunks
(1000–3000 chars each) at model batch size 64. Throughput is per-GPU model
inference; price is Modal's on-demand rate. `$/1000 chunks = price / throughput`.

| GPU | chunks/s | $/sec | $/1000 chunks |
| --- | --- | --- | --- |
| **RTX PRO 6000** (default) | **51.0** | 0.000842 | **$0.0165** |
| H100 | 42.3 | 0.001097 | $0.0259 |
| A100-40GB | 26.2 | 0.000583 | $0.0223 |
| L40S | 25.5 | 0.000542 | $0.0213 |
| A10 | 19.5 | 0.000306 | $0.0157 |
| A100-80GB | 19.6 | 0.000694 | $0.0354 |
| L4 | 15.2 | 0.000222 | $0.0146 |

Why **RTX PRO 6000 + batch 64**:

- **Fastest** by a wide margin (≈1.2× an H100, ≈2× an A100/L40S on this small NER
  model) and second-cheapest per chunk — the best speed-for-cost on the menu.
- Its 96 GB easily fits batch 64 plus the occasional multi-MB trace string.

Other findings worth knowing:

- **Whole-string caching gives ≈2× more** on real traces. Agent logs repeat the
  same short structural strings (`role`, `event_msg`, …) thousands of times; the
  pipeline is per-item-overhead-bound, so deduping ~60–80% of string leaves
  roughly doubles end-to-end throughput on every GPU. The network request batch
  is decoupled from the GPU batch so this does not enlarge GPU memory.
- **24 GB cards (L4, A10) must stay at batch ≤ 16** on real traces. The model
  uses eager O(seq²) attention and real (token-dense) content can OOM them at
  batch 32; the 40 GB+ cards are unaffected.
- **A100-80GB is the worst value here** — slower than the cheaper A100-40GB and
  L40S, and most expensive per chunk. Avoid it for this workload.

Reproduce with `bench_modal.py` (pure GPU throughput) and `bench_modal_v2.py`
(caching effect + per-GPU memory limits).

## Hugging Face Space Notes

The public Space is demo-able with bundled fake logs in `sample_logs/`. A public
Space is not the right place to upload real private traces unless you fully
understand and control the deployment.

This MVP intentionally does not upload sanitized traces to Hugging Face. After
local review, publish with the `hf` CLI or the Hugging Face web UI if desired.

## Build Small Positioning

Primary track:

- `track:backyard`: a practical privacy tool for everyday agent builders.

Sponsor quests:

- `sponsor:openai`: built and prepared with Codex, with Codex co-author trailers
  in the connected GitHub history.
- `sponsor:modal`: includes and deploys an opt-in Modal GPU backend, documented
  above.
- `sponsor:nvidia`: uses the OpenMed Nemotron privacy-filter checkpoint and the
  NVIDIA Nemotron-PII data lineage as the core model redaction path. If judges
  require only NVIDIA-published Nemotron 3 family checkpoints, this should be
  treated as conditional.

Achievements:

- `achievement:offbrand`: the app uses a custom workbench layout and CSS rather
  than the default Gradio look.
- `achievement:fieldnotes`: the repository includes a submission article and
  implementation notes.

Sharing is Caring is the reason this app exists: it lowers the risk of
publishing useful agent traces. The app itself does not automatically publish
datasets or claim that sanitized output is safe without review.

## Codex Contribution

Codex helped turn the project from a working prototype into a submission
package: reviewing the Build Small requirements, tightening README metadata and
tags, checking git hygiene, validating imports and tests, deploying the Modal
backend, recording a sample-log demo video, and drafting article/social/submission
notes.

## Privacy Design Principles

- Do not overwrite source logs.
- Do not send raw traces to hosted LLM APIs by default.
- Do not print raw sensitive spans into the console.
- Do not put raw matched values into the redaction report.
- Keep remote compute explicit and opt-in.
- Make long-running local processing observable with progress and ETA.
- Treat automatic redaction as a first pass, not a substitute for human review.

## Limitations

- PII detection is not perfect.
- Regex rules catch common secret shapes, not every proprietary credential
  format.
- The model is trained primarily for English PII and may miss unusual project
  context or private code.
- Modal mode sends regex-sanitized string values to a remote Modal deployment.
- This is not legal, compliance, or security certification advice.

## Development

Run tests:

```bash
pytest
```

Project structure:

```text
app.py
modal_app.py
src/trace_scrubber/
tests/
sample_logs/
docs/
requirements.txt
```
