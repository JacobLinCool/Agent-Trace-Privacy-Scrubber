---
title: Agent Trace Privacy Scrubber
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: true
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

Agent Trace Privacy Scrubber is a local-first Gradio workbench for cleaning raw agent traces before they are shared with teammates, uploaded to Hugging Face Datasets, or attached to public bug reports.

The small problem it solves is concrete: agent logs are useful, but they are messy and private. Codex, Claude Code, Pi Agent, and similar tools can leave prompts, shell output, file paths, tokens, credentials, private source snippets, emails, and other personal data in JSONL traces. This app gives builders a repeatable way to discover those logs, redact sensitive spans, inspect the report, and download a sanitized archive without overwriting the originals.

**For real private logs, run this locally. Do not upload sensitive traces to a public Space.**

## Hackathon Links

- Live Space: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
- GitHub repository: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber
- Demo video: https://youtu.be/Go2AcwcE72M
- Demo video file: [docs/demo-agent-trace-privacy-scrubber.mp4](docs/demo-agent-trace-privacy-scrubber.mp4)
- Article draft: [docs/article.md](docs/article.md)
- Social post draft: [docs/social-post.md](docs/social-post.md)
- Submission notes and quest checklist: [docs/submission-notes.md](docs/submission-notes.md)

Live social-post URL is still pending because the draft should not be posted without human approval.

## Why We Built It

Agent traces are becoming the new debugging artifact. They explain how an agent planned, which tools it called, where it failed, and what data it saw. That makes them valuable for reproducibility and community learning, especially in hackathons where sharing traces can help others understand the build.

The risk is that traces are not designed like polished logs. They often include whatever the agent observed while working: local paths, credentials copied into command output, private issue text, pasted prompts, or snippets from a closed repository. Builders then face a bad choice: share the trace and leak too much, or keep the trace private and lose the learning value.

This project takes the narrow middle path. It does not try to become a full compliance product. It gives individual builders and small teams a focused tool that runs on their machine, makes the privacy boundary explicit, and produces a redaction report they can inspect before publishing anything.

## Who It Is For

- Hackathon builders who want to share agent traces without leaking secrets.
- Open-source maintainers who need to attach useful logs to public issues.
- Researchers studying agent behavior from JSONL traces.
- Teams experimenting with Codex, Claude Code, Pi Agent, or custom agent runtimes.

## What It Does

1. Finds JSONL, NDJSON, and JSON traces from known local folders, custom paths, uploads, uploaded directories, or bundled fake sample logs.
2. Lets the user select exactly which files to process.
3. Runs deterministic secret redaction for API keys, bearer tokens, JWTs, Slack tokens, GitHub tokens, Hugging Face tokens, AWS-looking credentials, private-key blocks, credentialed URLs, `.env`-style assignments, and token-bearing query strings.
4. Optionally runs model PII redaction with `OpenMed/privacy-filter-nemotron` or the Apple Silicon MLX sibling.
5. Preserves JSONL structure where possible and records invalid JSON lines instead of silently dropping them.
6. Packages sanitized outputs, `redaction_report.json`, and `README_FIRST.txt` into a downloadable zip.
7. Keeps Modal cloud GPU inference opt-in and visible in the UI.

Original files are never overwritten.

## How To Use

1. Choose a source: known local source, custom local path, uploaded files, uploaded directory, or bundled sample logs.
2. Click **Scan logs**.
3. Review discovered files and select the traces to process.
4. Pick a backend, model, and mode:
   - `mask`: placeholders such as `<REDACTED:email>`.
   - `remove`: delete detected sensitive spans.
   - `hash`: stable placeholders such as `<HASHED:email:abc123>`.
   - `replace`: deterministic OpenMed/Faker replacement when supported.
5. Click **Process selected logs**.
6. Download the sanitized zip archive.
7. Manually review the sanitized output before publishing.

## Runtime Boundaries

The default current-runtime backend runs redaction inside the Python process where Gradio is running. On your Mac, that means local compute. On Hugging Face ZeroGPU, that means the Space runtime. The selected OpenMed model may download weights from Hugging Face the first time it runs, but trace contents are not sent to Hugging Face Inference API, OpenAI API, Anthropic API, analytics, telemetry, or any external LLM service.

The Modal backend is deliberately different: deterministic regex redaction runs first in the app process, then model-enabled string values are sent to the deployed Modal function for CUDA GPU inference. Use Modal only when you intentionally trust and control that deployment.

Do not launch with `share=True` for private logs.

## Models

Default model:

- `OpenMed/privacy-filter-nemotron`

Apple Silicon option:

- `OpenMed/privacy-filter-nemotron-mlx`

The OpenMed privacy-filter family is a small token-classification model family. The Hugging Face model listing identifies `OpenMed/privacy-filter-nemotron` as a 1B model, which is under the Build Small 32B cap and also below the Tiny Titan 4B threshold. The app currently does not tag Tiny Titan because the public Space demo is positioned around privacy workflow quality rather than smallest-weight impact.

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
  -> Modal L4/A10G function for OpenMed CUDA inference
```

The app intentionally keeps the data path simple. JSON values are parsed with Python's JSON tooling, string leaves are redacted recursively, non-string values are preserved, and invalid lines are sanitized as raw text with report warnings.

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

Current deployment used for the hackathon preparation:

- https://modal.com/apps/jacoblincool/main/deployed/agent-trace-privacy-scrubber

## Hugging Face Space Notes

The public Space is demo-able with the bundled fake logs in `sample_logs/`. A public Space is not the right place to upload real private traces unless you fully understand and control the deployment.

This MVP intentionally does not upload sanitized traces to Hugging Face. After local review, upload with the `hf` CLI if desired.

## Hackathon / Quest Positioning

Primary track:

- `track:backyard`: a practical privacy tool for everyday agent builders.

Sponsor quests:

- `sponsor:openai`: built and prepared with Codex, with Codex co-author trailers in the connected GitHub history.
- `sponsor:modal`: includes and deploys an opt-in Modal GPU backend, documented above.
- `sponsor:nvidia`: uses the OpenMed Nemotron privacy-filter checkpoint and the NVIDIA Nemotron-PII data lineage as the core model redaction path. If judges require only NVIDIA-published Nemotron 3 family checkpoints, this should be treated as conditional.

Achievements:

- `achievement:offbrand`: the app uses a custom workbench layout and CSS rather than the default Gradio look.
- `achievement:fieldnotes`: the repository includes a submission article and implementation notes.

## Codex Contribution

Codex helped turn the project from a working prototype into a submission package: reviewing the official Build Small requirements, tightening README metadata and tags, checking git hygiene, validating the app import path and tests, deploying the Modal backend, recording a short demo video with sample logs, drafting the article/social/submission notes, and ensuring commits include `Co-authored-by: Codex <codex@openai.com>`.

## Limitations

- PII detection is not perfect.
- Regex rules catch common secret shapes, not every proprietary credential format.
- Redaction reports contain counts and categories, not raw matched values.
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
