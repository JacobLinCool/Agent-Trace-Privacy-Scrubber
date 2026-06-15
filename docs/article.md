# Agent Trace Privacy Scrubber: Cleaning Agent Logs Before They Leave Your Machine

Agent Trace Privacy Scrubber is a small Gradio app for a problem that appears right after an agent becomes useful: the trace is worth sharing, but the trace may also be private.

Codex, Claude Code, Pi Agent, and similar tools leave behind JSONL logs that are excellent for debugging and learning. They show prompts, plans, tool calls, shell output, failures, and recovery steps. In a hackathon, that kind of trace can help other builders understand what happened. In an open-source issue, it can make a bug reproducible.

But raw traces are not polished logs. They can include local paths, pasted secrets, command output, email addresses, private repository snippets, bearer tokens, `.env` values, or a screenshot reference that should never become public. The safest choice is often not to share them at all, which means losing a useful artifact.

This project takes a deliberately narrow middle path: make it easy for a builder to sanitize agent traces locally, inspect what changed, and only then decide whether to share the result.

## The App

The workflow is built around a single workbench:

1. Pick a source: known local agent folders, a custom path, uploaded files, an uploaded directory, or bundled fake sample logs.
2. Scan JSONL, NDJSON, and JSON traces.
3. Select the files to process.
4. Choose a redaction policy.
5. Run deterministic secret redaction and optional model PII detection.
6. Download a sanitized archive with a redaction report.

The app never overwrites the original files. The output zip contains sanitized traces, `redaction_report.json`, and `README_FIRST.txt`, so the user has a review step before publishing anything.

## Technical Implementation

The main app is a Gradio Blocks interface in `app.py`. The core logic is split into small modules:

- `discovery.py` finds trace files and prepares table rows.
- `jsonl_processor.py` streams JSONL/NDJSON/JSON and preserves structure where possible.
- `redactors.py` runs deterministic secret rules and optional OpenMed PII redaction.
- `modal_backend.py` delegates model inference to a Modal function when the user opts in.
- `reporting.py` and `zipper.py` build preview rows and downloadable archives.

The deterministic pass catches common high-risk patterns: OpenAI-style keys, Anthropic keys, Hugging Face tokens, GitHub tokens, Slack tokens, JWTs, bearer tokens, AWS-looking credentials, private-key blocks, `.env` assignments, credentialed URLs, and token-bearing query parameters.

For model redaction, the app uses `OpenMed/privacy-filter-nemotron`, a small token-classification model for PII spans. Apple Silicon users can choose the MLX sibling. The default backend runs in the current Python process. On a local laptop, that is local compute; on the public Space, that is the Space runtime. Modal is available as an explicit remote GPU backend, but the UI calls out that boundary before the user sends anything.

## Build Small Fit

This is a Backyard AI project: practical, focused, and built for a real workflow. The target user is not an abstract enterprise buyer. It is a person who just finished an agentic coding session and wants to share the useful parts without exposing the private parts.

It also fits several Build Small quests:

- Best Use of Codex: Codex helped implement, review, package, document, and prepare the project, and the connected GitHub commits include Codex co-author trailers.
- Best Use of Modal: the app includes a deployed Modal backend for optional GPU model inference.
- Off-Brand / Custom UI: the app uses a custom workbench layout rather than the stock Gradio look.
- Field Notes: this article and the submission notes document what was built and what tradeoffs remain.

The NVIDIA Nemotron sponsor quest is a conditional fit: the model path is built around OpenMed's Nemotron privacy-filter checkpoint and NVIDIA Nemotron-PII data lineage. If judging requires only NVIDIA-published Nemotron 3 family checkpoints, that quest may not apply.

## Challenges

The hardest design issue was not the regex list. It was making the privacy boundary visible without turning the UI into a warning page. The app has two compute paths, and they have different trust implications. The current-runtime path keeps processing on the local machine or Space runtime. The Modal path is useful for CUDA inference, but it is remote compute. The interface and README say that directly.

Another challenge was preserving structure. Agent logs are often consumed by tools that expect one JSON object per line. The processor parses valid JSON lines, redacts string leaves recursively, preserves non-string values, and records invalid lines in the report rather than pretending everything was clean.

Finally, the public demo needed to avoid real sensitive data. The Space ships with fake sample traces that look like Codex, Claude Code, and Pi Agent logs but contain only fake credentials and fake personal data.

## How Codex Helped

Codex helped across the full submission loop:

- reviewed the official Build Small requirements and tag format,
- checked `.gitignore` and repository hygiene,
- validated tests and app import behavior,
- deployed the Modal backend,
- recorded a short sample-log demo video,
- rewrote the README for the hackathon audience,
- drafted the article, social post, and submission checklist,
- and ensured commit history uses `Co-authored-by: Codex <codex@openai.com>`.

The result is intentionally small: one Gradio app, one clear privacy workflow, one downloadable archive, and a story that fits the Build Small theme.

## Links

- Live Space: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
- GitHub: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber
- Demo video: `docs/demo-agent-trace-privacy-scrubber.mp4`
- Submission notes: `docs/submission-notes.md`
