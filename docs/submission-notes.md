# Build Small Submission Notes

Prepared on June 15, 2026.

Official guide checked:

- https://build-small-hackathon-field-guide.hf.space/submit
- https://build-small-hackathon-field-guide.hf.space/

## Submission Summary

Agent Trace Privacy Scrubber is a local-first Gradio workbench for sanitizing JSONL agent traces before sharing them publicly. It scans Codex, Claude Code, Pi Agent, custom, uploaded, or sample traces; redacts common secrets deterministically; optionally runs OpenMed privacy-filter PII redaction; preserves JSONL structure where possible; and packages sanitized logs with an auditable redaction report.

## Main Requirement Checklist

| Requirement | Status | Evidence / Action |
| --- | --- | --- |
| Stay under 32B parameters | Complete | `OpenMed/privacy-filter-nemotron` is listed as a 1B token-classification model on Hugging Face. The app does not depend on any model above 32B. |
| Ship a Gradio app in the official org | Complete after Space push | README front matter sets `sdk: gradio`, `app_file: app.py`, and the target Space is `build-small-hackathon/agent-trace-privacy-scrubber`. |
| Record a demo video | Complete | `docs/demo-agent-trace-privacy-scrubber.mp4` records the sample-log flow. |
| Post one social-media post | Human action required | Draft is in `docs/social-post.md`. It has not been posted because the user explicitly requested drafts only. |
| Mind ZeroGPU limit | Complete | This is one ZeroGPU-compatible Gradio Space. Modal is optional remote compute. |
| Tag README for tracks/prizes/badges | Complete | README uses official `track:*`, `sponsor:*`, and `achievement:*` tag formats discovered from the submission app. |
| Add short write-up of idea and tech | Complete | README and `docs/article.md` explain problem, user, workflow, architecture, model, Modal, and Codex contribution. |
| Public GitHub repository | Complete after GitHub push | Target: `https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber`. |
| Secrets and large files excluded | Complete | `.gitignore` excludes `.env`, virtualenvs, caches, artifacts, and build outputs. Fake credentials in `sample_logs/` and tests are clearly labeled fake fixtures. |

## Quest / Challenge Eligibility

| Quest / Prize | Eligibility | Current Status | Notes |
| --- | --- | --- | --- |
| Backyard AI track | Yes | Ready | Practical privacy tool for everyday agent builders. Tag: `track:backyard`. |
| Thousand Token Wood track | No | Not targeted | The project is practical rather than whimsical. |
| Best Use of Codex | Yes | Ready after history rewrite/push | Requires Codex-attributed commits in GitHub or Space. All commits should include `Co-authored-by: Codex <codex@openai.com>`. Tag: `sponsor:openai`. |
| Best MiniCPM Build | No | Not targeted | The project does not use MiniCPM models. |
| Nemotron Hardware Prize | Conditional | Tagged with caveat | The core model path uses `OpenMed/privacy-filter-nemotron` and NVIDIA Nemotron-PII lineage. If judges require only NVIDIA-published Nemotron 3 checkpoints, this may not qualify. Tag: `sponsor:nvidia`. |
| Best Use of Modal | Yes | Ready | `modal_app.py` deploys an opt-in CUDA redaction backend, and README documents Modal use. Tag: `sponsor:modal`. |
| Off-Brand / Custom UI | Yes | Ready | The app uses custom Gradio CSS and a workbench layout beyond default components. Tag: `achievement:offbrand`. |
| Tiny Titan | Possible but not targeted | Not tagged | The model is small enough, but the submission story is about privacy workflow, not smallest-weight impact. |
| Best Demo | Partially ready | Demo done; social post pending | Demo video exists. Public social post still requires human publishing. |
| Best Agent | No | Not targeted | The app processes agent artifacts but is not itself a multi-step agentic app. |
| Bonus Quest Champion | Possible | Depends on final social post and judge interpretation | Current strongest achievements are Off-Brand and Field Notes. |
| Field Notes achievement | Yes | Ready | `docs/article.md` and this file document the build. Tag: `achievement:fieldnotes`. |
| Off the Grid achievement | Not tagged | Not targeted | The app is local-first and does not call hosted LLM APIs by default, but optional Modal is a remote backend, so this is not claimed. |
| Well-Tuned achievement | No | Not targeted | The project uses an existing model and did not publish a new fine-tune. |
| Llama Champion achievement | No | Not targeted | The model does not run through llama.cpp. |
| Sharing is Caring achievement | No | Not targeted | We intentionally avoid publishing real agent traces. |

## Deployment Resources

- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
- Modal deployment: https://modal.com/apps/jacoblincool/main/deployed/agent-trace-privacy-scrubber
- GitHub repository: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber
- Demo video in repo/Space: `docs/demo-agent-trace-privacy-scrubber.mp4`

## Manual Actions Remaining

1. Publish the social post from `docs/social-post.md`.
2. Replace the pending social-post note in README with the public social URL.
3. Optionally upload the demo MP4 to YouTube or another public host; the file is already included in the repository and Space.
4. Submit the Space URL, GitHub URL, demo URL, social URL, and summary through the Build Small submission form.

## Submission Form Draft

Agent Trace Privacy Scrubber is a local-first Gradio workbench for sanitizing raw agent traces before sharing them. Agent logs from Codex, Claude Code, Pi Agent, and similar tools are useful for debugging, but they can leak prompts, file paths, shell output, tokens, credentials, emails, and private repository content. The app scans local or uploaded JSONL traces, lets users choose specific files, runs deterministic secret redaction plus optional OpenMed privacy-filter PII redaction, preserves structure where possible, and packages sanitized traces with a redaction report.

The project targets the Backyard AI track as a practical tool for builders working with agents. It uses a small OpenMed privacy-filter Nemotron model under the 32B limit, includes an opt-in Modal GPU backend, and has a custom Gradio workbench UI. Codex assisted with implementation review, tests, deployment prep, Modal deployment, demo recording, README cleanup, and submission docs. All connected GitHub commits are intended to include `Co-authored-by: Codex <codex@openai.com>`.
