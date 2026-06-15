# Social Post Draft

Do not publish this automatically. Human review is required before posting.

## X / Twitter Version

I built Agent Trace Privacy Scrubber for the Build Small hackathon.

Agent traces are great for debugging, but raw Codex / Claude / Pi logs can leak prompts, paths, tokens, emails, and private output. This Gradio app scans JSONL traces, redacts secrets + PII, and packages a sanitized archive with a report.

Built local-first with OpenMed privacy-filter, optional Modal GPU inference, and Codex helping across implementation, tests, deployment, demo recording, and docs.

Demo: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
GitHub: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber

#BuildSmall #Gradio #HuggingFace #Codex #LocalAI #Privacy #Modal

## LinkedIn / Longer Version

I built Agent Trace Privacy Scrubber for the Hugging Face x Gradio Build Small hackathon.

The problem is simple: agent traces are becoming one of the most useful debugging artifacts, but raw traces can contain exactly the things you do not want to publish: prompts, tool inputs, terminal output, local file paths, API keys, tokens, emails, and private repository snippets.

This app gives builders a local-first review step before sharing:

- scan Codex, Claude Code, Pi Agent, custom, or uploaded JSONL traces,
- select exactly which files to process,
- run deterministic secret redaction,
- optionally run OpenMed privacy-filter PII redaction,
- download sanitized traces with a redaction report.

It is a small Backyard AI tool: focused, practical, and made for people building with agents right now.

Codex helped with implementation cleanup, tests, official submission checks, Modal deployment, demo recording, and the final documentation package.

Live demo: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
GitHub: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber

#BuildSmall #Gradio #HuggingFace #Codex #OpenSource #Privacy #AgenticAI
