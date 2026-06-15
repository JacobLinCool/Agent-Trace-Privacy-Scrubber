# Demo Video Script

This is the script for the included short demo video at `docs/demo-agent-trace-privacy-scrubber.mp4`.

1. Open Agent Trace Privacy Scrubber.
2. Choose **Use sample logs** so no real private trace is uploaded.
3. Click **Scan logs**.
4. Show that fake Codex, Claude Code, and Pi Agent traces are discovered.
5. Turn off **Model PII** for the fast recorded demo, leaving deterministic secret redaction enabled.
6. Click **Process selected logs**.
7. Show the completed run, regex redaction count, downloadable sanitized archive, report tab, and preview tab.

Longer narrated version:

> Agent traces are useful, but raw traces can leak secrets. This small Gradio app gives builders a local-first review step. I scan sample Codex, Claude Code, and Pi Agent logs, select the traces, run deterministic secret redaction, and download a sanitized archive with a report. For deeper PII detection, the app can run OpenMed privacy-filter locally or use the optional Modal GPU backend.
