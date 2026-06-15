# Local Privacy Filtering for Shareable Agent Traces

The Build Small Hackathon surfaced a practical tension in agentic development: the most useful traces are also the most sensitive. Near the submission deadline, after finishing several projects, I reviewed the bonus badges and found Sharing is Caring, which encourages builders to publish traces from Codex, Claude Code, Pi Agent, and similar systems as [Hugging Face Datasets](https://huggingface.co/datasets). The premise is strong. Agent traces can show how a project was built with far more fidelity than a final demo or README.

A trace records the working process: prompts, plans, agent actions, files read, commands executed, errors encountered, and recovery steps. For other builders, this record can make agentic development more reproducible and easier to learn from. It shows collaboration with the agent as an empirical artifact in place of a polished retrospective.

That same fidelity creates the privacy problem. Session logs can contain local paths, private repository fragments, environment variables, API keys, bearer tokens, emails, project names, customer context, and personal data pasted into a prompt. They can also expose implementation context that has no obvious secret shape yet remains unsuitable for publication. A raw trace is therefore both a valuable learning artifact and a high-risk release artifact.

Agent Trace Privacy Scrubber addresses the missing step between local agent use and public trace sharing: a local privacy pass before the trace leaves the builder's control.

Demo video: https://youtu.be/Go2AcwcE72M

## Why Local Inference Matters

The hackathon's small-model constraint raised a broader design question for me: where does a local small model provide a capability that is distinct from hosted frontier inference?

Many AI tasks can be handled well by cloud APIs. Frontier models are often stronger, faster, and operationally easier than local models, especially when credits or reasonable pricing are available. Agent trace redaction has a different constraint. The raw input may contain the exact secrets and private context that redaction is meant to remove, so the redaction step should happen before any upload or remote inference call.

In this setting, locality is a privacy boundary. The model's value comes from where it runs: on the builder's machine, before dataset publication, bug-report attachment, or team sharing. A small local model becomes useful because it can operate at the point where disclosure risk is highest.

## System Scope

Agent Trace Privacy Scrubber is a local-first Gradio application for sanitizing agent session logs. The app supports common local sources such as Codex sessions at `~/.codex/sessions`, Claude Code projects at `~/.claude/projects`, and Pi Agent sessions at `~/.pi/agent/sessions`. It also accepts custom folders, uploaded files, uploaded directories, and bundled fake sample logs for public demonstration.

The workflow is intentionally narrow. The user scans a source, reviews the discovered JSONL, NDJSON, or JSON traces, selects the files to process, chooses a redaction mode, and downloads a sanitized archive. The archive contains the processed traces, `redaction_report.json`, and `README_FIRST.txt`. Source logs remain unchanged.

This scope keeps the application focused on the pre-publication privacy boundary. Dataset publication, trace viewing, and safety certification remain outside this scope. The application gives builders a structured first pass before human review.

## Redaction Method

The scrubber combines deterministic secret detection with optional model-based PII detection.

The deterministic layer handles high-risk credential patterns whose structure is known in advance: OpenAI-style keys, Anthropic keys, Hugging Face tokens, GitHub tokens, Slack tokens, JWTs, bearer tokens, AWS-looking credentials, private-key blocks, credentialed URLs, `.env` assignments, and token-bearing query strings. These cases benefit from explicit rules because a single missed credential can matter more than a graceful model judgment.

The model layer uses [`OpenMed/privacy-filter-nemotron`](https://huggingface.co/OpenMed/privacy-filter-nemotron), with the Apple Silicon MLX sibling [`OpenMed/privacy-filter-nemotron-mlx`](https://huggingface.co/OpenMed/privacy-filter-nemotron-mlx) available for local use. The model card describes the checkpoint as a fine-tune of [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) on [`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII), with 55 fine-grained PII categories. This specialized token-classification model fits the task better than a general prompt asking an LLM to identify sensitive text: the task is span-level privacy filtering, and the model is trained for that family of decisions.

The processor preserves JSON structure where possible. It parses valid JSON lines, recursively redacts string values, keeps non-string values intact, and records invalid JSON lines as warnings while sanitizing their raw text. This matters because downstream consumers often expect stable JSONL structure.

## Product Boundaries

Local-first privacy also shapes the interface and runtime design.

The default backend runs inside the same Python process as the Gradio app. On a local machine, the privacy pass uses local compute. On the public Hugging Face Space, it runs in the Space runtime. The model may download weights from Hugging Face on first use, while trace contents stay out of hosted LLM APIs, analytics, and telemetry.

The Modal backend is an explicit remote-compute option for CUDA inference. In that path, deterministic redaction runs first, then model-enabled string values are sent to the deployed Modal function. The app presents this as a trust boundary, so users can choose speed and hardware access with a clear understanding of where the data goes.

The interface also treats progress visibility as part of the privacy workflow. Trace folders can be large, model initialization can take time, and JSONL files are processed line by line. The UI reports the current phase, file index, current file, current line, total progress, ETA, and aggregate redaction counts. For local processing, these signals help users understand that the system is working and what stage it has reached.

The generated report follows the same privacy principle. It records categories, counts, timings, invalid-line warnings, and file-level summaries while omitting raw matched values, so the report remains a summary artifact.

Sharing is Caring supplied the motivating use case, and it clarified the product boundary. Trace sharing is valuable because it exposes the real path of agentic development; trace sharing is risky because that path may include private data. The scrubber lowers the barrier to participation by adding a local review step before dataset release.

The scrubber provides a bounded first pass. Automatic redaction can miss project-specific identifiers, unusual credential formats, sensitive code fragments, and contextual information that only the builder can judge. The model is primarily useful for PII spans; deterministic rules are strongest for known secret patterns. Human review remains part of the release process.

That scope defines the application's role. Agent Trace Privacy Scrubber helps builders convert raw traces into reviewable release candidates, while keeping the final publication decision with the person who understands the project context.

For me, this was the strongest lesson from the project: local small models are most compelling when locality is part of the value proposition. Agent trace redaction is one of those cases. The trace is worth sharing, the raw data is sensitive, and the privacy pass belongs on the builder's machine.

## Links

- Live Space: https://huggingface.co/spaces/build-small-hackathon/agent-trace-privacy-scrubber
- GitHub: https://github.com/JacobLinCool/Agent-Trace-Privacy-Scrubber
