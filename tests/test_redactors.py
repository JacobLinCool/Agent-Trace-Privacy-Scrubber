from __future__ import annotations

import json

from trace_scrubber.jsonl_processor import sanitize_json_value
from trace_scrubber.redactors import RedactionConfig, SecretRegexRedactor


def test_regex_redacts_common_secrets() -> None:
    text = """
    OPENAI_API_KEY=sk-proj-FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE
    ANTHROPIC_API_KEY=sk-ant-FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE
    HF_TOKEN=hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE
    GITHUB_TOKEN=ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE
    SLACK_TOKEN=xoxb-FAKEFAKEFAKE-FAKEFAKEFAKE-FAKEFAKEFAKE
    AWS_ACCESS_KEY_ID=AKIAFAKEFAKEFAKE1234
    AWS_SECRET_ACCESS_KEY=fakeAwsSecretValue1234567890
    Authorization: Bearer fakebearertokenfakebearertoken
    https://fake-user:fake-pass@example.invalid/api?token=fake-query-token-1234567890
    eyJmYWtlZmFrZWZha2U.eyJmYWtlZmFrZWZha2U.eyJmYWtlZmFrZWZha2U
    -----BEGIN OPENSSH PRIVATE KEY-----
    FAKEFAKEFAKEFAKEFAKE
    -----END OPENSSH PRIVATE KEY-----
    """

    result = SecretRegexRedactor(seed=7).redact(text, "mask")

    assert "sk-proj-FAKE" not in result.text
    assert "sk-ant-FAKE" not in result.text
    assert "hf_FAKE" not in result.text
    assert "ghp_FAKE" not in result.text
    assert "xoxb-FAKE" not in result.text
    assert "fakeAwsSecretValue" not in result.text
    assert "fakebearertoken" not in result.text
    assert "fake-user:fake-pass" not in result.text
    assert "BEGIN OPENSSH PRIVATE KEY" not in result.text
    assert result.regex_counts["openai_api_key"] >= 1
    assert result.regex_counts["private_key_block"] == 1


def test_recursive_json_string_redaction_preserves_structure() -> None:
    config = RedactionConfig(regex_enabled=True, model_enabled=False)
    payload = {
        "role": "user",
        "content": "email alice.fake@example.com with sk-proj-FAKEFAKEFAKEFAKEFAKEFAKE",
        "nested": ["HF token hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE", 123, True, None],
    }

    sanitized, result = sanitize_json_value(payload, config)
    dumped = json.dumps(sanitized)

    assert sanitized["role"] == "user"
    assert sanitized["nested"][1:] == [123, True, None]
    assert "sk-proj-FAKE" not in dumped
    assert "hf_FAKE" not in dumped
    assert result.regex_counts["openai_api_key"] == 1
    assert result.regex_counts["huggingface_token"] == 1
