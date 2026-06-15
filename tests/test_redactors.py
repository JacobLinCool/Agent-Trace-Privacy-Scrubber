from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from trace_scrubber.jsonl_processor import sanitize_json_value
from trace_scrubber.redactors import (
    OpenMedPIIRedactor,
    RedactionConfig,
    SecretRegexRedactor,
)


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


def test_openmed_privacy_filter_pipeline_is_cached() -> None:
    @dataclass
    class FakeEntity:
        text: str = "alice@example.com"
        label: str = "EMAIL"
        start: int = 6
        end: int = 23
        confidence: float = 0.99

    @dataclass
    class FakeResult:
        entities: list[FakeEntity]

    created_pipelines: list[object] = []
    used_pipelines: list[object] = []

    def create_pipeline(model_name: str) -> object:
        pipeline = object()
        created_pipelines.append(pipeline)
        return pipeline

    def extract_batch(*args: Any, **kwargs: Any) -> list[FakeResult]:
        used_pipelines.append(kwargs["privacy_filter_pipeline"])
        return [FakeResult(entities=[FakeEntity()])]

    redactor = OpenMedPIIRedactor()
    redactor._extract_pii = lambda *args, **kwargs: None
    redactor._extract_pii_batch = extract_batch
    redactor._create_privacy_filter_pipeline = create_pipeline
    redactor._privacy_backend_selector = lambda model_name: "mlx"
    redactor._privacy_model_detector = lambda model_name: True
    config = RedactionConfig(
        model_name="OpenMed/privacy-filter-nemotron-mlx",
        regex_enabled=False,
        model_enabled=True,
    )

    redactor.prepare_model(config)
    first = redactor.redact("email alice@example.com now", config)
    second = redactor.redact("email alice@example.com now", config)

    assert len(created_pipelines) == 1
    assert used_pipelines == [created_pipelines[0], created_pipelines[0]]
    assert first.text == "email <REDACTED:email> now"
    assert second.text == "email <REDACTED:email> now"

    redactor.release()
    assert redactor._privacy_filter_pipelines == {}


def test_structural_tokens_are_not_redacted() -> None:
    @dataclass
    class FakeEntity:
        text: str
        label: str
        start: int
        end: int
        confidence: float = 0.99

    @dataclass
    class FakeResult:
        entities: list[FakeEntity]

    def extract_batch(texts: list[str], **kwargs: Any) -> list[FakeResult]:
        # Label every input fully as an occupation, like the model does for roles.
        return [
            FakeResult(
                entities=[FakeEntity(text=t, label="occupation", start=0, end=len(t))]
            )
            for t in texts
        ]

    redactor = OpenMedPIIRedactor()
    redactor._extract_pii = lambda *args, **kwargs: None
    redactor._extract_pii_batch = extract_batch
    redactor._create_privacy_filter_pipeline = lambda model_name: object()
    redactor._privacy_backend_selector = lambda model_name: "mlx"
    redactor._privacy_model_detector = lambda model_name: True
    config = RedactionConfig(
        model_name="OpenMed/privacy-filter-nemotron-mlx",
        regex_enabled=False,
        model_enabled=True,
    )
    redactor.prepare_model(config)

    # "assistant" (a message role) must survive untouched...
    assert redactor.redact("assistant", config).text == "assistant"
    assert redactor.redact("response_item", config).text == "response_item"
    # ...while a genuine occupation is still redacted.
    assert redactor.redact("Dentist", config).text == "<REDACTED:occupation>"


def test_is_structural_token_is_case_and_whitespace_insensitive() -> None:
    from trace_scrubber.redactors import _is_structural_token

    assert _is_structural_token("assistant")
    assert _is_structural_token("  USER ")
    assert _is_structural_token("Response_Item")
    assert not _is_structural_token("alice@example.com")
    assert not _is_structural_token("Dr. Smith")


def test_non_mlx_privacy_filter_prefers_selected_torch_device() -> None:
    created: list[dict[str, Any]] = []

    class FakeTorchPrivacyFilterPipeline:
        def __init__(
            self, model_name: str, *, device: str, trust_remote_code: bool
        ) -> None:
            created.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "trust_remote_code": trust_remote_code,
                }
            )

    redactor = OpenMedPIIRedactor()
    redactor._extract_pii = lambda *args, **kwargs: None
    redactor._extract_pii_batch = lambda *args, **kwargs: []
    redactor._create_privacy_filter_pipeline = lambda model_name: object()
    redactor._privacy_backend_selector = lambda model_name: "torch"
    redactor._privacy_model_resolver = lambda model_name, backend: model_name
    redactor._torch_privacy_filter_pipeline_cls = FakeTorchPrivacyFilterPipeline
    redactor._is_trusted_for_remote_code = lambda model_name: True
    redactor._privacy_model_detector = lambda model_name: True
    redactor._select_torch_device = lambda: "mps"  # type: ignore[method-assign]

    pipeline = redactor._get_privacy_filter_pipeline("OpenMed/privacy-filter-nemotron")

    assert isinstance(pipeline, FakeTorchPrivacyFilterPipeline)
    assert created == [
        {
            "model_name": "OpenMed/privacy-filter-nemotron",
            "device": "mps",
            "trust_remote_code": True,
        }
    ]
