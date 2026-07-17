import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from openspace.application import OpenSpaceConfig
from openspace.skill_engine.evidence.types import (
    EvidencePacket,
    EvidenceScope,
    PacketBudget,
    ResourceRef,
)
from openspace.skill_engine.evolution.authoring import (
    SkillEvolverAuthoringBackend,
    _normalize_captured_edit_content,
)
from openspace.skill_engine.evolution.capture_semantic import (
    CaptureContractSemanticReviewer,
)
from openspace.skill_engine.evolution.engine import EvolutionEngine
from openspace.skill_engine.evolution.validator import (
    EvolutionValidator,
    ValidationResult,
    _has_secondary_frontmatter,
)


class _FakeClient:
    def __init__(
        self,
        payload: str,
        *,
        stop_reason: str = "stop",
        tool_calls=None,
    ) -> None:
        self.payload = payload
        self.stop_reason = stop_reason
        self.tool_calls = tool_calls or []
        self.calls = []

    async def call_model_with_fallback(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            assistant_message={"content": self.payload},
            tool_calls=self.tool_calls,
            stop_reason=self.stop_reason,
        )


def _review_payload(**overrides):
    payload = {
        "outcome": "approve",
        "support": "fully_supported",
        "validation_relation": "independent_method",
        "skill_fidelity": "bounded",
        "artifact_quality": "sound",
        "supporting_validation_ref_ids": ["tool:validation"],
        "unsupported_claims": [],
        "reuse_quality": "reusable",
        "safety": "safe",
        "reasons": ["A second decoder produced the same float."],
    }
    payload.update(overrides)
    return payload


def test_semantic_reviewer_approves_bounded_independent_method() -> None:
    client = _FakeClient(json.dumps(_review_payload()))
    reviewer = CaptureContractSemanticReviewer(
        client,
        model="test-model",
        max_tokens=1024,
    )

    result = asyncio.run(
        reviewer.validate(
            _authoring(),
            _packet(),
            _decision(),
            SimpleNamespace(),
        )
    )

    assert result["outcome"] == "approve"
    assert "capture_support:fully_supported" in result["warnings"]
    assert [tool.name for tool in client.calls[0]["tools"]] == [
        "submit_capture_review"
    ]
    assert client.calls[0]["tool_choice"] == "required"
    assert client.calls[0]["enable_thinking"] is False
    assert client.calls[0]["max_tokens"] == 1024
    prompt = client.calls[0]["messages"][1]["content"]
    assert "A limitation does not" in prompt
    assert "supporting_validation_ref_ids" in prompt


def test_semantic_reviewer_prefers_required_tool_result_over_truncation() -> None:
    payload = _review_payload(
        validation_relation="cross_implementation_comparison",
        reasons=["Both implementations returned the same values."],
    )
    client = _FakeClient(
        "",
        stop_reason="length",
        tool_calls=[
            {
                "id": "call_review",
                "type": "function",
                "function": {
                    "name": "submit_capture_review",
                    "arguments": json.dumps(payload),
                },
            }
        ],
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "approve"
    assert "capture_support:fully_supported" in result["warnings"]


def test_semantic_review_config_is_fail_closed_by_default(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENSPACE_EVOLUTION_CAPTURE_SEMANTIC_VALIDATION_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "OPENSPACE_EVOLUTION_CAPTURE_SEMANTIC_VALIDATION_MAX_TOKENS",
        "1536",
    )
    config = OpenSpaceConfig()

    assert config.evolution_capture_semantic_validation_enabled is True
    assert config.evolution_capture_semantic_validation_max_tokens == 1536


def test_semantic_reviewer_rejects_same_implementation_self_check() -> None:
    client = _FakeClient(
        json.dumps(
            _review_payload(
                validation_relation="same_implementation",
                reasons=["The validation reran the same detector."],
            )
        )
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(
            _authoring(),
            _packet(),
            _decision(),
            SimpleNamespace(),
        )
    )

    assert result["outcome"] == "reject"
    assert "capture_validation_relation:same_implementation" in result["warnings"]


def test_semantic_reviewer_rejects_defective_authored_artifact() -> None:
    client = _FakeClient(
        json.dumps(
            _review_payload(
                artifact_quality="defective",
                reasons=["The helper returns triples but dict() expects pairs."],
            )
        )
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "reject"
    assert "capture_artifact_quality:defective" in result["warnings"]


def test_semantic_reviewer_rejects_unvalidated_skill_claim() -> None:
    client = _FakeClient(
        json.dumps(
            _review_payload(
                unsupported_claims=[
                    "The skill computes all date ranges, but only today was checked."
                ],
                reasons=["The validation ref establishes only today's count."],
            )
        )
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "reject"
    assert any(
        warning.startswith("capture_unsupported_claim:")
        for warning in result["warnings"]
    )


def test_semantic_reviewer_rejects_invalid_validation_ref_binding() -> None:
    client = _FakeClient(
        json.dumps(
            _review_payload(
                supporting_validation_ref_ids=["tool:procedure"],
                reasons=["The procedure ref was incorrectly used as validation."],
            )
        )
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "reject"
    assert "capture_supporting_validation_refs:invalid" in result["warnings"]


def test_semantic_reviewer_rejects_incomplete_high_impact_skill() -> None:
    client = _FakeClient(
        json.dumps(
            _review_payload(
                safety="unsafe_or_incomplete",
                reasons=["The sanitizer documents known executable bypasses."],
            )
        )
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "reject"
    assert "capture_safety:unsafe_or_incomplete" in result["warnings"]


def test_semantic_reviewer_rejects_trivial_capture() -> None:
    client = _FakeClient(
        json.dumps(
            _review_payload(
                reuse_quality="trivial",
                reasons=["The artifact only restates one generic command."],
            )
        )
    )
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "reject"
    assert "capture_reuse_quality:trivial" in result["warnings"]


def test_semantic_reviewer_retries_invalid_structured_response_once() -> None:
    valid_payload = _review_payload(
        reasons=["The independent decoder returned the same value."],
    )

    class RecoveringClient:
        def __init__(self):
            self.calls = []

        async def call_model_with_fallback(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    assistant_message={"content": "not json"},
                    tool_calls=[],
                    stop_reason="stop",
                )
            return SimpleNamespace(
                assistant_message={"content": ""},
                tool_calls=[
                    {
                        "function": {
                            "name": "submit_capture_review",
                            "arguments": valid_payload,
                        }
                    }
                ],
                stop_reason="tool_calls",
            )

    client = RecoveringClient()
    reviewer = CaptureContractSemanticReviewer(client, model="test-model")

    result = asyncio.run(
        reviewer.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert result["outcome"] == "approve"
    assert len(client.calls) == 2
    assert "previous response did not satisfy" in client.calls[1]["messages"][1][
        "content"
    ]


def test_semantic_reviewer_fails_closed_on_invalid_or_truncated_output() -> None:
    invalid = CaptureContractSemanticReviewer(
        _FakeClient("not json"),
        model="test-model",
    )
    truncated = CaptureContractSemanticReviewer(
        _FakeClient("{}", stop_reason="length"),
        model="test-model",
    )

    invalid_result = asyncio.run(
        invalid.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )
    truncated_result = asyncio.run(
        truncated.validate(_authoring(), _packet(), _decision(), SimpleNamespace())
    )

    assert invalid_result == {
        "outcome": "reject",
        "warnings": ["capture_semantic_invalid_json"],
    }
    assert truncated_result == {
        "outcome": "reject",
        "warnings": ["capture_semantic_response_truncated"],
    }


def test_async_validator_applies_semantic_rejection() -> None:
    async def reject(*_args):
        return {
            "outcome": "reject",
            "warnings": ["capture_support:partially_supported"],
        }

    validator = EvolutionValidator(
        semantic_validator=reject,
        semantic_enabled=True,
    )
    base = ValidationResult(
        validation_id="val_test",
        authoring_id="auth_test",
        decision_id="dec_test",
        packet_id="pkt_test",
        outcome="approve",
    )
    validator._run_validation = lambda *_args, **_kwargs: base
    validator._persist_fail_closed = lambda result: result

    result = asyncio.run(
        validator.validate_async(
            _authoring(),
            _packet(),
            _decision(),
            SimpleNamespace(),
        )
    )

    assert result.outcome == "reject"
    assert result.checked_by.endswith("+semantic")
    assert result.semantic_warnings == ["capture_support:partially_supported"]


def test_engine_prefers_async_validator_entrypoint() -> None:
    calls = []

    class Validator:
        async def validate_async(self, *_args):
            calls.append("async")
            return "ok"

        def validate(self, *_args):
            raise AssertionError("sync validator must not be selected")

    engine = EvolutionEngine()
    result = asyncio.run(
        engine._call_validator(
            Validator(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )
    )

    assert result == "ok"
    assert calls == ["async"]


def test_semantic_validation_failure_becomes_audit_candidate() -> None:
    class Authoring:
        async def author_from_action_packet(self, _packet):
            return SimpleNamespace(
                status="staged",
                staged_edit=SimpleNamespace(),
            )

    class PacketBuilder:
        def build_action_packet(self, _scope):
            return SimpleNamespace(packet_type="action")

        def build_validator_packet(self, _authoring):
            return SimpleNamespace(packet_type="validator")

    class Validator:
        async def validate_async(self, *_args):
            return ValidationResult(
                validation_id="val_reject",
                authoring_id="auth_test",
                decision_id="dec_test",
                packet_id="pkt_validator",
                outcome="reject",
                semantic_warnings=["capture_support:partially_supported"],
            )

    class CandidateStore:
        def create_or_merge(self, **kwargs):
            return SimpleNamespace(reason=kwargs["reason"])

    engine = EvolutionEngine(
        packet_builder=PacketBuilder(),
        authoring_backend=Authoring(),
        validator=Validator(),
        candidate_store=CandidateStore(),
    )

    outcome = asyncio.run(
        engine._author_validate_commit(
            SimpleNamespace(
                decision_id="dec_test",
                proposed_action="CAPTURED",
            ),
            SimpleNamespace(admission_id="adm_test", outcome="direct"),
            SimpleNamespace(packet_type="analysis"),
            SimpleNamespace(job_id="job_test"),
        )
    )

    assert outcome.committed_action is None
    assert outcome.candidate.reason == "semantic_validation_failed"
    assert outcome.blocked_reason == "validation_failed"


def test_captured_authoring_discards_model_narration_before_frontmatter() -> None:
    content = (
        "I have enough evidence to write the skill.\n\n"
        "---\nname: recover-floats\ndescription: Decode floats.\n---\n"
        "# Recover floats\n"
    )

    normalized = _normalize_captured_edit_content(content)

    assert normalized.startswith("---\nname: recover-floats")
    assert "I have enough evidence" not in normalized


def test_captured_authoring_uses_contract_without_audit_tools(tmp_path) -> None:
    class Evolver:
        _model = "test-model"

        def __init__(self):
            self.available_tools = None

        async def _run_evolution_loop(self, _prompt, ctx):
            self.available_tools = ctx.available_tools
            return SimpleNamespace(
                edit_content=(
                    "---\nname: decode-float\ndescription: Decode one float.\n---\n"
                    "# Decode one float\n"
                ),
                overlay_fields={},
                overlay_metadata={},
                intent_spec={},
                eval_plan={},
                change_summary="Decode one float.",
            )

    evolver = Evolver()
    backend = SkillEvolverAuthoringBackend(
        evolver,
        tmp_path / "staging",
        evidence_store=None,
    )
    base_packet = _packet()
    selected_refs = dict(base_packet.selected_refs)
    selected_refs["decision_rationale_ref"] = [
        ResourceRef(
            ref_id="decision:dec_test",
            ref_type="decision_rationale_ref",
            preview="Capture the bounded float decoder.",
            metadata={
                "decision_id": "dec_test",
                "proposed_action": "CAPTURED",
                "reason_summary": "Capture the bounded float decoder.",
                "proposal_contract": _decision().proposal_contract,
            },
        )
    ]
    selected_refs["admission_result_ref"] = [
        ResourceRef(
            ref_id="admission:adm_test",
            ref_type="admission_result_ref",
            metadata={
                "admission_id": "adm_test",
                "outcome": "direct",
                "source_validation_passed": True,
            },
        )
    ]
    packet = replace(
        base_packet,
        packet_type="action",
        selected_refs=selected_refs,
        instructions={"capture_destination_root": str(tmp_path / "skills")},
    )

    result = asyncio.run(backend.author_from_action_packet(packet))

    assert result.status == "staged"
    assert evolver.available_tools == []


def test_secondary_frontmatter_is_rejected_but_fenced_example_is_allowed() -> None:
    malformed = (
        "---\nname: outer\ndescription: Outer.\n---\n"
        "Narration\n---\nname: inner\ndescription: Inner.\n---\n"
    )
    documented = (
        "---\nname: outer\ndescription: Outer.\n---\n"
        "```yaml\n---\nname: example\n---\n```\n"
    )

    assert _has_secondary_frontmatter(malformed) is True
    assert _has_secondary_frontmatter(documented) is False
    failures = EvolutionValidator()._schema_failures(
        {"SKILL.md": malformed},
        "CAPTURED",
        SimpleNamespace(),
    )
    assert "multiple_frontmatter_blocks" in failures


def _decision():
    return SimpleNamespace(
        proposed_action="CAPTURED",
        proposal_contract={
            "capability": "Decode one float and verify it independently.",
            "preconditions": ["The section is readable."],
            "procedure_refs": ["tool:procedure"],
            "validation_refs": ["tool:validation"],
            "validation_summary": "A second decoder returned the same value.",
            "limitations": [],
        },
    )


def _authoring():
    return SimpleNamespace(
        staged_edit=SimpleNamespace(
            content_snapshot={
                "SKILL.md": (
                    "---\nname: decode-float\ndescription: Decode one float.\n---\n"
                    "# Decode one float\n"
                )
            }
        )
    )


def _packet() -> EvidencePacket:
    return EvidencePacket(
        packet_id="pkt_test",
        trigger_job_id="trg_test",
        packet_type="validator",
        profile_name="analysis_current_task",
        subprofile="validator:test",
        manifest_watermark=1,
        scope=EvidenceScope(task_id="task_test"),
        selected_refs={
            "tool_event": [
                ResourceRef(
                    ref_id="tool:procedure",
                    ref_type="tool_event",
                    preview="decoder A returned -1.5",
                    metadata={
                        "status": "success",
                        "tool_use_id": "call_a",
                        "input_preview": "python decoder_a.py",
                    },
                ),
                ResourceRef(
                    ref_id="tool:validation",
                    ref_type="tool_event",
                    preview="decoder B returned -1.5",
                    metadata={
                        "status": "success",
                        "tool_use_id": "call_b",
                        "input_preview": "objdump then decode manually",
                    },
                ),
            ]
        },
        expanded_snippets=[],
        readable_paths=[],
        instructions={},
        budget=PacketBudget(max_chars=1000, used_chars=100),
        redaction_status="ok",
        build_status="ok",
        missing_ref_types=[],
    )
