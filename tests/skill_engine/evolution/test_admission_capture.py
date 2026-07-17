from types import SimpleNamespace

from openspace.skill_engine.evidence.types import (
    EvidencePacket,
    EvidenceScope,
    PacketBudget,
    ResourceRef,
)
from openspace.skill_engine.evolution.admission import EvolutionAdmission


def _packet_with_tool_ref(tool_ref: ResourceRef) -> EvidencePacket:
    runtime_ref = ResourceRef(
        ref_id="runtime:finish",
        ref_type="runtime_snapshot",
        preview="Task completed successfully.",
        metadata={
            "status": "success",
            "final_response_preview": "Done.",
        },
    )
    return EvidencePacket(
        packet_id="pkt_test",
        trigger_job_id="trg_test",
        packet_type="analysis",
        profile_name="analysis_current_task",
        subprofile="task_finished",
        manifest_watermark=1,
        scope=EvidenceScope(source_task_ids=("task_test",)),
        selected_refs={
            "runtime_snapshot": [runtime_ref],
            "tool_event": [
                tool_ref,
                ResourceRef(
                    ref_id="tool:validate",
                    ref_type="tool_event",
                    preview="Independent postcondition check passed.",
                    metadata={"tool_name": "bash", "status": "success"},
                ),
            ],
        },
        expanded_snippets=[],
        readable_paths=[],
        instructions={},
        budget=PacketBudget(max_chars=0, used_chars=0),
        redaction_status="ok",
        build_status="ok",
        missing_ref_types=[],
    )


def _packet_with_legacy_repeated_candidate_ref(tool_ref: ResourceRef) -> EvidencePacket:
    packet = _packet_with_tool_ref(tool_ref)
    selected_refs = {
        key: list(value)
        for key, value in packet.selected_refs.items()
    }
    selected_refs["evolution_candidate_ref"] = [
        ResourceRef(
            ref_id="candidate:cand_repeated",
            ref_type="evolution_candidate_ref",
            preview="CAPTURED candidate pending recurrence=2",
            metadata={
                "candidate_id": "cand_repeated",
                "recurrence": "repeated",
                "recurrence_count": 2,
                "source_task_ids": ["task_test", "task_other"],
            },
        )
    ]
    return EvidencePacket(
        packet_id=packet.packet_id,
        trigger_job_id=packet.trigger_job_id,
        packet_type=packet.packet_type,
        profile_name=packet.profile_name,
        subprofile=packet.subprofile,
        manifest_watermark=packet.manifest_watermark,
        scope=packet.scope,
        selected_refs=selected_refs,
        expanded_snippets=packet.expanded_snippets,
        readable_paths=packet.readable_paths,
        instructions=packet.instructions,
        budget=packet.budget,
        redaction_status=packet.redaction_status,
        build_status=packet.build_status,
        missing_ref_types=packet.missing_ref_types,
    )


def _captured_decision() -> SimpleNamespace:
    return SimpleNamespace(
        decision_id="dec_test",
        proposed_action="CAPTURED",
        candidate_policy="candidate",
        reason_summary=(
            "Capture the reusable asyncio cancellation workflow for cancelling "
            "child tasks, awaiting cleanup, and re-raising cancellation."
        ),
        reason_tags=["captured", "workflow"],
        risks=[],
        evidence_claims=[
            SimpleNamespace(
                claim="Reusable workflow was observed in runtime and tool evidence.",
                refs=["tool:test", "tool:validate"],
            )
        ],
        proposal_contract={
            "capability": "Cancel child tasks and verify cleanup.",
            "preconditions": ["Child tasks can be awaited."],
            "procedure_refs": ["tool:test"],
            "validation_refs": ["tool:validate"],
            "validation_summary": "A separate check observed completed cleanup.",
            "limitations": [],
        },
    )


def test_scratchpad_tmp_path_does_not_make_capture_ephemeral() -> None:
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview='python /tmp/sigint_test.py returned "cleanup complete"',
            metadata={"tool_name": "bash", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(_captured_decision(), packet)

    assert result.outcome == "direct"
    assert result.source_validation_passed is True
    assert "ephemeral_or_secret_dependent_capture" not in result.warnings


def test_secret_text_still_blocks_captured_admission() -> None:
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview="curl -H 'Authorization: Bearer secret-token' https://example.test",
            metadata={"tool_name": "bash", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(_captured_decision(), packet)

    assert result.outcome == "noop"
    assert result.warnings == ["ephemeral_or_secret_dependent_capture"]


def test_unrelated_secret_ref_does_not_block_contract_capture() -> None:
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview="python verify_workflow.py returned all checks passed",
            metadata={"tool_name": "bash", "status": "success"},
        )
    )
    packet.selected_refs["tool_event"].append(
        ResourceRef(
            ref_id="tool:unrelated",
            ref_type="tool_event",
            preview="Authorization: Bearer unrelated-secret-token",
            metadata={"tool_name": "bash", "status": "success"},
            contains_secret=True,
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(_captured_decision(), packet)

    assert result.outcome == "direct"
    assert result.source_validation_passed is True


def test_capture_that_disclaims_output_correctness_is_not_admitted() -> None:
    decision = _captured_decision()
    decision.proposal_contract["limitations"] = [
        "Correctness of the reported frame numbers was not independently verified."
    ]
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview="analyzer wrote takeoff and landing frame numbers",
            metadata={"tool_name": "bash", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(decision, packet)

    assert result.outcome == "noop"
    assert result.warnings == [
        "capture_contract_disclaims_output_correctness"
    ]


def test_bounded_hidden_answer_limitation_remains_admissible() -> None:
    decision = _captured_decision()
    decision.proposal_contract["limitations"] = [
        "The parse check does not prove that output matches a hidden answer set."
    ]
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview="query parsed and returned rows",
            metadata={"tool_name": "bash", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(decision, packet)

    assert result.outcome == "direct"


def test_permission_bypass_capture_is_not_admitted() -> None:
    decision = _captured_decision()
    decision.reason_summary = (
        "Use another tool to bypass a blocking permission approval prompt."
    )
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview="read-only file region returned",
            metadata={"tool_name": "read", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(decision, packet)

    assert result.outcome == "noop"
    assert result.warnings == ["permission_bypass_capture"]


def test_compiler_tokenizing_text_does_not_look_like_a_secret_token() -> None:
    packet = _packet_with_tool_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview=(
                "The C preprocessor is tokenizing a triple-quoted string in a "
                "skipped #if block; the compiled program still passes."
            ),
            metadata={"tool_name": "bash", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=True,
    ).admit(_captured_decision(), packet)

    assert result.outcome == "direct"
    assert result.source_validation_passed is True
    assert "ephemeral_or_secret_dependent_capture" not in result.warnings


def test_repeated_candidate_ref_does_not_bypass_provisional_policy() -> None:
    packet = _packet_with_legacy_repeated_candidate_ref(
        ResourceRef(
            ref_id="tool:test",
            ref_type="tool_event",
            preview="python verify_workflow.py returned all checks passed",
            metadata={"tool_name": "bash", "status": "success"},
        )
    )

    result = EvolutionAdmission(
        allow_single_observation_capture=False,
    ).admit(_captured_decision(), packet)

    assert result.outcome == "candidate"
    assert "provisional_evolution_disabled" in result.warnings
