from datetime import datetime

from openspace.skill_engine.decision.analysis_adapter import AnalyzerDecisionAdapter
from openspace.skill_engine.evidence.types import (
    EvidencePacket,
    EvidenceScope,
    PacketBudget,
    ResourceRef,
)
from openspace.skill_engine.evolution.admission import EvolutionAdmission
from openspace.skill_engine.types import (
    CaptureContract,
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
)


def test_incomplete_task_can_capture_a_validated_subworkflow() -> None:
    packet = _packet(status="error", validation_status="success")
    analysis = _analysis(task_completed=False, with_contract=True)

    decisions = AnalyzerDecisionAdapter().from_analysis(analysis, packet)

    assert len(decisions) == 1
    assert decisions[0].proposed_action == "CAPTURED"
    assert decisions[0].proposal_contract["capability"]
    assert decisions[0].evidence_claims[0].refs == ["tool:procedure"]
    assert decisions[0].evidence_claims[1].refs == ["tool:validation"]


def test_completed_task_without_capture_contract_is_not_captured() -> None:
    packet = _packet(status="success", validation_status="success")
    analysis = _analysis(task_completed=True, with_contract=False)

    decisions = AnalyzerDecisionAdapter().from_analysis(analysis, packet)

    assert len(decisions) == 1
    assert decisions[0].proposed_action == "NOOP"
    assert decisions[0].noop_reason == "missing_capture_contract"


def test_qemu_launch_with_failed_login_validation_is_not_captured() -> None:
    packet = _packet(status="success", validation_status="error")
    analysis = _analysis(task_completed=True, with_contract=True)

    decisions = AnalyzerDecisionAdapter().from_analysis(analysis, packet)

    assert len(decisions) == 1
    assert decisions[0].proposed_action == "NOOP"
    assert decisions[0].noop_reason == "capture_validation_failed"


def test_video_output_without_independent_validation_is_not_captured() -> None:
    packet = _packet(status="success", validation_status="success")
    analysis = _analysis(task_completed=True, with_contract=True)
    contract = analysis.evolution_suggestions[0].capture_contract
    assert contract is not None
    contract.validation_refs = []

    decisions = AnalyzerDecisionAdapter().from_analysis(analysis, packet)

    assert decisions[0].proposed_action == "NOOP"
    assert decisions[0].noop_reason == "missing_validation_refs"


def test_polyglot_write_plus_separate_compile_run_check_is_admitted() -> None:
    packet = _packet(status="success", validation_status="success")
    analysis = _analysis(task_completed=True, with_contract=True)
    decision = AnalyzerDecisionAdapter().from_analysis(analysis, packet)[0]

    admission = EvolutionAdmission().admit(decision, packet)

    assert decision.proposed_action == "CAPTURED"
    assert admission.outcome == "direct"
    assert admission.source_validation_passed is True


def test_procedure_and_validation_must_use_distinct_refs() -> None:
    packet = _packet(status="success", validation_status="success")
    analysis = _analysis(task_completed=True, with_contract=True)
    analysis.evolution_suggestions[0].capture_contract.validation_refs = [
        "tool:procedure"
    ]

    decisions = AnalyzerDecisionAdapter().from_analysis(analysis, packet)

    assert decisions[0].proposed_action == "NOOP"
    assert decisions[0].noop_reason == "procedure_validation_refs_overlap"


def test_event_and_result_from_same_tool_call_are_not_independent_validation() -> None:
    packet = _packet(status="success", validation_status="success")
    packet.selected_refs["tool_event"][0].metadata["tool_use_id"] = "call_one"
    packet.selected_refs["tool_event"][1].metadata["tool_use_id"] = "call_one"
    analysis = _analysis(task_completed=True, with_contract=True)

    decisions = AnalyzerDecisionAdapter().from_analysis(analysis, packet)

    assert decisions[0].proposed_action == "NOOP"
    assert decisions[0].noop_reason == "procedure_validation_observation_overlap"


def test_no_analyzer_suggestion_has_no_generic_capture_fallback() -> None:
    packet = _packet(status="success", validation_status="success")
    analysis = ExecutionAnalysis(
        task_id="task_test",
        timestamp=datetime.now(),
        task_completed=True,
        execution_note="Task completed successfully.",
        evolution_suggestions=[],
    )

    assert AnalyzerDecisionAdapter().from_analysis(analysis, packet) == []


def _analysis(*, task_completed: bool, with_contract: bool) -> ExecutionAnalysis:
    return ExecutionAnalysis(
        task_id="task_test",
        timestamp=datetime.now(),
        task_completed=task_completed,
        execution_note="The overall task may be incomplete, but extraction passed.",
        evolution_suggestions=[
            EvolutionSuggestion(
                evolution_type=EvolutionType.CAPTURED,
                direction="Capture only the validated extraction subworkflow.",
                capture_contract=(
                    CaptureContract(
                        capability="Extract one file and verify its bytes.",
                        preconditions=["The input archive is readable."],
                        procedure_refs=["tool:procedure"],
                        validation_refs=["tool:validation"],
                        validation_summary=(
                            "A separate checksum command matched the expected bytes."
                        ),
                        limitations=["No evidence covers encrypted archives."],
                    )
                    if with_contract
                    else None
                ),
            )
        ],
    )


def _packet(*, status: str, validation_status: str) -> EvidencePacket:
    return EvidencePacket(
        packet_id="pkt_test",
        trigger_job_id="trg_test",
        packet_type="analysis",
        profile_name="analysis_current_task",
        subprofile="task_finished",
        manifest_watermark=1,
        scope=EvidenceScope(source_task_ids=("task_test",)),
        selected_refs={
            "runtime_snapshot": [
                ResourceRef(
                    ref_id="runtime:finish",
                    ref_type="runtime_snapshot",
                    preview="Runtime status.",
                    metadata={"status": status},
                )
            ],
            "tool_event": [
                ResourceRef(
                    ref_id="tool:procedure",
                    ref_type="tool_event",
                    preview="Wrote the extracted bytes.",
                    metadata={"tool_name": "bash", "status": "success"},
                ),
                ResourceRef(
                    ref_id="tool:validation",
                    ref_type="tool_event",
                    preview="Checked the extracted checksum.",
                    metadata={
                        "tool_name": "bash",
                        "status": validation_status,
                    },
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
