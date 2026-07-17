import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace

from openspace.skill_engine.analyzer import ExecutionAnalyzer
from openspace.skill_engine.decision.analysis_adapter import AnalyzerDecisionAdapter
from openspace.skill_engine.evidence import EvidenceStore
from openspace.skill_engine.evidence.types import (
    EvidencePacket,
    EvidenceScope,
    PacketBudget,
    ResourceRef,
)
from openspace.skill_engine.evolution.admission import EvolutionAdmission
from openspace.skill_engine.evolution.authoring import SkillEvolverAuthoringBackend
from openspace.skill_engine.evolution.engine import _source_validation_passed
from openspace.skill_engine.evolution.validator import EvolutionValidator
from openspace.skill_engine.types import (
    CaptureContract,
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
    SkillCategory,
)


def test_analyzer_parses_capture_contract() -> None:
    parsed = ExecutionAnalyzer._parse_analysis(
        "task_test",
        {
            "task_completed": False,
            "execution_note": "A validated subworkflow completed.",
            "evolution_suggestions": [
                {
                    "type": "captured",
                    "target_skills": [],
                    "category": "workflow",
                    "direction": "Capture only extraction.",
                    "capture_contract": _contract(),
                }
            ],
        },
        {"selected_skills": []},
    )

    assert parsed is not None
    contract = parsed.evolution_suggestions[0].capture_contract
    assert contract is not None
    assert contract.capability == _contract()["capability"]
    assert contract.validation_refs == ["tool:validation"]


def test_capture_contract_round_trips_with_stored_analysis() -> None:
    suggestion = EvolutionSuggestion(
        evolution_type=EvolutionType.CAPTURED,
        category=SkillCategory.WORKFLOW,
        direction="Capture only extraction.",
        capture_contract=CaptureContract.from_dict(_contract()),
    )

    restored = EvolutionSuggestion.from_dict(suggestion.to_dict())

    assert restored.capture_contract is not None
    assert restored.capture_contract.to_dict() == _contract()


def test_contract_and_source_validation_are_durable(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.db")
    try:
        packet = _packet()
        store.persist_packet(packet)
        decision = AnalyzerDecisionAdapter().from_analysis(
            _analysis(),
            packet,
            source_analysis_id="analysis:test",
        )[0]
        store.persist_decision(decision, packet.packet_id)
        admission = EvolutionAdmission(evidence_store=store).admit(
            decision,
            packet,
        )

        conn = sqlite3.connect(store.db_path)
        try:
            raw_contract = conn.execute(
                "SELECT proposal_contract_json FROM decision_rationales "
                "WHERE decision_id=?",
                (decision.decision_id,),
            ).fetchone()[0]
            source_validated = conn.execute(
                "SELECT source_validation_passed FROM admission_results "
                "WHERE admission_id=?",
                (admission.admission_id,),
            ).fetchone()[0]
        finally:
            conn.close()

        assert json.loads(raw_contract) == _contract()
        assert source_validated == 1
        decision_ref = store.get_ref(f"decision:{decision.decision_id}")
        assert decision_ref is not None
        assert decision_ref.metadata["proposal_contract"] == _contract()
        loaded = store.load_admission(admission.admission_id)
        assert loaded is not None
        assert loaded.source_validation_passed is True
    finally:
        store.close()


def test_capture_authoring_prompt_is_bounded_by_contract() -> None:
    backend = object.__new__(SkillEvolverAuthoringBackend)

    prompt = backend._build_prompt(
        "CAPTURED",
        [],
        _packet(),
        "Capture only extraction.",
        category=SkillCategory.WORKFLOW,
        proposal_contract=_contract(),
    )

    assert "Admitted Capture Contract" in prompt
    assert _contract()["capability"] in prompt
    assert "Do not add unsupported algorithms" in prompt
    assert "No evidence covers encrypted archives" in prompt


def test_validator_rejects_contract_mutation_or_missing_provenance() -> None:
    validator = EvolutionValidator()
    decision = SimpleNamespace(proposal_contract=_contract())
    admission = SimpleNamespace(source_validation_passed=True)
    staged = SimpleNamespace(
        apply_metadata={"proposal_contract": _contract()},
        evidence_refs=["tool:procedure", "tool:validation"],
    )

    assert validator._capture_contract_failures(
        staged,
        "CAPTURED",
        decision,
        admission,
    ) == []

    changed = dict(_contract())
    changed["capability"] = "An unsupported broader capability."
    staged.apply_metadata = {"proposal_contract": changed}
    staged.evidence_refs = ["tool:procedure"]
    failures = validator._capture_contract_failures(
        staged,
        "CAPTURED",
        decision,
        admission,
    )

    assert "captured_contract_changed_during_authoring" in failures
    assert "captured_contract_refs_missing_from_provenance" in failures


def test_origin_trust_requires_source_validation_admission() -> None:
    assert _source_validation_passed(SimpleNamespace()) is False
    assert _source_validation_passed(
        SimpleNamespace(source_validation_passed=False)
    ) is False
    assert _source_validation_passed(
        SimpleNamespace(source_validation_passed=True)
    ) is True


def _analysis() -> ExecutionAnalysis:
    return ExecutionAnalysis(
        task_id="task_test",
        timestamp=datetime.now(),
        task_completed=False,
        execution_note="The overall task failed after extraction passed.",
        evolution_suggestions=[
            EvolutionSuggestion(
                evolution_type=EvolutionType.CAPTURED,
                category=SkillCategory.WORKFLOW,
                direction="Capture only extraction.",
                capture_contract=CaptureContract.from_dict(_contract()),
            )
        ],
    )


def _contract() -> dict:
    return {
        "capability": "Extract one file and verify its bytes.",
        "preconditions": ["The input archive is readable."],
        "procedure_refs": ["tool:procedure"],
        "validation_refs": ["tool:validation"],
        "validation_summary": "A separate checksum matched the expected bytes.",
        "limitations": ["No evidence covers encrypted archives."],
    }


def _packet() -> EvidencePacket:
    return EvidencePacket(
        packet_id="pkt_test",
        trigger_job_id="trg_test",
        packet_type="analysis",
        profile_name="analysis_current_task",
        subprofile="task_finished",
        manifest_watermark=1,
        scope=EvidenceScope(task_id="task_test", source_task_ids=("task_test",)),
        selected_refs={
            "runtime_snapshot": [
                ResourceRef(
                    ref_id="runtime:finish",
                    ref_type="runtime_snapshot",
                    task_id="task_test",
                    preview="Overall task failed after extraction.",
                    metadata={"status": "error"},
                )
            ],
            "tool_event": [
                ResourceRef(
                    ref_id="tool:procedure",
                    ref_type="tool_event",
                    task_id="task_test",
                    preview="Extracted the file.",
                    metadata={"status": "success", "tool_name": "bash"},
                ),
                ResourceRef(
                    ref_id="tool:validation",
                    ref_type="tool_event",
                    task_id="task_test",
                    preview="Checksum matched.",
                    metadata={"status": "success", "tool_name": "bash"},
                ),
            ],
        },
        expanded_snippets=[],
        readable_paths=[],
        instructions={},
        budget=PacketBudget(max_chars=1000, used_chars=100),
        redaction_status="ok",
        build_status="ok",
        missing_ref_types=[],
    )
