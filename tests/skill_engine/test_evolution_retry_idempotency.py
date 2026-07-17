import asyncio
from datetime import datetime
from types import SimpleNamespace

from openspace.skill_engine.decision.analysis_adapter import AnalyzerDecisionAdapter
from openspace.skill_engine.evidence import EvidenceStore
from openspace.skill_engine.evidence.types import (
    EvidencePacket,
    EvidenceScope,
    PacketBudget,
    ResourceRef,
)
from openspace.skill_engine.evolution.engine import EvolutionEngine
from openspace.skill_engine.types import (
    CaptureContract,
    EvolutionSuggestion,
    EvolutionType,
    ExecutionAnalysis,
)


def _packet(packet_id: str, trigger_job_id: str = "trg_retry") -> EvidencePacket:
    return EvidencePacket(
        packet_id=packet_id,
        trigger_job_id=trigger_job_id,
        packet_type="analysis",
        profile_name="analysis_current_task",
        subprofile="task_finished",
        manifest_watermark=1,
        scope=EvidenceScope(task_id="task_retry", source_task_ids=("task_retry",)),
        selected_refs={
            "runtime_snapshot": [
                ResourceRef(
                    ref_id="runtime:finish",
                    ref_type="runtime_snapshot",
                    task_id="task_retry",
                    preview="Task completed successfully.",
                    metadata={"status": "success"},
                )
            ],
            "tool_event": [
                ResourceRef(
                    ref_id="tool:git",
                    ref_type="tool_event",
                    task_id="task_retry",
                    preview="Verified the reusable workflow.",
                    metadata={"tool_name": "bash", "status": "success"},
                ),
                ResourceRef(
                    ref_id="tool:verify",
                    ref_type="tool_event",
                    task_id="task_retry",
                    preview="Independent verification passed.",
                    metadata={"tool_name": "bash", "status": "success"},
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


def _analysis(suggestions: list[EvolutionSuggestion]) -> ExecutionAnalysis:
    return ExecutionAnalysis(
        task_id="task_retry",
        timestamp=datetime.now(),
        task_completed=True,
        execution_note="Completed and verified.",
        evolution_suggestions=suggestions,
    )


def test_decision_identity_survives_packet_rebuild_and_suggestion_reordering() -> None:
    first = EvolutionSuggestion(
        evolution_type=EvolutionType.CAPTURED,
        local_category_path="local/workflow/git-recovery",
        direction="Capture the exact dangling-commit recovery workflow.",
        capture_contract=_capture_contract("Recover a dangling commit."),
    )
    second = EvolutionSuggestion(
        evolution_type=EvolutionType.CAPTURED,
        local_category_path="local/tool-guide/read-before-write",
        direction="Capture the exact read-before-write guard workflow.",
        capture_contract=_capture_contract("Read a file before overwriting it."),
    )
    adapter = AnalyzerDecisionAdapter()

    initial = adapter.from_analysis(
        _analysis([first, second]),
        _packet("pkt_initial"),
    )
    retried = adapter.from_analysis(
        _analysis([second, first]),
        _packet("pkt_retry"),
    )

    initial_ids = {item.reason_summary: item.decision_id for item in initial}
    retried_ids = {item.reason_summary: item.decision_id for item in retried}
    assert retried_ids == initial_ids


def _capture_contract(capability: str) -> CaptureContract:
    return CaptureContract(
        capability=capability,
        procedure_refs=["tool:git"],
        validation_refs=["tool:verify"],
        validation_summary="A separate verification command passed.",
    )


def test_evidence_store_loads_only_committed_action_for_decision(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.db")
    try:
        failed = store.begin_action(
            action_id="act_failed",
            decision_id="dec_retry",
            trigger_job_id="trg_retry",
            authoring_id="auth_failed",
            validation_id="val_failed",
            action_type="CAPTURED",
            staging_dir="/tmp/stage-failed",
            active_target_dir="/tmp/failed",
        )
        store.finalize_action(failed.action_id, status="failed")
        assert store.load_committed_action_for_decision("dec_retry") is None

        committed = store.begin_action(
            action_id="act_committed",
            decision_id="dec_retry",
            trigger_job_id="trg_retry",
            authoring_id="auth_committed",
            validation_id="val_committed",
            action_type="CAPTURED",
            skill_id="git-recovery__v0_test",
            staging_dir="/tmp/stage-committed",
            active_target_dir="/tmp/committed",
        )
        store.finalize_action(
            committed.action_id,
            status="committed",
            skill_id="git-recovery__v0_test",
        )

        loaded = store.load_committed_action_for_decision("dec_retry")
        assert loaded is not None
        assert loaded.action_id == "act_committed"
        assert loaded.commit_status == "committed"
    finally:
        store.close()


def test_engine_reuses_committed_action_without_reauthoring() -> None:
    decision = SimpleNamespace(
        decision_id="dec_retry",
        trigger_job_id="trg_retry",
        proposed_action="CAPTURED",
        admission={"outcome": "direct"},
    )
    existing_action = SimpleNamespace(
        action_id="act_committed",
        decision_id="dec_retry",
        trigger_job_id="trg_retry",
        commit_status="committed",
        skill_id="git-recovery__v0_test",
    )

    class FakeEvidenceStore:
        def load_committed_action_for_decision(self, decision_id: str):
            assert decision_id == "dec_retry"
            return existing_action

    class ExplodingAuthoringBackend:
        called = False

        async def author_from_action_packet(self, *args, **kwargs):
            self.called = True
            raise AssertionError("committed decisions must not be re-authored")

    authoring = ExplodingAuthoringBackend()
    engine = EvolutionEngine(
        authoring_backend=authoring,
        committer=SimpleNamespace(
            evidence_store=FakeEvidenceStore(),
            skill_store=SimpleNamespace(load_record=lambda _skill_id: None),
        ),
        evolution_mode="autonomous",
    )
    job = SimpleNamespace(
        job_id="trg_retry",
        status="running",
        packet=SimpleNamespace(decisions=[decision]),
    )

    result = asyncio.run(engine.process_job(job))

    assert result.status == "completed"
    assert result.actions == [existing_action]
    assert result.errors == []
    assert authoring.called is False
