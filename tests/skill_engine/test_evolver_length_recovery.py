import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openspace.application import OpenSpaceConfig
from openspace.llm.client import LLMClient
from openspace.llm.types import ModelResponse, TokenUsage
from openspace.recording import RecordingManager
from openspace.services.conversation.messages import (
    build_assistant_api_error_message,
)
from openspace.skill_engine.evolver import (
    EvolutionContext,
    EvolutionTrigger,
    SkillEvolver,
)
from openspace.skill_engine.types import EvolutionSuggestion, EvolutionType


class FakeLLMClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.model = "test-model"
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def call_model_with_fallback(self, **kwargs) -> ModelResponse:
        self.calls.append(
            {
                "messages": copy.deepcopy(kwargs["messages"]),
                "tools": kwargs.get("tools"),
                "model": kwargs.get("model"),
                "max_tokens": kwargs.get("max_tokens"),
            }
        )
        return self._responses.pop(0)

    @staticmethod
    def get_model_response_followup_messages(
        response: ModelResponse,
    ) -> list[dict]:
        return LLMClient.get_model_response_followup_messages(response)

    @staticmethod
    def model_response_has_api_error(response: ModelResponse) -> bool:
        return LLMClient.model_response_has_api_error(response)


def _response(
    content: str,
    *,
    stop_reason: str = "stop",
    output_tokens: int = 100,
) -> ModelResponse:
    assistant_message = {"role": "assistant", "content": content}
    messages = [assistant_message]
    if stop_reason == "length":
        messages.append(
            build_assistant_api_error_message(
                "Response reached the maximum output token limit.",
                error_details="stop_reason=length, model=test-model",
            )
        )
    elif stop_reason in {"refusal", "content_filter"}:
        messages.append(
            build_assistant_api_error_message(
                "The model declined the request.",
                error_details=f"stop_reason={stop_reason}, model=test-model",
            )
        )
    return ModelResponse(
        assistant_message=assistant_message,
        tool_calls=[],
        tool_map={},
        stop_reason=stop_reason,
        usage=TokenUsage(
            input_tokens=200,
            output_tokens=output_tokens,
            total_tokens=200 + output_tokens,
        ),
        messages=messages,
        effective_model="test-model",
    )


def _complete_response() -> str:
    return (
        "---\n"
        "name: concise-workflow\n"
        "description: Apply a reusable workflow.\n"
        "---\n\n"
        "# Concise workflow\n\n"
        "1. Inspect the relevant state.\n"
        "2. Apply the smallest verified change.\n\n"
        "*** Begin Evolution Finalization\n"
        '{"status":"complete","change_summary":"Captured the workflow",'
        '"intent_spec":{"goal":"Apply the verified workflow"},'
        '"eval_plan":{"checks":["Confirm the expected result"]}}\n'
        "*** End Evolution Finalization"
    )


def _context() -> EvolutionContext:
    return EvolutionContext(
        trigger=EvolutionTrigger.ANALYSIS,
        suggestion=EvolutionSuggestion(
            evolution_type=EvolutionType.CAPTURED,
            direction="Capture a reusable verified workflow.",
        ),
        source_task_id="task_test",
        available_tools=[object()],
    )


def _evolver(
    client: FakeLLMClient,
    *,
    max_tokens: int | None = None,
) -> SkillEvolver:
    return SkillEvolver(
        store=SimpleNamespace(),
        registry=SimpleNamespace(),
        llm_client=client,
        max_tokens=max_tokens,
    )


def test_length_response_keeps_remaining_rounds_tool_free_until_finalization(
    monkeypatch,
) -> None:
    setup_recorder = AsyncMock()
    iteration_recorder = AsyncMock()
    monkeypatch.setattr(
        RecordingManager,
        "record_conversation_setup",
        setup_recorder,
    )
    monkeypatch.setattr(
        RecordingManager,
        "record_iteration_context",
        iteration_recorder,
    )
    client = FakeLLMClient(
        [
            _response(
                "partial skill content", stop_reason="length", output_tokens=4096
            ),
            _response("I will now provide the concise replacement."),
            _response(_complete_response()),
        ]
    )

    result = asyncio.run(
        _evolver(client, max_tokens=8192)._run_evolution_loop(
            "Author a skill.",
            _context(),
        )
    )

    assert result is not None
    assert result.change_summary == "Captured the workflow"
    assert len(client.calls) == 3
    assert client.calls[0]["tools"] is not None
    assert client.calls[1]["tools"] is None
    assert client.calls[2]["tools"] is None
    assert {call["max_tokens"] for call in client.calls} == {8192}
    assert any(
        message.get("_meta", {}).get("type") == "evolution_max_output_tokens_recovery"
        for message in client.calls[1]["messages"]
    )

    first_metadata = iteration_recorder.await_args_list[0].kwargs["response_metadata"]
    assert first_metadata["stop_reason"] == "length"
    assert first_metadata["output_tokens"] == 4096
    assert first_metadata["content_length"] == len("partial skill content")
    assert first_metadata["has_api_error"] is True
    assert first_metadata["length_recovery_attempt"] == 1


def test_repeated_length_response_stops_after_one_recovery(monkeypatch) -> None:
    monkeypatch.setattr(
        RecordingManager,
        "record_conversation_setup",
        AsyncMock(),
    )
    monkeypatch.setattr(
        RecordingManager,
        "record_iteration_context",
        AsyncMock(),
    )
    client = FakeLLMClient(
        [
            _response("first partial", stop_reason="length", output_tokens=4096),
            _response("second partial", stop_reason="length", output_tokens=4096),
        ]
    )

    result = asyncio.run(
        _evolver(client)._run_evolution_loop("Author a skill.", _context())
    )

    assert result is None
    assert len(client.calls) == 2


def test_refusal_is_recorded_and_not_retried(monkeypatch) -> None:
    iteration_recorder = AsyncMock()
    monkeypatch.setattr(
        RecordingManager,
        "record_conversation_setup",
        AsyncMock(),
    )
    monkeypatch.setattr(
        RecordingManager,
        "record_iteration_context",
        iteration_recorder,
    )
    client = FakeLLMClient([_response("", stop_reason="refusal", output_tokens=0)])

    result = asyncio.run(
        _evolver(client)._run_evolution_loop("Author a skill.", _context())
    )

    assert result is None
    assert len(client.calls) == 1
    metadata = iteration_recorder.await_args.kwargs["response_metadata"]
    assert metadata["stop_reason"] == "refusal"
    assert metadata["has_api_error"] is True
    assert metadata["length_recovery_attempt"] == 0


def test_skill_evolver_max_tokens_can_be_configured_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENSPACE_SKILL_EVOLVER_MAX_TOKENS", "12288")

    config = OpenSpaceConfig(skill_evolver_max_tokens=8192)

    assert config.skill_evolver_max_tokens == 12288
