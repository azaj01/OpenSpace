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
from openspace.skill_engine.analyzer import ExecutionAnalyzer


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
                "enable_thinking": kwargs.get("enable_thinking"),
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


def _analysis_json() -> str:
    return (
        '{"task_completed":true,"execution_note":"Verified completion",'
        '"tool_issues":[],"skill_judgments":[],'
        '"skill_phase_failed_skill_ids":[],"evolution_suggestions":[],'
        '"analyzed_by":"test-model"}'
    )


def _analyzer(
    client: FakeLLMClient,
    *,
    max_tokens: int | None = None,
) -> ExecutionAnalyzer:
    return ExecutionAnalyzer(
        store=SimpleNamespace(),
        llm_client=client,
        max_tokens=max_tokens,
    )


def test_length_response_retries_with_compact_tool_free_json(monkeypatch) -> None:
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
    client = FakeLLMClient(
        [
            _response("", stop_reason="length", output_tokens=4096),
            _response(_analysis_json()),
        ]
    )

    result = asyncio.run(
        _analyzer(client, max_tokens=8192)._run_analysis_loop(
            "Analyze this execution.",
            available_tools=[object()],
        )
    )

    assert result is not None
    assert result["task_completed"] is True
    assert len(client.calls) == 2
    assert client.calls[0]["tools"] is not None
    assert client.calls[1]["tools"] is None
    assert client.calls[0]["enable_thinking"] is None
    assert client.calls[1]["enable_thinking"] is False
    assert {call["max_tokens"] for call in client.calls} == {8192}
    assert any(
        message.get("_meta", {}).get("type")
        == "analysis_max_output_tokens_recovery"
        for message in client.calls[1]["messages"]
    )

    first_metadata = iteration_recorder.await_args_list[0].kwargs[
        "response_metadata"
    ]
    assert first_metadata["stop_reason"] == "length"
    assert first_metadata["output_tokens"] == 4096
    assert first_metadata["content_length"] == 0
    assert first_metadata["has_api_error"] is True
    assert first_metadata["length_recovery_attempt"] == 1


def test_repeated_analysis_length_response_stops_after_one_recovery(
    monkeypatch,
) -> None:
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
        _analyzer(client)._run_analysis_loop("Analyze this execution.")
    )

    assert result is None
    assert len(client.calls) == 2
    assert client.calls[0]["enable_thinking"] is None
    assert client.calls[1]["enable_thinking"] is False


def test_invalid_json_response_retries_without_thinking(monkeypatch) -> None:
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
            _response("I will summarize the execution now."),
            _response(_analysis_json()),
        ]
    )

    result = asyncio.run(
        _analyzer(client, max_tokens=8192)._run_analysis_loop(
            "Analyze this execution."
        )
    )

    assert result is not None
    assert result["task_completed"] is True
    assert len(client.calls) == 2
    assert client.calls[0]["enable_thinking"] is None
    assert client.calls[1]["enable_thinking"] is False
    assert any(
        message.get("_meta", {}).get("type")
        == "analysis_invalid_json_recovery"
        for message in client.calls[1]["messages"]
    )


def test_analysis_refusal_is_recorded_and_not_retried(monkeypatch) -> None:
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
        _analyzer(client)._run_analysis_loop("Analyze this execution.")
    )

    assert result is None
    assert len(client.calls) == 1
    metadata = iteration_recorder.await_args.kwargs["response_metadata"]
    assert metadata["stop_reason"] == "refusal"
    assert metadata["has_api_error"] is True
    assert metadata["length_recovery_attempt"] == 0


def test_execution_analyzer_max_tokens_can_be_configured_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENSPACE_EXECUTION_ANALYZER_MAX_TOKENS", "12288")

    config = OpenSpaceConfig(execution_analyzer_max_tokens=8192)

    assert config.execution_analyzer_max_tokens == 12288
