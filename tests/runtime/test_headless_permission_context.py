from types import SimpleNamespace

from openspace.runtime.app import OpenSpaceRuntime


class _GroundingAgent:
    def __init__(self) -> None:
        self.tui_bridge = object()
        self.runtime_event_sink = None

    def set_tui_bridge(self, bridge: object | None) -> None:
        self.tui_bridge = bridge

    def set_runtime_event_sink(self, sink: object | None) -> None:
        self.runtime_event_sink = sink


def _runtime_with_bridge(tui_bridge: object | None) -> SimpleNamespace:
    agent = _GroundingAgent()
    state = SimpleNamespace(
        grounding_agent=agent,
        event_proxy=object(),
        tui_bridge=tui_bridge,
        llm_client=None,
        multi_agent=None,
    )
    return SimpleNamespace(
        state=state,
        scheduler=None,
        emit_runtime_event=object(),
    )


def test_headless_runtime_does_not_advertise_event_proxy_as_tui() -> None:
    runtime = _runtime_with_bridge(None)

    OpenSpaceRuntime.propagate_service_hooks(runtime)

    assert runtime.state.grounding_agent.tui_bridge is None
    assert runtime.state.grounding_agent.runtime_event_sink is runtime.emit_runtime_event


def test_interactive_runtime_keeps_event_proxy_bridge() -> None:
    runtime = _runtime_with_bridge(object())

    OpenSpaceRuntime.propagate_service_hooks(runtime)

    assert runtime.state.grounding_agent.tui_bridge is runtime.state.event_proxy
