import asyncio
from pathlib import Path
from types import SimpleNamespace

from openspace.grounding.core.permissions import PermissionAllow
from openspace.grounding.backends.shell.file_tools import ReadFileTool
from openspace.grounding.core.permissions import bash_path_validation
from openspace.grounding.core.permissions.bash_permissions import (
    _check_permission_mode,
    bash_tool_check_permission,
)
from openspace.grounding.core.permissions.bash_path_validation import validate_path
from openspace.grounding.core.permissions.loader import (
    load_tool_permission_context,
)
from openspace.grounding.core.security.shell_parser import ExtractRedirectResult


def test_explicit_bypass_mode_is_available_to_bash(tmp_path: Path) -> None:
    context = load_tool_permission_context(
        str(tmp_path),
        mode="bypassPermissions",
    )

    decision = _check_permission_mode("systemctl status postfix", context)

    assert context.is_bypass_permissions_mode_available is True
    assert isinstance(decision, PermissionAllow)


def test_default_mode_does_not_enable_bypass(tmp_path: Path) -> None:
    context = load_tool_permission_context(str(tmp_path), mode="default")

    assert context.is_bypass_permissions_mode_available is False


def test_bypass_mode_allows_ordinary_path_outside_workspace(
    tmp_path: Path,
) -> None:
    context = load_tool_permission_context(
        str(tmp_path),
        mode="bypassPermissions",
    )

    result = validate_path(
        "/etc/mailman3/mailman.cfg",
        str(tmp_path),
        context,
        "read",
    )

    assert result.allowed is True
    assert result.resolved_path == "/etc/mailman3/mailman.cfg"


def test_bypass_mode_allows_bash_read_outside_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = load_tool_permission_context(
        str(tmp_path),
        mode="bypassPermissions",
    )
    monkeypatch.setattr(
        bash_path_validation,
        "_extract_output_redirections",
        lambda command: ExtractRedirectResult(command, [], False),
    )

    decision = bash_tool_check_permission(
        "cat /etc/mailman3/mailman.cfg",
        str(tmp_path),
        context,
    )

    assert isinstance(decision, PermissionAllow)


def test_read_tool_allows_current_session_tool_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool_results = tmp_path / ".openspace" / "sessions" / "one" / "tool-results"
    tool_results.mkdir(parents=True)
    output = tool_results / "large-output.txt"
    output.write_text("runtime output", encoding="utf-8")
    context = SimpleNamespace(
        permission_context=load_tool_permission_context(
            str(workspace),
            mode="default",
        ),
        tool_results_dir=str(tool_results),
        cwd=str(workspace),
    )

    decision = asyncio.run(
        ReadFileTool().check_permissions(
            {"file_path": str(output)},
            context,
        )
    )

    assert isinstance(decision, PermissionAllow)
