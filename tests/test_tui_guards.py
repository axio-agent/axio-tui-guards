"""Tests for axio_tui_guards — PathGuard and LLMGuard."""

from __future__ import annotations

from typing import Any

import pytest
from axio.blocks import TextBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.exceptions import GuardError
from axio.messages import Message
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio_tools_local.read_file import ReadFile
from axio_tools_local.shell import Shell
from axio_tools_local.write_file import WriteFile
from axio_tui.tools import Confirm
from axio_tui_guards.guards import LLMGuard, PathGuard


class TestPathGuard:
    @staticmethod
    async def _allow(_msg: str) -> str:
        return "y"

    @staticmethod
    async def _deny(_msg: str) -> str:
        return "n"

    @staticmethod
    async def _always_deny(_msg: str) -> str:
        return "deny"

    async def test_allow_grants_directory(self) -> None:
        guard = PathGuard(prompt_fn=self._allow)
        handler = ReadFile(filename="/tmp/test/file.txt")
        result = await guard.check(handler)
        assert result is handler
        # Same directory should be auto-allowed now
        handler2 = ReadFile(filename="/tmp/test/other.txt")
        result2 = await guard.check(handler2)
        assert result2 is handler2

    async def test_deny_raises_guard_error(self) -> None:
        guard = PathGuard(prompt_fn=self._deny)
        handler = ReadFile(filename="/secret/file.txt")
        with pytest.raises(GuardError, match="denied"):
            await guard.check(handler)

    async def test_always_deny_persists(self) -> None:
        guard = PathGuard(prompt_fn=self._always_deny)
        handler = ReadFile(filename="/deny/file.txt")
        with pytest.raises(GuardError):
            await guard.check(handler)
        # Second call should also be denied without prompting
        with pytest.raises(GuardError, match="denied"):
            await guard.check(handler)

    async def test_no_path_field_passes_through(self) -> None:
        guard = PathGuard(prompt_fn=self._deny)
        handler = Confirm(verdict="SAFE", reason="ok", category="test")
        result = await guard.check(handler)
        assert result is handler

    async def test_shell_extracts_cwd(self) -> None:
        guard = PathGuard(prompt_fn=self._allow)
        handler = Shell(command="ls", cwd="/tmp/project")
        result = await guard.check(handler)
        assert result is handler
        assert "/tmp/project" in guard.allowed

    async def test_subdirectory_auto_allowed(self) -> None:
        guard = PathGuard(prompt_fn=self._allow)
        handler = WriteFile(file_path="/home/user/project/src/main.py", content="x")
        await guard.check(handler)
        # Child path in same tree should be auto-allowed
        handler2 = WriteFile(file_path="/home/user/project/src/lib/util.py", content="y")
        result = await guard.check(handler2)
        assert result is handler2


class TestLLMGuard:
    async def test_safe_verdict_passes(self) -> None:
        confirm_input = {"verdict": "SAFE", "reason": "harmless", "category": "read"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("ok"),
            ]
        )
        agent_from_transport = _make_guard_agent(transport)
        guard = LLMGuard(agent_from_transport, MemoryContextStore())
        handler = ReadFile(filename="test.txt")
        result = await guard.check(handler)
        assert result is handler

    async def test_deny_verdict_raises(self) -> None:
        confirm_input = {"verdict": "DENY", "reason": "malicious", "category": "exec"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("blocked"),
            ]
        )
        agent = _make_guard_agent(transport)
        guard = LLMGuard(agent, MemoryContextStore())
        handler = Shell(command="rm -rf /")
        with pytest.raises(GuardError, match="DENIED"):
            await guard.check(handler)

    async def test_risky_verdict_user_allows(self) -> None:
        confirm_input = {"verdict": "RISKY", "reason": "writes file", "category": "write"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("review needed"),
            ]
        )
        agent = _make_guard_agent(transport)

        async def allow(_msg: str) -> str:
            return "y"

        guard = LLMGuard(agent, MemoryContextStore(), prompt_fn=allow)
        handler = WriteFile(file_path="out.txt", content="data")
        result = await guard.check(handler)
        assert result is handler

    async def test_risky_verdict_user_denies(self) -> None:
        confirm_input = {"verdict": "RISKY", "reason": "dangerous", "category": "exec"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("needs review"),
            ]
        )
        agent = _make_guard_agent(transport)

        async def deny(_msg: str) -> str:
            return "n"

        guard = LLMGuard(agent, MemoryContextStore(), prompt_fn=deny)
        handler = Shell(command="sudo reboot")
        with pytest.raises(GuardError, match="denied"):
            await guard.check(handler)

    async def test_always_allows_category(self) -> None:
        confirm_input = {"verdict": "RISKY", "reason": "writes", "category": "write_file"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("ok"),
                # Second call — should not reach transport since category is pre-approved
            ]
        )
        agent = _make_guard_agent(transport)

        async def always(_msg: str) -> str:
            return "always"

        guard = LLMGuard(agent, MemoryContextStore(), prompt_fn=always)
        handler = WriteFile(file_path="a.txt", content="data")
        await guard.check(handler)
        assert "write_file" in guard.allowed

    async def test_extract_confirm_from_context(self) -> None:
        guard = LLMGuard.__new__(LLMGuard)
        ctx = MemoryContextStore()
        await ctx.append(Message(role="user", content=[TextBlock(text="check this")]))
        await ctx.append(
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="c1", name="confirm", input={"verdict": "SAFE", "reason": "ok", "category": "read"}
                    ),
                ],
            )
        )
        confirm = await guard.extract_confirm(ctx)
        assert confirm.verdict == "SAFE"
        assert confirm.category == "read"

    async def test_extract_confirm_no_verdict(self) -> None:
        guard = LLMGuard.__new__(LLMGuard)
        ctx = MemoryContextStore()
        await ctx.append(Message(role="user", content=[TextBlock(text="check")]))
        await ctx.append(Message(role="assistant", content=[TextBlock(text="no tool call")]))
        confirm = await guard.extract_confirm(ctx)
        assert confirm.verdict == "SAFE"
        assert confirm.reason == "No verdict provided"


def _make_guard_agent(transport: Any) -> Any:
    from axio.agent import Agent
    from axio.tool import Tool
    from axio_tools_local.list_files import ListFiles
    from axio_tools_local.read_file import ReadFile
    from axio_tui.tools import Confirm, StatusLine

    return Agent(
        system="You are a safety classifier.",
        tools=[
            Tool(name="status_line", description="Set status", handler=StatusLine),
            Tool(name="read_file", description="Read file", handler=ReadFile),
            Tool(name="list", description="List files", handler=ListFiles),
            Tool(name="confirm", description="Submit verdict", handler=Confirm),
        ],
        transport=transport,
        max_iterations=5,
    )
