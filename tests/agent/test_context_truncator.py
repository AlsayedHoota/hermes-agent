"""Tests for the truncation-only context engine (agent/context_truncator.py).

Tests cover:
  - Basic truncation behavior
  - System prompt preservation
  - Tail message preservation with protect_last_n
  - Tool result pruning
  - Tool call/result pair integrity (no orphans)
  - Truncation marker injection
  - Edge cases (empty, small conversations)
  - should_compress / should_compress_preflight
  - update_model
"""

import pytest

from agent.context_truncator import (
    ContextTruncator,
    TRUNCATION_MARKER,
    _estimate_msg_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def truncator():
    """Small context window for easy threshold testing."""
    return ContextTruncator(
        model="test-model",
        threshold_percent=0.30,
        target_percent=0.20,
        protect_last_n=5,
        quiet_mode=True,
        config_context_length=10000,  # threshold=3000, target=2000
    )


@pytest.fixture
def large_truncator():
    """Realistic 200K context window."""
    return ContextTruncator(
        model="test-model",
        threshold_percent=0.30,
        target_percent=0.20,
        protect_last_n=15,
        quiet_mode=True,
        config_context_length=200000,
    )


def _make_exchange(idx, user_len=400, tool_len=400, asst_len=300):
    """Create a user -> assistant(tool_call) -> tool -> assistant exchange."""
    return [
        {"role": "user", "content": f"User message {idx}: " + "x" * user_len},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"cmd":"test"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{idx}",
            "content": f"Result {idx}: " + "y" * tool_len,
        },
        {"role": "assistant", "content": f"Response {idx}: " + "z" * asst_len},
    ]


def _make_conversation(n_exchanges, system_len=100, **kwargs):
    """Build a system + N exchanges conversation."""
    msgs = [{"role": "system", "content": "System prompt. " * (system_len // 15)}]
    for i in range(n_exchanges):
        msgs.extend(_make_exchange(i, **kwargs))
    return msgs


def _check_no_orphans(messages):
    """Verify all tool_calls have matching tool results and vice versa."""
    call_ids = set()
    result_ids = set()
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                call_ids.add(tc.get("id", ""))
        if msg.get("role") == "tool":
            result_ids.add(msg.get("tool_call_id", ""))
    orphans = (call_ids - result_ids) | (result_ids - call_ids)
    assert len(orphans) == 0, f"Orphaned tool IDs: {orphans}"


# ---------------------------------------------------------------------------
# Tests: Basic behavior
# ---------------------------------------------------------------------------

class TestTruncatorInit:
    def test_name(self, truncator):
        assert truncator.name == "truncator"

    def test_thresholds(self, truncator):
        assert truncator.context_length == 10000
        assert truncator.threshold_tokens == 3000
        assert truncator.target_tokens == 2000

    def test_protect_last_n_floor(self):
        t = ContextTruncator(
            model="test", protect_last_n=1, quiet_mode=True,
            config_context_length=10000,
        )
        assert t.protect_last_n == 3  # minimum floor

    def test_compression_count_starts_zero(self, truncator):
        assert truncator.compression_count == 0


class TestNoTruncationNeeded:
    def test_short_conversation_unchanged(self, truncator):
        msgs = _make_conversation(2)
        result = truncator.compress(msgs, current_tokens=500)
        assert len(result) == len(msgs)

    def test_two_messages_unchanged(self, truncator):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = truncator.compress(msgs, current_tokens=5000)
        assert result == msgs

    def test_empty_messages(self, truncator):
        result = truncator.compress([], current_tokens=5000)
        assert result == []

    def test_single_message(self, truncator):
        msgs = [{"role": "system", "content": "sys"}]
        result = truncator.compress(msgs, current_tokens=5000)
        assert result == msgs


class TestTruncation:
    def test_drops_old_messages(self, truncator):
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        assert len(result) < len(msgs)

    def test_system_prompt_preserved(self, truncator):
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        assert result[0]["role"] == "system"
        assert "System prompt" in result[0]["content"]

    def test_truncation_marker_present(self, truncator):
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        marker_found = any(
            TRUNCATION_MARKER in (m.get("content") or "") for m in result
        )
        assert marker_found, "Truncation marker not found in output"

    def test_marker_is_second_message(self, truncator):
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        assert TRUNCATION_MARKER in result[1]["content"]

    def test_recent_messages_preserved(self, truncator):
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        # The last assistant message should be preserved exactly
        last_original = msgs[-1]["content"]
        last_result = result[-1]["content"]
        assert last_original == last_result

    def test_compression_count_increments(self, truncator):
        msgs = _make_conversation(20)
        assert truncator.compression_count == 0
        truncator.compress(msgs, current_tokens=4000)
        assert truncator.compression_count == 1
        truncator.compress(msgs, current_tokens=4000)
        assert truncator.compression_count == 2


class TestProtectLastN:
    def test_minimum_tail_messages(self, truncator):
        """At least protect_last_n messages should survive."""
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        # Subtract system prompt and marker
        tail_count = len(result) - 2  # system + marker
        assert tail_count >= truncator.protect_last_n

    def test_protect_last_n_respected_even_over_budget(self):
        """protect_last_n should be respected even if it exceeds token budget."""
        t = ContextTruncator(
            model="test",
            threshold_percent=0.30,
            target_percent=0.05,  # very tight budget
            protect_last_n=10,
            quiet_mode=True,
            config_context_length=10000,
        )
        msgs = _make_conversation(20, user_len=200, tool_len=200, asst_len=150)
        result = t.compress(msgs, current_tokens=4000)
        tail_count = len(result) - 2
        assert tail_count >= 10


class TestToolIntegrity:
    def test_no_orphaned_tool_calls(self, truncator):
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        _check_no_orphans(result)

    def test_tool_groups_not_split(self, truncator):
        """A tool result should never appear without its preceding tool_call."""
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        for i, msg in enumerate(result):
            if msg.get("role") == "tool":
                # Find the matching assistant with tool_calls before it
                found = False
                for j in range(i - 1, -1, -1):
                    if result[j].get("tool_calls"):
                        for tc in result[j]["tool_calls"]:
                            if isinstance(tc, dict) and tc.get("id") == msg.get("tool_call_id"):
                                found = True
                                break
                    if found:
                        break
                assert found, f"Tool result at index {i} has no matching tool_call"


class TestToolPruning:
    def test_old_tool_results_pruned(self, truncator):
        msgs = _make_conversation(20, tool_len=1000)
        result = truncator.compress(msgs, current_tokens=4000)
        # Old tool results (not in last protect_last_n) should be pruned
        pruned_count = sum(
            1
            for m in result
            if m.get("role") == "tool"
            and m.get("content") == "[Old tool output cleared to save context space]"
        )
        # At least some should be pruned (the ones outside protect_last_n)
        total_tools = sum(1 for m in result if m.get("role") == "tool")
        if total_tools > truncator.protect_last_n:
            assert pruned_count > 0

    def test_recent_tool_results_not_pruned(self, truncator):
        msgs = _make_conversation(20, tool_len=500)
        result = truncator.compress(msgs, current_tokens=4000)
        # The very last tool result should NOT be pruned
        last_tool = None
        for m in reversed(result):
            if m.get("role") == "tool":
                last_tool = m
                break
        if last_tool:
            assert last_tool["content"] != "[Old tool output cleared to save context space]"


class TestMarkerRole:
    def test_marker_role_differs_from_next(self, truncator):
        """Marker role should not match the first tail message role."""
        msgs = _make_conversation(20)
        result = truncator.compress(msgs, current_tokens=4000)
        if len(result) > 2:
            marker = result[1]
            next_msg = result[2]
            # They should have different roles (to avoid consecutive same-role)
            assert marker["role"] != next_msg["role"], (
                f"Marker role '{marker['role']}' same as next '{next_msg['role']}'"
            )


class TestShouldCompress:
    def test_below_threshold(self, truncator):
        assert not truncator.should_compress(2000)

    def test_above_threshold(self, truncator):
        assert truncator.should_compress(4000)

    def test_at_threshold(self, truncator):
        assert truncator.should_compress(3000)

    def test_preflight_small_conversation(self, truncator):
        msgs = _make_conversation(2)
        assert not truncator.should_compress_preflight(msgs)


class TestUpdateModel:
    def test_update_recalculates(self, truncator):
        truncator.update_model(
            model="new-model",
            context_length=50000,
            base_url="",
            api_key="",
            provider="test",
        )
        assert truncator.context_length == 50000
        assert truncator.threshold_tokens == 15000  # 30% of 50K
        assert truncator.target_tokens == 10000  # 20% of 50K


class TestUpdateFromResponse:
    def test_updates_token_tracking(self, truncator):
        truncator.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 500,
            "total_tokens": 5500,
        })
        assert truncator.last_prompt_tokens == 5000
        assert truncator.last_completion_tokens == 500
        assert truncator.last_total_tokens == 5500


class TestEstimateMsgTokens:
    def test_simple_message(self):
        msg = {"role": "user", "content": "hello world"}
        tokens = _estimate_msg_tokens(msg)
        assert tokens > 0

    def test_tool_calls_counted(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"arguments": '{"x": 1}' * 100}},
            ],
        }
        tokens = _estimate_msg_tokens(msg)
        assert tokens > 100  # the arguments should add significant tokens

    def test_empty_content(self):
        msg = {"role": "assistant", "content": ""}
        tokens = _estimate_msg_tokens(msg)
        assert tokens >= 10  # at least the metadata overhead


class TestNoSystemPrompt:
    """Edge case: conversation without a system prompt."""

    def test_works_without_system(self):
        t = ContextTruncator(
            model="test", quiet_mode=True, config_context_length=5000,
            protect_last_n=3,
        )
        msgs = [
            {"role": "user", "content": "hello " + "x" * 1000},
            {"role": "assistant", "content": "world " + "y" * 1000},
            {"role": "user", "content": "recent " + "x" * 1000},
            {"role": "assistant", "content": "latest " + "y" * 1000},
        ]
        result = t.compress(msgs, current_tokens=2000)
        # Should still work and keep recent messages
        assert len(result) >= 2
        assert result[-1]["content"].startswith("latest")
