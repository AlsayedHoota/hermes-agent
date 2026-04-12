"""Truncation-only context engine — drops old messages without summarization.

Instead of summarizing middle turns with an LLM (lossy, slow, expensive),
this engine simply removes the oldest non-system messages when the context
window fills up.  The most recent messages are preserved exactly, giving
the agent perfect continuity of its current work.

The agent can always use session_search to recall earlier context on demand.
Memory, skills, and the todo list are injected separately and survive
truncation automatically.

Selection: set ``context.engine: truncator`` in config.yaml.

Algorithm:
  1. Prune old tool result bodies (cheap pre-pass, same as compressor)
  2. Keep system prompt (index 0) always
  3. Fill remaining budget with the most recent messages (walk backward)
  4. Insert a static truncation marker so the agent knows context was lost
  5. Respect protect_last_n as a minimum floor for tail messages
"""

import logging
from typing import Any, Dict, List

from agent.context_engine import ContextEngine
from agent.model_metadata import (
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)

# Placeholder for pruned tool outputs (same as compressor for consistency)
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Rough chars-per-token estimate (same as compressor)
_CHARS_PER_TOKEN = 4

# Static marker injected when messages are truncated
TRUNCATION_MARKER = (
    "[CONTEXT TRUNCATION] Earlier messages in this conversation were removed "
    "to free context space. No summary was generated — use session_search to "
    "recall earlier details if needed. Save important discoveries to memory "
    "or skills so they persist across truncations."
)


def _estimate_msg_tokens(msg: Dict[str, Any]) -> int:
    """Rough token estimate for a single message."""
    content = msg.get("content") or ""
    tokens = len(content) // _CHARS_PER_TOKEN + 10  # +10 for role/metadata
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            args = tc.get("function", {}).get("arguments", "")
            tokens += len(args) // _CHARS_PER_TOKEN
    return tokens


class ContextTruncator(ContextEngine):
    """Drop-oldest context engine — no LLM summarization.

    Keeps the system prompt + as many recent messages as fit within
    the target token budget.  Fast, cheap, and lossless for the tail.
    """

    @property
    def name(self) -> str:
        return "truncator"

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.30,
        target_percent: float = 0.20,
        protect_last_n: int = 15,
        quiet_mode: bool = False,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.threshold_percent = threshold_percent
        self.target_percent = target_percent
        self.protect_last_n = max(3, protect_last_n)  # floor of 3
        self.quiet_mode = quiet_mode

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        self.threshold_tokens = int(self.context_length * threshold_percent)
        self.target_tokens = int(self.context_length * target_percent)
        self.compression_count = 0

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

        if not quiet_mode:
            logger.info(
                "Context truncator initialized: model=%s context=%d "
                "threshold=%d (%.0f%%) target=%d (%.0f%%) protect_last_n=%d",
                model, self.context_length,
                self.threshold_tokens, threshold_percent * 100,
                self.target_tokens, target_percent * 100,
                self.protect_last_n,
            )

    # -- ContextEngine interface -------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens or self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        rough = estimate_messages_tokens_rough(messages)
        return rough >= self.threshold_tokens

    def on_session_reset(self) -> None:
        super().on_session_reset()

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
        self.target_tokens = int(context_length * self.target_percent)

    # -- Core truncation ---------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        """Truncate old messages to fit within the target token budget.

        Algorithm:
          1. Prune old tool result bodies (cheap, no LLM)
          2. Keep system prompt (index 0)
          3. Walk backward from end, accumulating recent messages
             until we fill the target budget (minus system prompt)
          4. Insert a truncation marker between system prompt and tail
          5. Respect protect_last_n as minimum tail floor
        """
        n = len(messages)
        if n <= 2:
            return messages

        display_tokens = (
            current_tokens or self.last_prompt_tokens
            or estimate_messages_tokens_rough(messages)
        )

        # Phase 1: Prune old tool result bodies (cheap pre-pass).
        # Prune everything except the last protect_last_n messages.
        pruned_messages = self._prune_tool_results(messages)

        # Phase 2: Identify system prompt
        system_msgs = []
        rest_start = 0
        if pruned_messages and pruned_messages[0].get("role") == "system":
            system_msgs = [pruned_messages[0].copy()]
            rest_start = 1

        system_tokens = sum(_estimate_msg_tokens(m) for m in system_msgs)
        tail_budget = self.target_tokens - system_tokens
        if tail_budget < 1000:
            # System prompt is huge — give at least 1000 tokens to tail
            tail_budget = 1000

        # Phase 3: Walk backward, filling tail budget
        tail_start = len(pruned_messages)
        accumulated = 0
        for i in range(len(pruned_messages) - 1, rest_start - 1, -1):
            msg_tokens = _estimate_msg_tokens(pruned_messages[i])
            if accumulated + msg_tokens > tail_budget:
                # Check if we've hit the minimum message floor
                tail_count = len(pruned_messages) - i - 1
                if tail_count >= self.protect_last_n:
                    break
                # Under the floor — keep going even over budget
            accumulated += msg_tokens
            tail_start = i

        # Phase 4: Align tail_start to avoid splitting tool call groups.
        # A tool result must stay with its preceding assistant tool_call.
        tail_start = self._align_tool_groups(pruned_messages, tail_start, rest_start)

        # Count what we're dropping
        dropped = tail_start - rest_start
        if dropped <= 0:
            # Nothing to drop
            return messages

        # Phase 5: Assemble result
        result = list(system_msgs)

        # Insert truncation marker
        # Pick a role that doesn't clash with the first tail message
        first_tail_role = pruned_messages[tail_start].get("role", "user") if tail_start < len(pruned_messages) else "user"
        marker_role = "assistant" if first_tail_role != "assistant" else "user"
        result.append({"role": marker_role, "content": TRUNCATION_MARKER})

        # Add tail messages (exact, unmodified)
        for i in range(tail_start, len(pruned_messages)):
            result.append(pruned_messages[i].copy())

        # Sanitize orphaned tool pairs
        result = self._sanitize_tool_pairs(result)

        self.compression_count += 1

        if not self.quiet_mode:
            new_estimate = estimate_messages_tokens_rough(result)
            saved = display_tokens - new_estimate
            tail_count = len(pruned_messages) - tail_start
            logger.info(
                "Truncation #%d: dropped %d old messages, kept %d tail "
                "messages (~%d tokens saved, ~%d remaining)",
                self.compression_count, dropped, tail_count,
                saved, new_estimate,
            )

        return result

    # -- Helpers -----------------------------------------------------------

    def _prune_tool_results(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Replace old tool result bodies with placeholders.

        Protects the last protect_last_n messages from pruning.
        """
        result = [m.copy() for m in messages]
        prune_boundary = max(0, len(result) - self.protect_last_n)

        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            if len(content) > 200:
                result[i] = {**msg, "content": _PRUNED_TOOL_PLACEHOLDER}

        return result

    @staticmethod
    def _align_tool_groups(
        messages: List[Dict[str, Any]],
        cut_idx: int,
        min_idx: int,
    ) -> int:
        """Move cut_idx forward if it would split a tool_call/result group.

        A tool result message (role=tool) must stay with the assistant
        message that issued the tool_call.  If cut_idx lands on a tool
        result, advance past the entire group.
        """
        n = len(messages)
        while cut_idx < n and messages[cut_idx].get("role") == "tool":
            cut_idx += 1
        # Also skip forward if we'd start on an assistant with tool_calls
        # but miss its tool results
        if (
            cut_idx < n
            and messages[cut_idx].get("role") == "assistant"
            and messages[cut_idx].get("tool_calls")
        ):
            # Check if any tool results follow
            j = cut_idx + 1
            while j < n and messages[j].get("role") == "tool":
                j += 1
            # If tool results exist after this assistant, include them
            # by NOT cutting here — move to after the tool results
            if j > cut_idx + 1:
                cut_idx = j
        return max(cut_idx, min_idx)

    @staticmethod
    def _sanitize_tool_pairs(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Remove orphaned tool_call or tool_result messages.

        After truncation, some tool_calls may have lost their results
        or vice versa.  The API rejects mismatched IDs, so clean them up.
        """
        # Collect all tool_call IDs and tool result IDs
        call_ids = set()
        result_ids = set()
        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    call_ids.add(tc.get("id", ""))
            if msg.get("role") == "tool":
                result_ids.add(msg.get("tool_call_id", ""))

        orphan_call_ids = call_ids - result_ids
        orphan_result_ids = result_ids - call_ids

        if not orphan_call_ids and not orphan_result_ids:
            return messages

        cleaned = []
        for msg in messages:
            # Remove orphaned tool results
            if msg.get("role") == "tool" and msg.get("tool_call_id") in orphan_result_ids:
                continue
            # Strip orphaned tool_calls from assistant messages
            if msg.get("tool_calls"):
                original_calls = msg["tool_calls"]
                filtered = [
                    tc for tc in original_calls
                    if not (isinstance(tc, dict) and tc.get("id") in orphan_call_ids)
                ]
                if filtered != original_calls:
                    msg = msg.copy()
                    msg["tool_calls"] = filtered if filtered else None
                    if not filtered:
                        # Remove tool_calls key entirely if empty
                        msg.pop("tool_calls", None)
            cleaned.append(msg)

        return cleaned
