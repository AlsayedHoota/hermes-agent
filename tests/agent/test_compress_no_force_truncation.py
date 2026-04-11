"""Regression test: compress() must NOT accept force_truncation.

This test was created after a half-implemented feature caused crashes:
  ContextCompressor.compress() got an unexpected keyword argument 'force_truncation'

The fix was to remove force_truncation entirely and match upstream behavior:
- On 413/context overflow: compress normally, retry, exit gracefully if still too big
- No special "forced truncation" mode that drops context without summary

See: https://github.com/NousResearch/hermes-agent (upstream behavior)
"""

import inspect
import pytest
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


class TestCompressSignatureRegression:
    """Ensure compress() signature stays clean and matches upstream."""

    def test_compress_does_not_accept_force_truncation(self):
        """compress() must reject force_truncation kwarg (regression for crash bug)."""
        compressor = ContextCompressor(model="test", quiet_mode=True)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Welcome"},
            {"role": "user", "content": "Bye"},
        ]
        with pytest.raises(TypeError, match="force_truncation"):
            compressor.compress(messages, current_tokens=50000, force_truncation=True)

    def test_compress_signature_has_no_force_truncation_param(self):
        """The compress() method signature must not include force_truncation."""
        sig = inspect.signature(ContextCompressor.compress)
        assert "force_truncation" not in sig.parameters, (
            "compress() should NOT have a force_truncation parameter. "
            "This was removed to match upstream. If you need forced truncation, "
            "handle it in _compress_context or the caller, not in compress()."
        )

    def test_compress_accepts_only_messages_and_current_tokens(self):
        """compress() should accept exactly: self, messages, current_tokens."""
        sig = inspect.signature(ContextCompressor.compress)
        param_names = list(sig.parameters.keys())
        assert param_names == ["self", "messages", "current_tokens"], (
            f"compress() signature changed unexpectedly: {param_names}. "
            f"Expected ['self', 'messages', 'current_tokens']."
        )


class TestCompressNormalBehavior:
    """Verify compress() works correctly in normal and fallback scenarios."""

    def test_compress_returns_messages_when_too_few(self):
        """compress() should return messages unchanged if there aren't enough to compress."""
        compressor = ContextCompressor(model="test", quiet_mode=True)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = compressor.compress(messages)
        assert result == messages

    def test_compress_with_current_tokens(self):
        """compress() should accept current_tokens without error."""
        compressor = ContextCompressor(model="test", quiet_mode=True)
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        # Should not raise
        result = compressor.compress(messages, current_tokens=1000)
        assert isinstance(result, list)

    def test_static_fallback_when_summary_fails(self):
        """When LLM summary fails, compress() inserts a static fallback marker."""
        compressor = ContextCompressor(model="test", quiet_mode=True)
        compressor.context_length = 200_000
        compressor.threshold_tokens = 1000
        compressor.protect_first_n = 2
        compressor.tail_token_budget = 500

        # Build enough messages to trigger compression
        messages = [{"role": "system", "content": "System prompt"}]
        for i in range(30):
            messages.append({"role": "user", "content": f"Question {i} " + "x" * 100})
            messages.append({"role": "assistant", "content": f"Answer {i} " + "y" * 100})

        # No LLM client = summary will fail = static fallback
        result = compressor.compress(messages, current_tokens=50000)

        # Should still return compressed messages (not crash)
        assert len(result) < len(messages), "Messages should be compressed"
        # The static fallback marker should be present
        all_content = " ".join(m.get("content", "") for m in result if isinstance(m.get("content"), str))
        assert "Summary generation was unavailable" in all_content or "summary" in all_content.lower()
