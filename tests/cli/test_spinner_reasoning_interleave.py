"""Regression tests for spinner/reasoning interleave bug.

When reasoning tokens stream in, each _cprint() call triggers prompt_toolkit
to redraw the full TUI layout, including the spinner widget.  In PTY mode
(web-chat's xterm.js), this interleaves the thinking face emoji with every
reasoning line.

The fix: suppress _spinner_text while the reasoning box is open, and block
_on_thinking() updates during that window.

See: fix(cli): suppress spinner widget during reasoning box display
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from unittest.mock import patch, MagicMock
import shutil


def _make_cli_stub():
    """Create a minimal HermesCLI-like object with spinner and reasoning state."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)

    # Spinner state
    cli._spinner_text = ""
    cli._tool_start_time = 0.0

    # Reasoning state
    cli.show_reasoning = True  # Enable reasoning display
    cli.streaming_enabled = True
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._reasoning_shown_this_turn = False
    cli._stream_box_opened = False
    cli._deferred_content = ""

    # Stream state (needed by _flush_reasoning_preview)
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_prefilt = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False

    # Status bar state (needed by _suppress/_restore_status_bar)
    cli._status_bar_visible = True
    cli._status_bar_user_pref = True
    cli._status_bar_suppress_depth = 0

    # Capture _cprint output instead of printing to terminal
    cli._cprinted = []

    # Mock _invalidate (no-op, we're not in a real TUI)
    cli._invalidate = lambda: None

    # Mock _flush_reasoning_preview (called by _on_thinking(""))
    cli._flush_reasoning_preview = lambda force=False: None

    return cli


class TestSpinnerSuppressedDuringReasoning:
    """_spinner_text must be cleared when reasoning box opens."""

    def test_spinner_cleared_on_reasoning_box_open(self):
        """When the first reasoning token arrives, _spinner_text must be ''."""
        cli = _make_cli_stub()

        # Simulate: thinking callback fires before API returns tokens
        cli._spinner_text = "(´･_･`) brainstorming..."

        # Now reasoning tokens start arriving
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._stream_reasoning_delta("I need to think about this")

        assert cli._reasoning_box_opened is True
        assert cli._spinner_text == "", (
            "Spinner text must be cleared when reasoning box opens; "
            "otherwise it interleaves with reasoning lines in PTY output"
        )

    def test_spinner_stays_clear_during_subsequent_reasoning_tokens(self):
        """_spinner_text remains '' throughout reasoning streaming."""
        cli = _make_cli_stub()
        cli._spinner_text = "(´･_･`) brainstorming..."

        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._stream_reasoning_delta("First reasoning line\n")
                assert cli._spinner_text == ""

                cli._stream_reasoning_delta("Second reasoning line\n")
                assert cli._spinner_text == ""

                cli._stream_reasoning_delta("Third line with conclusion\n")
                assert cli._spinner_text == ""


class TestOnThinkingSuppressedDuringReasoning:
    """_on_thinking() must not re-set _spinner_text while reasoning box is open."""

    def test_on_thinking_noop_when_reasoning_box_open(self):
        """_on_thinking('face verb...') should be a no-op during reasoning."""
        cli = _make_cli_stub()
        cli._reasoning_box_opened = True
        cli._spinner_text = ""

        # This simulates the agent calling thinking_callback while
        # reasoning tokens are still streaming
        cli._on_thinking("(´･_･`) brainstorming...")

        assert cli._spinner_text == "", (
            "_on_thinking must not re-set spinner text while reasoning box is open"
        )

    def test_on_thinking_allowed_after_reasoning_box_closes(self):
        """After reasoning box closes, _on_thinking should work normally."""
        cli = _make_cli_stub()
        cli._reasoning_box_opened = False
        cli._spinner_text = ""

        cli._on_thinking("(´･_･`) brainstorming...")
        assert cli._spinner_text == "(´･_･`) brainstorming..."

    def test_on_thinking_empty_still_clears_when_box_open(self):
        """_on_thinking('') (thinking stops) must still work during reasoning."""
        cli = _make_cli_stub()
        cli._reasoning_box_opened = True
        cli._spinner_text = "leftover"

        # Empty text = thinking stopped, should always clear
        cli._on_thinking("")
        assert cli._spinner_text == ""

    def test_on_thinking_before_reasoning_sets_spinner(self):
        """Before any reasoning, _on_thinking sets spinner normally."""
        cli = _make_cli_stub()
        cli._reasoning_box_opened = False

        cli._on_thinking("(´-_-`) contemplating...")
        assert cli._spinner_text == "(´-_-`) contemplating..."


class TestFullSequence:
    """End-to-end sequence: thinking -> reasoning -> close -> normal."""

    def test_full_thinking_reasoning_lifecycle(self):
        """Simulate the complete lifecycle and verify spinner state at each step."""
        cli = _make_cli_stub()

        # Step 1: Thinking starts (before API call)
        cli._on_thinking("(´･_･`) brainstorming...")
        assert cli._spinner_text == "(´･_･`) brainstorming..."

        # Step 2: First reasoning token arrives, opens box
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._stream_reasoning_delta("Analyzing the problem\n")

        assert cli._reasoning_box_opened is True
        assert cli._spinner_text == "", "Spinner must be clear after reasoning box opens"

        # Step 3: More thinking callbacks fire (should be suppressed)
        cli._on_thinking("(´･_･`) pondering...")
        assert cli._spinner_text == "", "Spinner must stay clear during reasoning"

        # Step 4: More reasoning tokens
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._stream_reasoning_delta("Considering edge cases\n")
        assert cli._spinner_text == ""

        # Step 5: Reasoning box closes
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._close_reasoning_box()
        assert cli._reasoning_box_opened is False

        # Step 6: After close, thinking callback works again
        cli._on_thinking("(´-_-`) writing response...")
        assert cli._spinner_text == "(´-_-`) writing response..."

    def test_no_reasoning_spinner_works_normally(self):
        """When there's no reasoning at all, spinner is unaffected."""
        cli = _make_cli_stub()

        cli._on_thinking("(´･_･`) brainstorming...")
        assert cli._spinner_text == "(´･_･`) brainstorming..."

        cli._on_thinking("(´-_-`) pondering...")
        assert cli._spinner_text == "(´-_-`) pondering..."

        cli._on_thinking("")
        assert cli._spinner_text == ""
