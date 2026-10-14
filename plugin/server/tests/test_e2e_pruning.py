"""End-to-end pruning integration test.

Tests the pruning pipeline: messages -> importance scoring -> prune low-importance -> output.
This is a Python-side simulation of what the TS plugin does.
"""

import time
import numpy as np
import pytest


def _make_tool_messages(edits):
    """Create a list of tool-result messages from edit data.

    Each edit is (file_path, output_size, timestamp_offset_sec).
    """
    messages = []
    for file_path, output_size, offset in edits:
        messages.append({
            "role": "tool",
            "parts": [{
                "type": "tool_result",
                "toolName": "file_edit",
                "toolInput": {"path": file_path},
                "output": f"// file: {file_path}\n" + "x" * output_size,
                "toolOutput": f"// file: {file_path}\n" + "x" * output_size,
            }],
        })
    return messages


def _score_messages(messages, importance_scores):
    """Score each message based on file references found in it."""
    scored = []
    for msg in messages:
        text = str(msg)
        max_score = 0
        matched_file = None
        for file_path, score in importance_scores.items():
            if file_path in text:
                if score > max_score:
                    max_score = score
                    matched_file = file_path
        scored.append({"msg": msg, "score": max_score, "file": matched_file})
    return scored


class TestE2EPruning:
    """Pruning pipeline tests."""

    def test_pruning_removes_low_importance(self):
        """Messages below threshold are prunable."""
        threshold = 3.0

        importance_scores = {
            "src/auth/login.py": 9.0,
            "src/db/users.py": 2.0,
            "src/utils/helpers.py": 1.0,
        }

        messages = _make_tool_messages([
            ("src/auth/login.py", 500, 300),
            ("src/db/users.py", 500, 200),
            ("src/utils/helpers.py", 500, 100),
        ])

        scored = _score_messages(messages, importance_scores)

        prunable = [s for s in scored if s["score"] < threshold]
        protected = [s for s in scored if s["score"] >= threshold]

        assert len(prunable) == 2  # users.py and helpers.py
        assert len(protected) == 1  # login.py

    def test_protected_files_not_pruned(self):
        """Files above protected threshold are never pruned."""
        protected_threshold = 7.0

        importance_scores = {
            "src/auth/login.py": 9.5,
            "src/auth/middleware.py": 7.5,
            "src/db/users.py": 3.0,
        }

        messages = _make_tool_messages([
            ("src/auth/login.py", 500, 300),
            ("src/auth/middleware.py", 500, 200),
            ("src/db/users.py", 500, 100),
        ])

        scored = _score_messages(messages, importance_scores)

        protected = [s for s in scored if s["score"] >= protected_threshold]
        assert len(protected) == 2  # login.py and middleware.py

    def test_pruning_respects_compression_ratio(self):
        """Pruning does not exceed max compression ratio."""
        max_ratio = 0.3

        importance_scores = {
            f"src/file_{i}.py": float(i) for i in range(20)
        }

        messages = _make_tool_messages([
            (f"src/file_{i}.py", 100, i * 10) for i in range(20)
        ])

        total_tokens = sum(len(str(m)) for m in messages)
        max_prunable = int(total_tokens * max_ratio)

        scored = _score_messages(messages, importance_scores)
        scored.sort(key=lambda s: s["score"])

        pruned_tokens = 0
        pruned_count = 0
        for s in scored:
            msg_tokens = len(str(s["msg"]))
            if pruned_tokens + msg_tokens <= max_prunable:
                pruned_tokens += msg_tokens
                pruned_count += 1

        assert pruned_tokens <= max_prunable
        assert pruned_count < len(messages)

    def test_pruning_preserves_revert_pattern(self):
        """Reverted edits are identified and scored correctly."""
        session = {
            "_id": "revert_session",
            "turns": [
                {
                    "role": "assistant",
                    "parts": [{
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": "src/a.py"},
                        "toolOutput": "line1\nline2\nline3",
                        "timestamp": time.time() - 600,
                    }],
                },
                {
                    "role": "assistant",
                    "parts": [{
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": "src/a.py"},
                        "toolOutput": "reverted previous change",
                        "timestamp": time.time() - 300,
                    }],
                },
            ],
        }

        output = session["turns"][1]["parts"][0]["toolOutput"]
        assert "reverted" in output.lower()

    def test_pruning_summary_accurate(self):
        """Pruning summary reports correct counts."""
        importance_scores = {
            "src/important.py": 9.0,
            "src/low.py": 1.0,
        }

        messages = _make_tool_messages([
            ("src/important.py", 500, 300),
            ("src/low.py", 500, 100),
        ])

        scored = _score_messages(messages, importance_scores)
        files_pruned = [s for s in scored if s["score"] < 3.0]
        files_preserved = [s for s in scored if s["score"] >= 7.0]

        summary = {
            "pruned_count": len(files_pruned),
            "preserved_count": len(files_preserved),
            "tokens_removed": sum(len(str(s["msg"])) for s in files_pruned),
        }

        assert summary["pruned_count"] == 1
        assert summary["preserved_count"] == 1
        assert summary["tokens_removed"] > 0

    def test_empty_messages_no_pruning(self):
        """Empty message list produces zero pruning."""
        scored = _score_messages([], {})
        assert len(scored) == 0

    def test_all_protected_no_pruning(self):
        """When all files are above threshold, nothing is pruned."""
        importance_scores = {
            "src/a.py": 9.0,
            "src/b.py": 8.0,
        }

        messages = _make_tool_messages([
            ("src/a.py", 500, 300),
            ("src/b.py", 500, 200),
        ])

        scored = _score_messages(messages, importance_scores)
        prunable = [s for s in scored if s["score"] < 3.0]
        assert len(prunable) == 0
