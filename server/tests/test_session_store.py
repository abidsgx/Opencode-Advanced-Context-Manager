"""Tests for SessionStore."""

import json
import time
import pytest

from core.session.store import SessionStore


class TestSessionStore:
    """Tests for SessionStore."""

    def test_record_edit(self, tmp_sessions_dir, monkeypatch):
        """Edit stored with metadata."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        result = store.record_edit(
            "session_1",
            "src/auth/login.py",
            "authenticate",
            "Fix timeout bug",
            "Correctly calculates TTL",
            "Slightly more tokens",
        )

        assert result["status"] == "ok"
        assert result["edit"]["file"] == "src/auth/login.py"
        assert result["edit"]["function"] == "authenticate"
        assert result["edit"]["reason"] == "Fix timeout bug"

    def test_record_importance(self, tmp_sessions_dir, monkeypatch):
        """Scores stored."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        scores = {"files": {"src/auth/login.py": 9.5}, "functions": {}}
        store.record_importance("session_1", scores)

        session = store._load("session_1")
        assert session["importanceScores"] == scores

    def test_record_interconnectedness(self, tmp_sessions_dir, monkeypatch):
        """Matrix stored."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        matrix = {"files": {"src/a.py": {"src/b.py": {"weight": 0.8}}}, "functions": {}}
        store.record_interconnectedness("session_1", matrix)

        session = store._load("session_1")
        assert session["interconnectedness"] == matrix

    def test_get_summary(self, tmp_sessions_dir, monkeypatch):
        """Summary has all fields."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        store.record_edit("session_1", "src/auth/login.py", "authenticate", "Fix", "TTL")

        summary = store.get_summary("session_1")

        assert summary["session_id"] == "session_1"
        assert "turn_count" in summary
        assert "edit_count" in summary
        assert "files_edited" in summary
        assert "functions_edited" in summary

    def test_get_summary_not_found(self, tmp_sessions_dir, monkeypatch):
        """Unknown -> error."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        summary = store.get_summary("nonexistent")

        assert "error" in summary

    def test_list_sessions(self, tmp_sessions_dir, monkeypatch):
        """Sessions listed."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        store.record_edit("session_1", "src/a.py", "func", "Fix", "Impact")
        store.record_edit("session_2", "src/b.py", "func", "Fix", "Impact")

        sessions = store.list_sessions()

        assert len(sessions) == 2

    def test_load_or_create(self, tmp_sessions_dir, monkeypatch):
        """Creates if missing."""
        monkeypatch.setattr("core.session.store.SESSIONS_DIR", tmp_sessions_dir)

        store = SessionStore()
        session = store._load_or_create("new_session")

        assert session["_id"] == "new_session"
        assert session["turns"] == []
        assert session["editImpact"] == []
