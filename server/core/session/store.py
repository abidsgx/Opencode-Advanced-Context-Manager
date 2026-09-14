import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# --- 1. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
SESSIONS_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'sessions'


# --- 2. Session store ---

class SessionStore:
    def __init__(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def record_edit(self, session_id: str, file_path: str, function_name: str,
                    reason: str, goal_contribution: str, tradeoffs: str = "") -> dict:
        """Record an edit with metadata."""
        session = self._load_or_create(session_id)

        edit = {
            'file': file_path,
            'function': function_name,
            'timestamp': time.time(),
            'reason': reason,
            'impact': goal_contribution,
            'tradeoffs': tradeoffs,
            'tested': False,
        }

        if 'editImpact' not in session:
            session['editImpact'] = []
        session['editImpact'].append(edit)
        session['updatedAt'] = time.time()

        self._save(session_id, session)
        return {'status': 'ok', 'edit': edit}

    def record_importance(self, session_id: str, scores: dict):
        """Record importance scores for a session."""
        session = self._load_or_create(session_id)
        session['importanceScores'] = scores
        session['updatedAt'] = time.time()
        self._save(session_id, session)

    def record_interconnectedness(self, session_id: str, matrix: dict):
        """Record interconnectedness matrix for a session."""
        session = self._load_or_create(session_id)
        session['interconnectedness'] = matrix
        session['updatedAt'] = time.time()
        self._save(session_id, session)

    def get_summary(self, session_id: str) -> dict:
        """Get a comprehensive session summary."""
        session = self._load(session_id)
        if not session:
            return {'error': 'Session not found'}

        turns = session.get('turns', [])
        edit_impact = session.get('editImpact', [])
        importance = session.get('importanceScores', {})
        interconnectedness = session.get('interconnectedness', {})

        files_edited = set()
        functions_edited = set()
        for edit in edit_impact:
            files_edited.add(edit.get('file', ''))
            func = edit.get('function', '')
            if func:
                functions_edited.add(func)

        return {
            'session_id': session_id,
            'title': session.get('title', ''),
            'startedAt': session.get('startedAt', ''),
            'turn_count': len(turns),
            'edit_count': len(edit_impact),
            'files_edited': list(files_edited),
            'functions_edited': list(functions_edited),
            'importance_scores': importance,
            'interconnectedness_files': list((interconnectedness or {}).get('files', {}).keys()),
        }

    def list_sessions(self) -> list:
        """List all sessions."""
        sessions = []
        for path in SESSIONS_DIR.glob('*.json'):
            if path.name == 'index.json':
                continue
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                sessions.append({
                    'id': data.get('_id', path.stem),
                    'title': data.get('title', ''),
                    'startedAt': data.get('startedAt', ''),
                })
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s.get('startedAt', ''), reverse=True)

    def _load_or_create(self, session_id: str) -> dict:
        """Load existing session or create a new one."""
        session = self._load(session_id)
        if session:
            return session

        return {
            '_id': session_id,
            'title': '',
            'startedAt': time.time(),
            'updatedAt': time.time(),
            'turns': [],
            'editImpact': [],
            'importanceScores': None,
            'interconnectedness': None,
        }

    def _load(self, session_id: str) -> Optional[dict]:
        """Load a session from disk."""
        path = SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _save(self, session_id: str, session: dict):
        """Save a session to disk."""
        session['updatedAt'] = time.time()
        path = SESSIONS_DIR / f"{session_id}.json"
        with open(path, 'w') as f:
            json.dump(session, f, indent=2)


# --- 3. Test it ---

if __name__ == '__main__':
    store = SessionStore()

    # Record an edit
    result = store.record_edit(
        'session_1', 'src/auth/login.py', 'authenticate',
        'Fix timeout bug', 'Correctly calculates TTL', 'Slightly more tokens'
    )
    print(f"Recorded edit: {result}")

    # Get summary
    summary = store.get_summary('session_1')
    print("\nSession summary:")
    print(f"  Files: {summary['files_edited']}")
    print(f"  Functions: {summary['functions_edited']}")
    print(f"  Edits: {summary['edit_count']}")

    # List sessions
    sessions = store.list_sessions()
    print(f"\nSessions: {len(sessions)}")

    print("\nDone.")
