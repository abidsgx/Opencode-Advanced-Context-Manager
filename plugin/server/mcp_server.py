"""MCP Server for Advanced Context Manager.

Exposes tools for importance scoring, interconnectedness analysis,
and active learning via the Model Context Protocol.
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("ctx-manager")


@mcp.tool()
def score_importance(session_id: str, goal: str = "") -> dict:
    """Score importance of files and functions for a given session/goal.

    Returns a dict with 'files' and 'functions' mapping paths to 0-10 scores.
    """
    from core.importance.scorer import ImportanceScorer

    scorer = ImportanceScorer()
    return scorer.score_session(session_id, goal)


@mcp.tool()
def get_interconnectedness(session_id: str, goal: str = "") -> dict:
    """Get the interconnectedness matrix for files and functions.

    Returns hierarchical file->function relationship data.
    """
    from core.graph.interconnectedness import InterconnectednessManager

    manager = InterconnectednessManager()
    return manager.get_matrix(session_id, goal)


@mcp.tool()
def record_edit(
    session_id: str,
    file_path: str,
    function_name: str,
    reason: str,
    goal_contribution: str,
    tradeoffs: str = "",
) -> dict:
    """Record an edit with metadata for importance tracking.

    Stores the edit reason, goal contribution, and tradeoffs.
    """
    from core.session.store import SessionStore

    store = SessionStore()
    return store.record_edit(
        session_id, file_path, function_name, reason, goal_contribution, tradeoffs
    )


@mcp.tool()
def get_pending_feedback() -> dict:
    """Get pending active learning samples that need user labeling.

    Returns a list of uncertain predictions for the user to review.
    """
    from core.importance.active_learning import ActiveLearner

    learner = ActiveLearner()
    return learner.get_pending_samples()


@mcp.tool()
def submit_labels(labels: list[dict]) -> dict:
    """Submit user labels for active learning samples.

    Each label dict should have: sample_id (str), score (float 0-10).
    """
    from core.importance.active_learning import ActiveLearner

    learner = ActiveLearner()
    return learner.submit_labels(labels)


@mcp.tool()
def get_session_summary(session_id: str) -> dict:
    """Get a comprehensive summary of a session's data.

    Returns turns, edit impacts, importance scores, and interconnectedness.
    """
    from core.session.store import SessionStore

    store = SessionStore()
    return store.get_summary(session_id)


@mcp.tool()
def update_config(section: str, key: str, value: str) -> dict:
    """Update a configuration value in data/config.yaml.

    Args:
        section: Top-level config section (e.g., 'importance', 'pruning')
        key: Config key to update
        value: New value (will be parsed as YAML)
    """
    from config.settings import update_setting

    return update_setting(section, key, value)


@mcp.tool()
def install_grammars(languages: list[str] = None) -> dict:
    """Install tree-sitter grammars for specified languages.

    If no languages specified, installs grammars for all enabled languages
    in the config.

    Args:
        languages: List of language names (e.g., ['python', 'typescript'])
    """
    from core.graph.tree_sitter import get_enabled_languages, install_all_enabled, install_grammar

    if languages:
        installed = []
        failed = []
        for lang in languages:
            success = install_grammar(lang)
            if success:
                installed.append(lang)
            else:
                failed.append(lang)
        return {'installed': installed, 'failed': failed}
    else:
        return install_all_enabled()


@mcp.tool()
def analyze_file_tree(directory: str) -> dict:
    """Analyze all source files in a directory using tree-sitter.

    Returns per-file analysis: language, line count, imports, functions, call depth.
    Only analyzes files for enabled languages in config.
    """
    from core.graph.tree_sitter import analyze_directory

    results = analyze_directory(directory)
    return {
        'file_count': len(results),
        'files': results,
    }


@mcp.tool()
def get_languages() -> dict:
    """Get configured languages and their status.

    Returns which languages are enabled and their file extensions.
    """
    from core.graph.tree_sitter import get_enabled_languages, get_extension_map

    enabled = get_enabled_languages()
    ext_map = get_extension_map()

    return {
        'enabled': {name: info for name, info in enabled.items()},
        'extension_map': ext_map,
    }


@mcp.tool()
def set_language(language: str, enabled: bool, extensions: list[str] = None) -> dict:
    """Enable or disable a language for tree-sitter analysis.

    Args:
        language: Language name (e.g., 'python', 'typescript')
        enabled: Whether to enable this language
        extensions: File extensions (e.g., ['.py', '.pyw']). If None, uses defaults.
    """
    from config.settings import load_config, save_config

    config = load_config()
    if 'languages' not in config:
        config['languages'] = {}

    if language not in config['languages']:
        config['languages'][language] = {}

    config['languages'][language]['enabled'] = enabled
    if extensions:
        config['languages'][language]['extensions'] = extensions

    save_config(config)
    return {'status': 'ok', 'language': language, 'enabled': enabled}


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
