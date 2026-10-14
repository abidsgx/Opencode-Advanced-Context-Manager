import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")


# --- 1. Config loading ---

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_enabled_languages() -> dict:
    """Get languages that are enabled in config."""
    config = load_config()
    langs = config.get('languages', {})
    return {name: info for name, info in langs.items() if info.get('enabled', False)}


def get_extension_map() -> dict:
    """Build extension -> language name map from config."""
    config = load_config()
    langs = config.get('languages', {})
    ext_map = {}
    for lang_name, info in langs.items():
        if not info.get('enabled', False):
            continue
        for ext in info.get('extensions', []):
            ext_map[ext] = lang_name
    return ext_map


# --- 2. Grammar installer ---

GRAMMAR_PACKAGES = {
    'python': 'tree-sitter-python',
    'javascript': 'tree-sitter-javascript',
    'typescript': 'tree-sitter-typescript',
    'rust': 'tree-sitter-rust',
    'go': 'tree-sitter-go',
    'java': 'tree-sitter-java',
    'ruby': 'tree-sitter-ruby',
    'c': 'tree-sitter-c',
    'cpp': 'tree-sitter-cpp',
    'csharp': 'tree-sitter-c-sharp',
    'php': 'tree-sitter-php',
    'elixir': 'tree-sitter-elixir',
}


def install_grammar(language: str) -> bool:
    """Install a tree-sitter grammar package for a language."""
    package = GRAMMAR_PACKAGES.get(language)
    if not package:
        print(f"Unknown language: {language}")
        return False

    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False


def install_all_enabled():
    """Install grammars for all enabled languages."""
    enabled = get_enabled_languages()
    installed = []
    failed = []

    for lang_name in enabled:
        success = install_grammar(lang_name)
        if success:
            installed.append(lang_name)
        else:
            failed.append(lang_name)

    return {'installed': installed, 'failed': failed}


# --- 3. Parser cache ---

_parser_cache: Dict[str, object] = {}
_language_cache: Dict[str, object] = {}


def get_parser(language: str):
    """Get or create a tree-sitter parser for a language."""
    if language in _parser_cache:
        return _parser_cache[language]

    try:
        import tree_sitter
        parser_module = _import_language_module(language)
        if parser_module is None:
            return None

        lang_fn = getattr(parser_module, 'language', None)
        if lang_fn is None:
            return None

        lang = lang_fn() if callable(lang_fn) else lang_fn
        parser = tree_sitter.Parser()
        parser.language = lang

        _parser_cache[language] = parser
        return parser
    except ImportError:
        return None


def _import_language_module(language: str):
    """Import the tree-sitter language module for a given language."""
    module_names = {
        'python': 'tree_sitter_python',
        'javascript': 'tree_sitter_javascript',
        'typescript': 'tree_sitter_typescript',
        'rust': 'tree_sitter_rust',
        'go': 'tree_sitter_go',
        'java': 'tree_sitter_java',
        'ruby': 'tree_sitter_ruby',
        'c': 'tree_sitter_c',
        'cpp': 'tree_sitter_cpp',
        'csharp': 'tree_sitter_c_sharp',
        'php': 'tree_sitter_php',
        'elixir': 'tree_sitter_elixir',
    }

    module_name = module_names.get(language)
    if not module_name:
        return None

    try:
        return __import__(module_name)
    except ImportError:
        return None


# --- 4. AST analysis functions ---

def parse_file(file_path: str, content: str = None) -> Optional[object]:
    """Parse a file and return the tree-sitter tree."""
    ext = Path(file_path).suffix
    ext_map = get_extension_map()
    language = ext_map.get(ext)

    if not language:
        return None

    parser = get_parser(language)
    if parser is None:
        return None

    if content is None:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            return None

    return parser.parse(bytes(content, 'utf-8'))


def get_imports(tree, file_path: str) -> List[str]:
    """Extract import statements from a parsed tree.

    Returns list of imported module/file paths.
    """
    ext = Path(file_path).suffix
    ext_map = get_extension_map()
    language = ext_map.get(ext)

    if not language or tree is None:
        return []

    imports = []
    root = tree.root_node

    # Python imports
    if language == 'python':
        for node in _walk_tree(root):
            if node.type == 'import_statement':
                text = node.text.decode('utf-8') if node.text else ''
                # Parse "import foo" or "from foo import bar"
                parts = text.replace('import ', '').strip().split('.')
                if parts:
                    imports.append(parts[0])
            elif node.type == 'import_from_statement':
                text = node.text.decode('utf-8') if node.text else ''
                # "from foo.bar import baz" -> "foo"
                parts = text.split('import')[0].replace('from ', '').strip().split('.')
                if parts:
                    imports.append(parts[0])

    # JavaScript/TypeScript imports
    elif language in ('javascript', 'typescript'):
        for node in _walk_tree(root):
            if node.type in ('import_statement', 'import_clause'):
                text = node.text.decode('utf-8') if node.text else ''
                # Extract from string literal
                if "'" in text:
                    module = text.split("'")[1]
                    imports.append(module)
                elif '"' in text:
                    module = text.split('"')[1]
                    imports.append(module)

    # Rust imports
    elif language == 'rust':
        for node in _walk_tree(root):
            if node.type == 'use_declaration':
                text = node.text.decode('utf-8') if node.text else ''
                # "use crate::module::func" -> "crate"
                parts = text.replace('use ', '').strip().split('::')
                if parts:
                    imports.append(parts[0])

    # Go imports
    elif language == 'go':
        for node in _walk_tree(root):
            if node.type == 'import_declaration':
                text = node.text.decode('utf-8') if node.text else ''
                # Extract from string literals in import block
                for line in text.split('\n'):
                    line = line.strip().strip('"')
                    if line and not line.startswith('import'):
                        imports.append(line)

    # Generic fallback: look for string literals after import/using/from keywords
    else:
        for node in _walk_tree(root):
            if node.type in ('import_statement', 'using_statement'):
                text = node.text.decode('utf-8') if node.text else ''
                # Try to extract quoted string
                for quote in ('"', "'"):
                    if quote in text:
                        module = text.split(quote)[1]
                        imports.append(module)
                        break

    return imports


def get_function_definitions(tree, file_path: str) -> List[Dict]:
    """Extract function/method definitions from a parsed tree.

    Returns list of dicts with 'name', 'line', 'depth'.
    """
    ext = Path(file_path).suffix
    ext_map = get_extension_map()
    language = ext_map.get(ext)

    if not language or tree is None:
        return []

    definitions = []
    root = tree.root_node

    # Language-specific definition node types
    def_node_types = {
        'python': ['function_definition', 'class_definition'],
        'javascript': ['function_declaration', 'arrow_function', 'class_declaration'],
        'typescript': ['function_declaration', 'arrow_function', 'class_declaration', 'interface_declaration'],
        'rust': ['function_item', 'impl_item', 'struct_item'],
        'go': ['function_declaration', 'method_declaration'],
        'java': ['method_declaration', 'class_declaration'],
        'ruby': ['method', 'class'],
        'c': ['function_definition'],
        'cpp': ['function_definition', 'class_specifier'],
        'csharp': ['method_declaration', 'class_declaration'],
        'php': ['function_declaration', 'class_declaration'],
        'elixir': ['function_definition', 'call'],
    }

    target_types = def_node_types.get(language, ['function_definition', 'class_definition'])

    for node in _walk_tree(root):
        if node.type in target_types:
            name = _get_definition_name(node, language)
            if name:
                definitions.append({
                    'name': name,
                    'line': node.start_point[0],
                    'type': node.type,
                })

    return definitions


def get_call_depth(tree, file_path: str, target_function: str = None) -> int:
    """Estimate call depth from the AST.

    Counts nesting depth of function calls within the target function.
    If no target_function specified, returns max depth in file.
    """
    ext = Path(file_path).suffix
    ext_map = get_extension_map()
    language = ext_map.get(ext)

    if not language or tree is None:
        return 0

    root = tree.root_node
    max_depth = 0

    def _count_depth(node, current_depth=0):
        nonlocal max_depth
        if node.type in ('call_expression', 'call'):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        for child in node.children:
            _count_depth(child, current_depth)
        if node.type in ('function_definition', 'function_declaration', 'function_item'):
            # Reset depth at function boundary
            for child in node.children:
                if child.type != node.type:
                    _count_depth(child, 0)

    _count_depth(root)
    return max_depth


def count_lines(content: str) -> int:
    """Count lines in file content."""
    return content.count('\n') + 1 if content else 0


# --- 5. Helper functions ---

def _walk_tree(node):
    """Recursively walk all nodes in the tree."""
    yield node
    for child in node.children:
        yield from _walk_tree(child)


def _get_definition_name(node, language: str) -> Optional[str]:
    """Extract the name from a definition node."""
    # Try to find a 'name' child node
    for child in node.children:
        if child.type == 'name' or child.type == 'identifier':
            return child.text.decode('utf-8') if child.text else None
        if child.type == 'attribute' and child.type == 'name':
            return child.text.decode('utf-8') if child.text else None

    # Fallback: parse text
    text = node.text.decode('utf-8') if node.text else ''

    if language == 'python':
        # "def func_name(" or "class ClassName("
        for keyword in ('def ', 'class '):
            if keyword in text:
                rest = text.split(keyword)[1]
                name = rest.split('(')[0].split(':')[0].strip()
                return name

    elif language in ('javascript', 'typescript'):
        # "function funcName(" or "const funcName ="
        if 'function ' in text:
            rest = text.split('function ')[1]
            return rest.split('(')[0].strip()
        if '=>' in text:
            # Arrow function assigned to variable
            parts = text.split('=')
            if len(parts) > 1:
                return parts[0].strip().replace('const ', '').replace('let ', '').replace('var ', '')

    elif language == 'rust':
        # "fn func_name("
        if 'fn ' in text:
            rest = text.split('fn ')[1]
            return rest.split('(')[0].strip()

    elif language == 'go':
        # "func func_name(" or "func (recv) func_name("
        if 'func ' in text:
            rest = text.split('func ')[1]
            if ')' in rest:
                rest = rest.split(')')[1]
            return rest.split('(')[0].strip()

    elif language == 'java':
        # "public void methodName("
        parts = text.split('(')
        if len(parts) > 1:
            tokens = parts[0].split()
            return tokens[-1] if tokens else None

    return None


# --- 6. High-level interface ---

def analyze_file(file_path: str) -> dict:
    """Analyze a single file: parse, extract imports, functions, call depth."""
    ext = Path(file_path).suffix
    ext_map = get_extension_map()
    language = ext_map.get(ext)

    if not language:
        return {
            'language': None,
            'supported': False,
            'lines': 0,
            'imports': [],
            'functions': [],
            'call_depth': 0,
        }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError):
        return {
            'language': language,
            'supported': True,
            'lines': 0,
            'imports': [],
            'functions': [],
            'call_depth': 0,
        }

    tree = parse_file(file_path, content)
    if tree is None:
        return {
            'language': language,
            'supported': True,
            'lines': count_lines(content),
            'imports': [],
            'functions': [],
            'call_depth': 0,
        }

    return {
        'language': language,
        'supported': True,
        'lines': count_lines(content),
        'imports': get_imports(tree, file_path),
        'functions': get_function_definitions(tree, file_path),
        'call_depth': get_call_depth(tree, file_path),
    }


def analyze_directory(directory: str, extensions: Set[str] = None) -> dict:
    """Analyze all files in a directory tree.

    Returns dict mapping file_path -> analysis result.
    """
    ext_map = get_extension_map()
    if extensions is None:
        extensions = set(ext_map.keys())

    results = {}
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', 'venv', '.git')]

        for fname in files:
            ext = Path(fname).suffix
            if ext not in extensions:
                continue

            file_path = os.path.join(root, fname)
            rel_path = os.path.relpath(file_path, directory)
            results[rel_path] = analyze_file(file_path)

    return results


# --- 7. Test it ---

if __name__ == '__main__':
    print("=== Enabled Languages ===")
    enabled = get_enabled_languages()
    for name, info in enabled.items():
        exts = info.get('extensions', [])
        print(f"  {name}: {exts}")

    print("\n=== Extension Map ===")
    ext_map = get_extension_map()
    for ext, lang in ext_map.items():
        print(f"  {ext} -> {lang}")

    print("\n=== Install Grammars ===")
    result = install_all_enabled()
    print(f"  Installed: {result['installed']}")
    print(f"  Failed: {result['failed']}")

    # Test parsing a Python file if available
    test_file = None
    for f in Path('.').rglob('*.py'):
        if 'tree_sitter' not in str(f) and '__pycache__' not in str(f):
            test_file = str(f)
            break

    if test_file:
        print(f"\n=== Analyzing {test_file} ===")
        analysis = analyze_file(test_file)
        print(f"  Language: {analysis['language']}")
        print(f"  Lines: {analysis['lines']}")
        print(f"  Imports: {analysis['imports']}")
        print(f"  Functions: {[f['name'] for f in analysis['functions']]}")
        print(f"  Call depth: {analysis['call_depth']}")

    print("\nDone.")
