from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

_TS_SYMBOL_RE = re.compile(
    r"(?P<kind>export\s+class|class|export\s+function|function)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<tail>\([^)]*\))?",
    re.MULTILINE,
)
_TS_IMPORT_RE = re.compile(r"""from\s+["'](?P<module>\.[^"']+)["']|import\s+["'](?P<side>\.[^"']+)["']""")


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: str
    path: str
    language: str
    symbol_name: str
    symbol_kind: str
    signature: str
    start_line: int
    end_line: int
    content: str
    terms: set[str] = field(default_factory=set)


class CodeIndex:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.chunks: list[IndexedChunk] = []
        self.import_graph: dict[str, set[str]] = {}
        self.failed_files: list[str] = []

    def build(self, workspace: str | Path) -> "CodeIndex":
        self.workspace = Path(workspace)
        self.chunks.clear()
        self.import_graph.clear()
        self.failed_files.clear()
        source_files: list[Path] = []
        for pattern in ("*.py", "*.ts", "*.tsx"):
            source_files.extend(sorted(self.workspace.rglob(pattern)))
        for path in source_files:
            if not path.is_file():
                continue
            try:
                if path.suffix == ".py":
                    self._index_python_file(path)
                else:
                    self._index_ts_file(path)
            except Exception:
                self.failed_files.append(str(path))
        return self

    def _index_python_file(self, path: Path) -> None:
        assert self.workspace is not None
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(self.workspace).as_posix()
        lines = text.splitlines()
        tree = ast.parse(text)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.replace(".", "/") + ".py")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.replace(".", "/") + ".py")
        self.import_graph[rel] = imports
        self.chunks.append(self._build_chunk(rel, "python", path.stem, "file", f"file {rel}", 1, max(1, len(lines)), text))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self.chunks.append(
                    self._build_chunk(
                        rel,
                        "python",
                        node.name,
                        "class",
                        f"class {node.name}",
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        self._slice(lines, node.lineno, getattr(node, "end_lineno", node.lineno)),
                    )
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.chunks.append(
                            self._build_chunk(
                                rel,
                                "python",
                                child.name,
                                "method",
                                f"def {child.name}",
                                child.lineno,
                                getattr(child, "end_lineno", child.lineno),
                                self._slice(lines, child.lineno, getattr(child, "end_lineno", child.lineno)),
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.chunks.append(
                    self._build_chunk(
                        rel,
                        "python",
                        node.name,
                        "function",
                        f"def {node.name}",
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        self._slice(lines, node.lineno, getattr(node, "end_lineno", node.lineno)),
                    )
                )

    def _index_ts_file(self, path: Path) -> None:
        assert self.workspace is not None
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(self.workspace).as_posix()
        lines = text.splitlines()
        imports: set[str] = set()
        for match in _TS_IMPORT_RE.finditer(text):
            value = match.group("module") or match.group("side")
            if value:
                imports.add(self._resolve_ts_import(rel, value))
        self.import_graph[rel] = {item for item in imports if item}
        self.chunks.append(self._build_chunk(rel, "typescript", path.stem, "file", f"file {rel}", 1, max(1, len(lines)), text))
        for match in _TS_SYMBOL_RE.finditer(text):
            kind_text = match.group("kind")
            name = match.group("name")
            kind = "class" if "class" in kind_text else "function"
            start_line = text[: match.start()].count("\n") + 1
            end_line = self._find_block_end(lines, start_line)
            self.chunks.append(
                self._build_chunk(
                    rel,
                    "typescript",
                    name,
                    kind,
                    match.group(0).strip(),
                    start_line,
                    end_line,
                    self._slice(lines, start_line, end_line),
                )
            )

    def _build_chunk(
        self,
        path: str,
        language: str,
        symbol_name: str,
        symbol_kind: str,
        signature: str,
        start_line: int,
        end_line: int,
        content: str,
    ) -> IndexedChunk:
        chunk_id = f"{path}::{symbol_kind}::{symbol_name}::{start_line}"
        terms = set(_tokenize(" ".join([path, symbol_name, symbol_kind, signature, content])))
        return IndexedChunk(
            chunk_id=chunk_id,
            path=path,
            language=language,
            symbol_name=symbol_name,
            symbol_kind=symbol_kind,
            signature=signature,
            start_line=start_line,
            end_line=end_line,
            content=content,
            terms=terms,
        )

    @staticmethod
    def _slice(lines: list[str], start_line: int, end_line: int) -> str:
        return "\n".join(lines[start_line - 1:end_line]).strip()

    @staticmethod
    def _find_block_end(lines: list[str], start_line: int) -> int:
        depth = 0
        started = False
        for index in range(start_line - 1, len(lines)):
            line = lines[index]
            depth += line.count("{")
            if "{" in line:
                started = True
            depth -= line.count("}")
            if started and depth <= 0:
                return index + 1
        return min(len(lines), start_line + 20)

    @staticmethod
    def _resolve_ts_import(current_path: str, value: str) -> str:
        current = Path(current_path)
        base = (current.parent / value).as_posix()
        for suffix in (".ts", ".tsx", "/index.ts", "/index.tsx"):
            candidate = f"{base}{suffix}" if not suffix.startswith("/") else f"{base}{suffix}"
            normalized = Path(candidate).as_posix()
            return normalized
        return ""


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
