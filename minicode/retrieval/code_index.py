from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

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
    references: set[str] = field(default_factory=set)


@dataclass(slots=True)
class IndexedReference:
    path: str
    source_symbol: str
    target_symbol: str
    line: int
    kind: str


class CodeIndex:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.chunks: list[IndexedChunk] = []
        self.chunks_by_path: dict[str, list[IndexedChunk]] = {}
        self.symbol_definitions: dict[str, list[IndexedChunk]] = {}
        self.symbol_references: dict[str, list[IndexedReference]] = {}
        self.import_graph: dict[str, set[str]] = {}
        self.failed_files: list[str] = []

    def build(self, workspace: str | Path) -> "CodeIndex":
        self.workspace = Path(workspace)
        self.chunks.clear()
        self.chunks_by_path.clear()
        self.symbol_definitions.clear()
        self.symbol_references.clear()
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

    def get_chunk(self, path: str, symbol_name: str, start_line: int) -> IndexedChunk | None:
        for chunk in self.chunks_by_path.get(path, []):
            if chunk.symbol_name == symbol_name and chunk.start_line == start_line:
                return chunk
        return None

    def definition_chunks(self, symbol: str) -> list[IndexedChunk]:
        return list(self.symbol_definitions.get(symbol.lower(), []))

    def reference_entries(self, symbol: str) -> list[IndexedReference]:
        return list(self.symbol_references.get(symbol.lower(), []))

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
        self._register_chunk(
            self._build_chunk(
                rel,
                "python",
                path.stem,
                "file",
                f"file {rel}",
                1,
                max(1, len(lines)),
                text,
                references=set(),
            )
        )
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._register_python_symbol(rel, lines, node, "class", f"class {node.name}")
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._register_python_symbol(rel, lines, child, "method", f"def {child.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._register_python_symbol(rel, lines, node, "function", f"def {node.name}")

    def _register_python_symbol(
        self,
        rel: str,
        lines: list[str],
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        symbol_kind: str,
        signature: str,
    ) -> None:
        chunk = self._build_chunk(
            rel,
            "python",
            node.name,
            symbol_kind,
            signature,
            node.lineno,
            getattr(node, "end_lineno", node.lineno),
            self._slice(lines, node.lineno, getattr(node, "end_lineno", node.lineno)),
            references=self._python_reference_terms(node),
        )
        self._register_chunk(chunk)
        self._register_references(rel, chunk.symbol_name, self._python_references(rel, chunk.symbol_name, node))

    def _index_ts_file(self, path: Path) -> None:
        assert self.workspace is not None
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(self.workspace).as_posix()
        lines = text.splitlines()
        imports: set[str] = set()
        for match in _TS_IMPORT_RE.finditer(text):
            value = match.group("module") or match.group("side")
            if value:
                resolved = self._resolve_ts_import(rel, value)
                if resolved:
                    imports.add(resolved)
        self.import_graph[rel] = imports
        self._register_chunk(
            self._build_chunk(
                rel,
                "typescript",
                path.stem,
                "file",
                f"file {rel}",
                1,
                max(1, len(lines)),
                text,
                references=set(),
            )
        )
        for match in _TS_SYMBOL_RE.finditer(text):
            kind_text = match.group("kind")
            name = match.group("name")
            kind = "class" if "class" in kind_text else "function"
            start_line = text[: match.start()].count("\n") + 1
            end_line = self._find_block_end(lines, start_line)
            content = self._slice(lines, start_line, end_line)
            chunk = self._build_chunk(
                rel,
                "typescript",
                name,
                kind,
                match.group(0).strip(),
                start_line,
                end_line,
                content,
                references=self._ts_reference_terms(content, name),
            )
            self._register_chunk(chunk)
            self._register_references(rel, chunk.symbol_name, self._ts_references(rel, chunk.symbol_name, content, start_line))

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
        references: set[str],
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
            references=references,
        )

    def _register_chunk(self, chunk: IndexedChunk) -> None:
        self.chunks.append(chunk)
        self.chunks_by_path.setdefault(chunk.path, []).append(chunk)
        if chunk.symbol_kind != "file":
            self.symbol_definitions.setdefault(chunk.symbol_name.lower(), []).append(chunk)

    def _register_references(
        self,
        path: str,
        source_symbol: str,
        references: Iterable[IndexedReference],
    ) -> None:
        del path, source_symbol
        for reference in references:
            self.symbol_references.setdefault(reference.target_symbol.lower(), []).append(reference)

    def _python_reference_terms(self, node: ast.AST) -> set[str]:
        return {reference.target_symbol.lower() for reference in self._python_references("", "", node)}

    def _python_references(self, path: str, source_symbol: str, node: ast.AST) -> list[IndexedReference]:
        refs: list[IndexedReference] = []
        for child in ast.walk(node):
            line = getattr(child, "lineno", getattr(node, "lineno", 1))
            if isinstance(child, ast.Call):
                target = _python_callable_name(child.func)
                if target:
                    refs.append(IndexedReference(path=path, source_symbol=source_symbol, target_symbol=target, line=line, kind="call"))
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                refs.append(IndexedReference(path=path, source_symbol=source_symbol, target_symbol=child.id, line=line, kind="name"))
            elif isinstance(child, ast.Attribute):
                refs.append(IndexedReference(path=path, source_symbol=source_symbol, target_symbol=child.attr, line=line, kind="attribute"))
        return refs

    @staticmethod
    def _ts_reference_terms(content: str, declared_symbol: str) -> set[str]:
        return {
            reference.target_symbol.lower()
            for reference in CodeIndex._ts_references("", declared_symbol, content, 1)
        }

    @staticmethod
    def _ts_references(path: str, source_symbol: str, content: str, start_line: int) -> list[IndexedReference]:
        refs: list[IndexedReference] = []
        declared = source_symbol.lower()
        keywords = {
            "export", "class", "function", "const", "let", "var", "return", "if", "else",
            "for", "while", "switch", "case", "break", "continue", "new", "this", "true",
            "false", "null", "undefined", "async", "await", "try", "catch", "throw",
        }
        for offset, line in enumerate(content.splitlines()):
            for token in _tokenize(line):
                if token == declared or token in keywords:
                    continue
                refs.append(
                    IndexedReference(
                        path=path,
                        source_symbol=source_symbol,
                        target_symbol=token,
                        line=start_line + offset,
                        kind="name",
                    )
                )
        return refs

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

    def _resolve_ts_import(self, current_path: str, value: str) -> str:
        assert self.workspace is not None
        current = Path(current_path)
        base = (current.parent / value).as_posix()
        fallback = ""
        for suffix in (".ts", ".tsx", "/index.ts", "/index.tsx"):
            candidate = f"{base}{suffix}" if not suffix.startswith("/") else f"{base}{suffix}"
            normalized = Path(candidate).as_posix()
            fallback = normalized
            if (self.workspace / normalized).exists():
                return normalized
        return fallback


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    tokens: list[str] = []
    for token in raw_tokens:
        lower = token.lower()
        tokens.append(lower)
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
        parts = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", split)
        tokens.extend(part.lower() for part in parts if part.lower() != lower)
    return tokens


def _python_callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
