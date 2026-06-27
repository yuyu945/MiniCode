from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from minicode.config import load_effective_settings
from minicode.retrieval.code_index import CodeIndex


@dataclass(slots=True)
class CodeIntelResponse:
    ok: bool
    output: str
    backend: str


class CodeIntelBackend(Protocol):
    backend_name: str

    def run(self, operation: str, symbol: str | None = None, file_path: str | None = None) -> CodeIntelResponse: ...


@dataclass(slots=True)
class BackendRoute:
    language: str
    env_var: str
    extensions: tuple[str, ...]
    backend_name: str


class IndexCodeIntelBackend:
    backend_name = "index_fallback"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = CodeIndex().build(root)

    def run(self, operation: str, symbol: str | None = None, file_path: str | None = None) -> CodeIntelResponse:
        if operation in {"go_to_definition", "go_to_implementation"}:
            assert symbol is not None
            definitions = self.index.definition_chunks(symbol)
            if not definitions:
                label = "definitions" if operation == "go_to_definition" else "implementations"
                return CodeIntelResponse(True, f"No {label} found for {symbol}", self.backend_name)
            title = "Definitions" if operation == "go_to_definition" else "Implementations"
            lines = [f"{title} for {symbol}:"]
            for chunk in definitions[:20]:
                lines.append(
                    f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                    f"{chunk.symbol_kind} {chunk.symbol_name} :: {chunk.signature}"
                )
            return CodeIntelResponse(True, "\n".join(lines), self.backend_name)

        if operation == "find_references":
            assert symbol is not None
            definitions = self.index.definition_chunks(symbol)
            references = self.index.reference_entries(symbol)
            if not definitions and not references:
                return CodeIntelResponse(True, f"No references found for {symbol}", self.backend_name)
            lines = [f"References for {symbol}:"]
            if definitions:
                lines.append("Definitions:")
                for chunk in definitions[:20]:
                    lines.append(
                        f"  {chunk.path}:{chunk.start_line}-{chunk.end_line} "
                        f"{chunk.symbol_kind} {chunk.symbol_name}"
                    )
            if references:
                lines.append("Usages:")
                for ref in references[:50]:
                    lines.append(
                        f"  {ref.path}:{ref.line} {ref.kind} from {ref.source_symbol or '?'}"
                    )
            return CodeIntelResponse(True, "\n".join(lines), self.backend_name)

        if operation == "hover":
            assert symbol is not None
            definitions = self.index.definition_chunks(symbol)
            if not definitions:
                return CodeIntelResponse(True, f"No hover information found for {symbol}", self.backend_name)
            chunk = definitions[0]
            snippet = "\n".join(chunk.content.splitlines()[:8]).strip()
            lines = [
                f"Hover for {symbol}:",
                f"Location: {chunk.path}:{chunk.start_line}-{chunk.end_line}",
                f"Kind: {chunk.symbol_kind}",
                f"Signature: {chunk.signature}",
            ]
            if snippet:
                lines.extend(["", snippet])
            return CodeIntelResponse(True, "\n".join(lines), self.backend_name)

        if operation == "workspace_symbol":
            assert symbol is not None
            query = symbol.lower()
            matches = [
                chunk for chunk in self.index.chunks
                if chunk.symbol_kind != "file" and query in chunk.symbol_name.lower()
            ]
            if not matches:
                return CodeIntelResponse(True, f"No workspace symbols found for {symbol}", self.backend_name)
            lines = [f"Workspace symbols for {symbol}:"]
            for chunk in matches[:50]:
                lines.append(
                    f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                    f"{chunk.symbol_kind} {chunk.symbol_name}"
                )
            return CodeIntelResponse(True, "\n".join(lines), self.backend_name)

        assert file_path is not None
        target = (self.root / file_path).resolve()
        if not target.exists():
            return CodeIntelResponse(False, f"File not found: {file_path}", self.backend_name)
        rel = target.relative_to(self.root).as_posix()
        chunks = [chunk for chunk in self.index.chunks_by_path.get(rel, []) if chunk.symbol_kind != "file"]
        if not chunks:
            return CodeIntelResponse(True, f"No symbols found in {rel}", self.backend_name)
        lines = [f"Document symbols for {rel}:"]
        for chunk in chunks[:100]:
            lines.append(f"{chunk.start_line}-{chunk.end_line} {chunk.symbol_kind} {chunk.symbol_name}")
        return CodeIntelResponse(True, "\n".join(lines), self.backend_name)


class ExternalLspCodeIntelBackend:
    def __init__(self, root: Path, command: list[str], backend_name: str) -> None:
        self.root = root
        self.command = command
        self.backend_name = backend_name
        self._proc: subprocess.Popen | None = None
        self._next_id = 1
        self._index_locator = IndexCodeIntelBackend(root)

    def run(self, operation: str, symbol: str | None = None, file_path: str | None = None) -> CodeIntelResponse:
        with self._session():
            if operation == "document_symbols":
                assert file_path is not None
                target = (self.root / file_path).resolve()
                self._open_file(target)
                result = self._request(
                    "textDocument/documentSymbol",
                    {"textDocument": {"uri": _path_to_uri(target)}},
                )
                return CodeIntelResponse(True, _format_document_symbols(file_path, result), self.backend_name)

            assert symbol is not None
            symbols: list[dict[str, Any]] = []
            try:
                symbols = self._request("workspace/symbol", {"query": symbol}) or []
            except Exception:
                symbols = []

            location_info = self._resolve_symbol_location(symbol, symbols)
            if operation == "workspace_symbol":
                if symbols:
                    return CodeIntelResponse(True, _format_workspace_symbols(symbol, symbols), self.backend_name)
                return self._index_locator.run("workspace_symbol", symbol=symbol)
            if location_info is None:
                return CodeIntelResponse(True, f"No results found for {symbol}", self.backend_name)

            text_document = {"uri": location_info["uri"]}
            self._prime_related_files(location_info["uri"], symbol)
            position = location_info["position"]
            range_ = location_info["range"]
            if operation == "hover":
                result = self._request("textDocument/hover", {"textDocument": text_document, "position": position})
                return CodeIntelResponse(True, _format_hover(symbol, location_info["uri"], range_, result), self.backend_name)
            if operation == "find_references":
                result = self._request(
                    "textDocument/references",
                    {"textDocument": text_document, "position": position, "context": {"includeDeclaration": True}},
                )
                return CodeIntelResponse(
                    True,
                    self._format_references(symbol, result),
                    self.backend_name,
                )
            if operation == "go_to_implementation":
                result = self._request("textDocument/implementation", {"textDocument": text_document, "position": position})
                return CodeIntelResponse(True, _format_location_list(f"Implementations for {symbol}:", result), self.backend_name)
            result = self._request("textDocument/definition", {"textDocument": text_document, "position": position})
            return CodeIntelResponse(True, _format_location_list(f"Definitions for {symbol}:", result), self.backend_name)

    def _resolve_symbol_location(
        self,
        symbol: str,
        workspace_symbols: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if workspace_symbols:
            first = workspace_symbols[0]
            location = first.get("location") or {}
            uri = location.get("uri")
            range_ = location.get("range") or {}
            start = (range_.get("start") or {})
            if uri:
                return {
                    "uri": uri,
                    "range": range_,
                    "position": {
                        "line": int(start.get("line", 0)),
                        "character": int(start.get("character", 0)),
                    },
                }

        definitions = self._index_locator.index.definition_chunks(symbol)
        references = self._index_locator.index.reference_entries(symbol)
        if references:
            reference = references[0]
            uri = _path_to_uri((self.root / reference.path).resolve())
            line_number = max(1, reference.line)
            char = _find_symbol_character((self.root / reference.path).resolve(), line_number, symbol)
            range_ = {
                "start": {"line": line_number - 1, "character": char},
                "end": {"line": line_number - 1, "character": char + len(symbol)},
            }
            return {
                "uri": uri,
                "range": range_,
                "position": dict(range_["start"]),
            }
        if not definitions:
            return None
        chunk = definitions[0]
        uri = _path_to_uri((self.root / chunk.path).resolve())
        char = _find_symbol_character((self.root / chunk.path).resolve(), chunk.start_line, symbol)
        range_ = {
            "start": {"line": max(0, chunk.start_line - 1), "character": char},
            "end": {"line": max(0, chunk.start_line - 1), "character": char + len(symbol)},
        }
        return {
            "uri": uri,
            "range": range_,
            "position": dict(range_["start"]),
        }

    def _prime_related_files(self, symbol_uri: str, symbol: str) -> None:
        target = _uri_to_path(symbol_uri)
        related_paths: set[Path] = set()
        if target.exists():
            related_paths.add(target)
            try:
                rel = target.relative_to(self.root).as_posix()
            except ValueError:
                rel = ""
            if rel:
                related_paths.update(self._related_workspace_files(rel, symbol))
        else:
            for chunk in self._index_locator.index.definition_chunks(symbol):
                related_paths.add((self.root / chunk.path).resolve())
                related_paths.update(self._related_workspace_files(chunk.path, symbol))
        for path in sorted(related_paths):
            self._open_file(path)

    def _related_workspace_files(self, rel_path: str, symbol: str) -> set[Path]:
        related = {(self.root / rel_path).resolve()}
        for imported in self._index_locator.index.import_graph.get(rel_path, set()):
            related.add((self.root / imported).resolve())
        for candidate, imported in self._index_locator.index.import_graph.items():
            if rel_path in imported:
                related.add((self.root / candidate).resolve())
        for reference in self._index_locator.index.reference_entries(symbol):
            related.add((self.root / reference.path).resolve())
        for definition in self._index_locator.index.definition_chunks(symbol):
            related.add((self.root / definition.path).resolve())
        return {path for path in related if path.exists()}

    def _format_references(self, symbol: str, result: Any) -> str:
        items = result if isinstance(result, list) else ([result] if result else [])
        if not items:
            return f"References for {symbol}:\nNo results found"
        ref_map = {}
        for reference in self._index_locator.index.reference_entries(symbol):
            ref_map[(reference.path, reference.line)] = self._reference_label(reference.path, reference.line, reference.source_symbol)
        lines = [f"References for {symbol}:"]
        for item in items[:50]:
            uri = item.get("uri") or item.get("targetUri")
            range_ = item.get("range") or item.get("targetRange") or {}
            start = (range_.get("start") or {})
            end = (range_.get("end") or {})
            path = _uri_to_path(uri)
            rel_path = _relative_display_path(self.root, path)
            line_number = int(start.get("line", 0)) + 1
            source_symbol = ref_map.get((rel_path, line_number))
            suffix = f" :: {source_symbol}" if source_symbol else ""
            lines.append(
                f"{Path(rel_path).name}:{line_number}-{int(end.get('line', 0)) + 1}{suffix}"
            )
        return "\n".join(lines)

    def _reference_label(self, rel_path: str, line_number: int, source_symbol: str) -> str:
        chunks = self._index_locator.index.chunks_by_path.get(rel_path, [])
        method_chunk = next(
            (
                chunk
                for chunk in chunks
                if chunk.symbol_name == source_symbol
                and chunk.symbol_kind in {"method", "function"}
                and chunk.start_line <= line_number <= chunk.end_line
            ),
            None,
        )
        if method_chunk is None:
            return source_symbol
        owner = next(
            (
                chunk.symbol_name
                for chunk in chunks
                if chunk.symbol_kind == "class"
                and chunk.start_line <= method_chunk.start_line <= chunk.end_line
            ),
            None,
        )
        if owner and method_chunk.symbol_kind == "method":
            return f"{owner}.{source_symbol}"
        return source_symbol

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.root),
        )
        self._request(
            "initialize",
            {
                "processId": None,
                "rootUri": _path_to_uri(self.root.resolve()),
                "capabilities": {},
            },
        )
        self._notify("initialized", {})

    def _stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._request("shutdown", None)
        except Exception:
            pass
        try:
            self._notify("exit", {})
        except Exception:
            pass
        try:
            self._proc.terminate()
        except Exception:
            pass
        self._proc = None

    def _session(self):
        backend = self

        class _Session:
            def __enter__(self_nonlocal):
                backend._start()
                return backend

            def __exit__(self_nonlocal, exc_type, exc, tb):
                backend._stop()
                return False

        return _Session()

    def _notify(self, method: str, params: Any) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _open_file(self, path: Path) -> None:
        if not path.exists():
            return
        language_id = _language_id_for_path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": _path_to_uri(path),
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def _request(self, method: str, params: Any) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            response = self._read()
            if response.get("id") == request_id:
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                return response.get("result")

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._proc.stdin.write(header)
        self._proc.stdin.write(body)
        self._proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("LSP server closed stdout")
            if line in (b"\r\n", b"\n"):
                break
            key, value = line.decode("utf-8").split(":", 1)
            headers[key.strip().lower()] = value.strip()
        length = int(headers["content-length"])
        body = self._proc.stdout.read(length)
        return json.loads(body.decode("utf-8"))


class CodeIntelBackendRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._fallback = IndexCodeIntelBackend(root)
        self._routes = [
            BackendRoute(
                language="python",
                env_var="MINICODE_PYTHON_LSP_COMMAND",
                extensions=(".py",),
                backend_name="python_external_lsp",
            ),
            BackendRoute(
                language="typescript",
                env_var="MINICODE_TYPESCRIPT_LSP_COMMAND",
                extensions=(".ts", ".tsx"),
                backend_name="typescript_external_lsp",
            ),
        ]

    @property
    def fallback(self) -> IndexCodeIntelBackend:
        return self._fallback

    def select(self, file_path: str | None = None) -> CodeIntelBackend:
        route = self._select_route(file_path)
        if route is None:
            return self._fallback
        value = _get_lsp_command_value(route.env_var, self.root)
        if not value:
            return self._fallback
        return ExternalLspCodeIntelBackend(self.root, json.loads(value), route.backend_name)

    def diagnostics(self) -> dict[str, dict[str, str]]:
        data: dict[str, dict[str, str]] = {}
        for route in self._routes:
            value = _get_lsp_command_value(route.env_var, self.root)
            data[route.language] = {
                "configured": "yes" if value else "no",
                "mode": route.backend_name if value else self._fallback.backend_name,
                "has_sources": "yes" if any(any(self.root.rglob(f"*{ext}")) for ext in route.extensions) else "no",
            }
        return data

    def _select_route(self, file_path: str | None) -> BackendRoute | None:
        ext = Path(file_path).suffix.lower() if file_path else ""
        if ext:
            for route in self._routes:
                if ext in route.extensions:
                    return route
            return None

        for route in self._routes:
            if any(any(self.root.rglob(f"*{suffix}")) for suffix in route.extensions):
                value = _get_lsp_command_value(route.env_var, self.root)
                if value:
                    return route
        return None


def select_code_intel_backend(root: Path, file_path: str | None = None) -> CodeIntelBackend:
    return CodeIntelBackendRegistry(root).select(file_path)


def get_lsp_backend_diagnostics(root: Path) -> dict[str, dict[str, str]]:
    return CodeIntelBackendRegistry(root).diagnostics()


def _get_lsp_command_value(name: str, root: Path) -> str:
    env_value = os.environ.get(name, "").strip()
    if env_value:
        return env_value
    try:
        effective = load_effective_settings(root)
        env_section = effective.get("env", {}) if isinstance(effective, dict) else {}
        value = env_section.get(name, "")
        return str(value).strip() if value else ""
    except Exception:
        return ""


def _path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _uri_to_display_path(uri: str) -> str:
    path = _uri_to_path(uri)
    return path.name if str(path) else uri


def _relative_display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    text = unquote(parsed.path if parsed.scheme else uri.replace("file://", ""))
    if re.match(r"^/[A-Za-z]:", text):
        text = text[1:]
    return Path(text)


def _language_id_for_path(path: Path) -> str:
    if path.suffix.lower() == ".py":
        return "python"
    if path.suffix.lower() == ".tsx":
        return "typescriptreact"
    if path.suffix.lower() == ".ts":
        return "typescript"
    return "plaintext"


def _find_symbol_character(path: Path, line_number: int, symbol: str) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    if not (1 <= line_number <= len(lines)):
        return 0
    idx = lines[line_number - 1].find(symbol)
    return idx if idx >= 0 else 0


def _format_location_list(title: str, result: Any) -> str:
    items = result if isinstance(result, list) else ([result] if result else [])
    if not items:
        return f"{title}\nNo results found"
    lines = [title]
    for item in items[:50]:
        uri = item.get("uri") or item.get("targetUri")
        range_ = item.get("range") or item.get("targetRange") or {}
        start = (range_.get("start") or {})
        end = (range_.get("end") or {})
        lines.append(
            f"{_uri_to_display_path(uri)}:{int(start.get('line', 0)) + 1}-{int(end.get('line', 0)) + 1}"
        )
    return "\n".join(lines)


def _format_workspace_symbols(symbol: str, result: Any) -> str:
    items = result if isinstance(result, list) else []
    if not items:
        return f"No workspace symbols found for {symbol}"
    lines = [f"Workspace symbols for {symbol}:"]
    for item in items[:50]:
        location = item.get("location") or {}
        uri = location.get("uri")
        range_ = location.get("range") or {}
        start = (range_.get("start") or {})
        end = (range_.get("end") or {})
        lines.append(
            f"{_uri_to_display_path(uri)}:{int(start.get('line', 0)) + 1}-{int(end.get('line', 0)) + 1} "
            f"{item.get('name', symbol)}"
        )
    return "\n".join(lines)


def _format_hover(symbol: str, uri: str, range_: dict[str, Any], result: Any) -> str:
    start = (range_.get("start") or {})
    end = (range_.get("end") or {})
    contents = result.get("contents") if isinstance(result, dict) else result
    if isinstance(contents, dict):
        text = contents.get("value") or json.dumps(contents)
    elif isinstance(contents, list):
        text = "\n".join(item.get("value", str(item)) if isinstance(item, dict) else str(item) for item in contents)
    else:
        text = str(contents)
    lines = [
        f"Hover for {symbol}:",
        f"Location: {_uri_to_display_path(uri)}:{int(start.get('line', 0)) + 1}-{int(end.get('line', 0)) + 1}",
        "",
        text,
    ]
    return "\n".join(lines)


def _format_document_symbols(file_path: str, result: Any) -> str:
    items = result if isinstance(result, list) else []
    if not items:
        return f"No symbols found in {file_path}"
    lines = [f"Document symbols for {file_path}:"]
    for item in items[:100]:
        range_ = item.get("range") or {}
        start = (range_.get("start") or {})
        end = (range_.get("end") or {})
        lines.append(
            f"{int(start.get('line', 0)) + 1}-{int(end.get('line', 0)) + 1} "
            f"{item.get('name', '?')}"
        )
    return "\n".join(lines)
