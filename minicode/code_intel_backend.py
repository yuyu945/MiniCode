from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minicode.retrieval.code_index import CodeIndex


@dataclass(slots=True)
class CodeIntelResponse:
    ok: bool
    output: str


class IndexCodeIntelBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = CodeIndex().build(root)

    def run(self, operation: str, symbol: str | None = None, file_path: str | None = None) -> CodeIntelResponse:
        if operation in {"go_to_definition", "go_to_implementation"}:
            assert symbol is not None
            definitions = self.index.definition_chunks(symbol)
            if not definitions:
                label = "definitions" if operation == "go_to_definition" else "implementations"
                return CodeIntelResponse(True, f"No {label} found for {symbol}")
            title = "Definitions" if operation == "go_to_definition" else "Implementations"
            lines = [f"{title} for {symbol}:"]
            for chunk in definitions[:20]:
                lines.append(
                    f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                    f"{chunk.symbol_kind} {chunk.symbol_name} :: {chunk.signature}"
                )
            return CodeIntelResponse(True, "\n".join(lines))

        if operation == "find_references":
            assert symbol is not None
            definitions = self.index.definition_chunks(symbol)
            references = self.index.reference_entries(symbol)
            if not definitions and not references:
                return CodeIntelResponse(True, f"No references found for {symbol}")
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
            return CodeIntelResponse(True, "\n".join(lines))

        if operation == "hover":
            assert symbol is not None
            definitions = self.index.definition_chunks(symbol)
            if not definitions:
                return CodeIntelResponse(True, f"No hover information found for {symbol}")
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
            return CodeIntelResponse(True, "\n".join(lines))

        if operation == "workspace_symbol":
            assert symbol is not None
            query = symbol.lower()
            matches = [
                chunk for chunk in self.index.chunks
                if chunk.symbol_kind != "file" and query in chunk.symbol_name.lower()
            ]
            if not matches:
                return CodeIntelResponse(True, f"No workspace symbols found for {symbol}")
            lines = [f"Workspace symbols for {symbol}:"]
            for chunk in matches[:50]:
                lines.append(
                    f"{chunk.path}:{chunk.start_line}-{chunk.end_line} "
                    f"{chunk.symbol_kind} {chunk.symbol_name}"
                )
            return CodeIntelResponse(True, "\n".join(lines))

        assert file_path is not None
        target = (self.root / file_path).resolve()
        if not target.exists():
            return CodeIntelResponse(False, f"File not found: {file_path}")
        rel = target.relative_to(self.root).as_posix()
        chunks = [chunk for chunk in self.index.chunks_by_path.get(rel, []) if chunk.symbol_kind != "file"]
        if not chunks:
            return CodeIntelResponse(True, f"No symbols found in {rel}")
        lines = [f"Document symbols for {rel}:"]
        for chunk in chunks[:100]:
            lines.append(f"{chunk.start_line}-{chunk.end_line} {chunk.symbol_kind} {chunk.symbol_name}")
        return CodeIntelResponse(True, "\n".join(lines))


class ExternalLspCodeIntelBackend:
    def __init__(self, root: Path, command: list[str]) -> None:
        self.root = root
        self.command = command
        self._proc: subprocess.Popen | None = None
        self._next_id = 1

    def run(self, operation: str, symbol: str | None = None, file_path: str | None = None) -> CodeIntelResponse:
        with self._session():
            if operation == "document_symbols":
                assert file_path is not None
                target = (self.root / file_path).resolve()
                result = self._request(
                    "textDocument/documentSymbol",
                    {"textDocument": {"uri": _path_to_uri(target)}},
                )
                return CodeIntelResponse(True, _format_document_symbols(file_path, result))

            assert symbol is not None
            symbols = self._request("workspace/symbol", {"query": symbol}) or []
            if not symbols:
                return CodeIntelResponse(True, f"No results found for {symbol}")
            first = symbols[0]
            location = first.get("location") or {}
            uri = location.get("uri")
            range_ = location.get("range") or {}
            start = (range_.get("start") or {})
            if not uri:
                return CodeIntelResponse(True, f"No results found for {symbol}")
            text_document = {"uri": uri}
            position = {
                "line": int(start.get("line", 0)),
                "character": int(start.get("character", 0)),
            }

            if operation == "workspace_symbol":
                return CodeIntelResponse(True, _format_workspace_symbols(symbol, symbols))
            if operation == "hover":
                result = self._request("textDocument/hover", {"textDocument": text_document, "position": position})
                return CodeIntelResponse(True, _format_hover(symbol, uri, range_, result))
            if operation == "find_references":
                result = self._request(
                    "textDocument/references",
                    {"textDocument": text_document, "position": position, "context": {"includeDeclaration": True}},
                )
                return CodeIntelResponse(True, _format_location_list(f"References for {symbol}:", result))
            if operation == "go_to_implementation":
                result = self._request("textDocument/implementation", {"textDocument": text_document, "position": position})
                return CodeIntelResponse(True, _format_location_list(f"Implementations for {symbol}:", result))
            result = self._request("textDocument/definition", {"textDocument": text_document, "position": position})
            return CodeIntelResponse(True, _format_location_list(f"Definitions for {symbol}:", result))

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


def select_code_intel_backend(root: Path, file_path: str | None = None) -> IndexCodeIntelBackend | ExternalLspCodeIntelBackend:
    command = _configured_lsp_command(root, file_path)
    if command:
        return ExternalLspCodeIntelBackend(root, command)
    return IndexCodeIntelBackend(root)


def _configured_lsp_command(root: Path, file_path: str | None) -> list[str] | None:
    ext = Path(file_path).suffix.lower() if file_path else ""
    if ext == ".py" or (not ext and any(root.rglob("*.py"))):
        value = os.environ.get("MINICODE_PYTHON_LSP_COMMAND")
        if value:
            return json.loads(value)
    if ext in {".ts", ".tsx"} or (not ext and any(root.rglob("*.ts"))):
        value = os.environ.get("MINICODE_TYPESCRIPT_LSP_COMMAND")
        if value:
            return json.loads(value)
    return None


def _path_to_uri(path: Path) -> str:
    normalized = path.resolve().as_posix()
    if normalized.startswith("/"):
        return f"file://{normalized}"
    return f"file:///{normalized}"


def _uri_to_display_path(uri: str) -> str:
    text = uri.replace("file:///", "").replace("file://", "")
    return Path(text).name if text else uri


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
