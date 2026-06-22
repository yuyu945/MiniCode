# MiniCode

MiniCode is my Python-first local coding agent project. It focuses on an
inspectable agent loop, terminal-first interaction, controlled tool execution,
and runtime feedback mechanisms instead of hiding behavior inside prompts.

## What This Repository Contains

- `minicode/`: the active Python package
- `tests/`: the active pytest suite
- `benchmarks/`: focused benchmark and stress scripts kept for reproducible evaluation
- `.github/workflows/ci.yml`: the CI definition for the root package

This repository is intentionally scoped to the Python implementation that is
packaged and tested from the repository root.

## Core Capabilities

- Local coding-agent loop with model/tool iteration
- Terminal UI and TTY interaction flow
- Permission-aware local tools for files, search, patching, and shell commands
- Runtime controllers for context pressure, progress, memory, and recovery
- Search-oriented code retrieval utilities with benchmark coverage

## Quick Start

```bash
git clone https://github.com/yuyu945/MiniCode.git
cd MiniCode
python -m pip install -e .[dev]
```

Run the CLI:

```bash
minicode-py
```

Or run the module directly:

```bash
python -m minicode.main
```

## Project Entry Points

- `minicode-py` -> `minicode.main:main`
- `minicode-gateway` -> `minicode.gateway:run_gateway`
- `minicode-cron` -> `minicode.cron_runner:main`
- `minicode-headless` -> `minicode.headless:main`

## Validation

Use the same root surfaces that CI uses:

```bash
python -m compileall -q minicode tests benchmarks
pytest -q
```

## Architecture Notes

- `minicode/agent_loop.py` owns the main model/tool loop
- `minicode/tooling.py` and `minicode/tools/` own the tool contract and execution surface
- `minicode/tty_app.py` plus `minicode/tui/` own the terminal UI flow
- `minicode/context_cybernetics.py`, `minicode/memory_pipeline.py`, and related controllers own runtime feedback behavior
- `minicode/code_retrieval.py` owns repository indexing and retrieval benchmarking support

## Design Principles

- Keep behavior inspectable and testable
- Prefer explicit runtime signals over prompt-only heuristics
- Keep permissions and filesystem boundaries visible
- Make benchmark and verification workflows reproducible