"""
Targeted stress test for resume metrics.
Measures: memory search throughput/precision, context compression ratio,
agent loop multi-turn stability, token estimation throughput.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minicode.config import MINI_CODE_DIR
from minicode.context_manager import estimate_tokens, estimate_message_tokens, ContextManager
from minicode.context.compactor import (
    ContextCompactor, AutoCompactConfig, CompactionResult, CompactStrategy,
    ReadDedupManager, ToolResultBudgetManager,
)
from minicode.memory import MemoryManager, MemoryScope, MemoryEntry


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def fmt(n: float, decimals: int = 2) -> str:
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{decimals}f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.{decimals}f}K"
    return f"{n:.{decimals}f}"


# ---------------------------------------------------------------------------
# 1. Memory BM25 Search Throughput & Recall
# ---------------------------------------------------------------------------

def stress_memory_search() -> dict:
    header("1. Memory BM25 Search Performance")

    results = {}

    # Isolate from real user memory by temporarily overriding MINI_CODE_DIR
    import minicode.memory as mem_module
    import minicode.config as config_module
    real_mini_code_dir = config_module.MINI_CODE_DIR
    real_user_paths_func = mem_module.MemoryPaths.for_workspace

    for corpus_size in [100, 500, 1000, 2000]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Override to use temp dir for ALL memory scopes
            config_module.MINI_CODE_DIR = tmp_path / ".mini-code"
            def _isolated_paths(workspace):
                return mem_module.MemoryPaths(
                    user_memory=tmp_path / ".mini-code" / "memory",
                    project_memory=tmp_path / ".mini-code-memory",
                    local_memory=tmp_path / ".mini-code-memory-local",
                )
            mem_module.MemoryPaths.for_workspace = staticmethod(_isolated_paths)

            mgr = MemoryManager(project_root=tmpdir)

            # Populate with realistic entries
            categories = ["architecture", "code-pattern", "convention",
                          "testing", "performance", "security", "workflow"]
            templates = [
                "Use {tool} for {task} operations in {domain}",
                "Always run tests with {flag} before committing {target}",
                "The {component} module should follow {pattern} pattern",
                "Avoid using {bad_practice} in {context}, prefer {good_practice}",
                "When deploying to {env}, run {command} first",
            ]
            fillers = {
                "tool": ["pytest", "ruff", "mypy", "pre-commit", "docker"],
                "task": ["lint", "format", "build", "test", "deploy"],
                "domain": ["backend", "frontend", "auth", "database", "api"],
                "flag": ["--verbose", "--strict", "--parallel", "--coverage"],
                "target": ["code", "migrations", "config", "dependencies"],
                "component": ["auth", "router", "storage", "cache", "gateway"],
                "pattern": ["factory", "strategy", "observer", "decorator", "adapter"],
                "bad_practice": ["global state", "magic numbers", "god objects"],
                "context": ["production", "testing", "development", "CI/CD"],
                "good_practice": ["dependency injection", "explicit params"],
                "env": ["staging", "production", "local", "CI"],
                "command": ["migrate", "seed", "verify", "clean", "build"],
            }

            import random
            random.seed(42)

            for i in range(corpus_size):
                template = random.choice(templates)
                content = template.format(**{k: random.choice(v) for k, v in fillers.items()})
                mgr.add_entry(
                    scope=MemoryScope.PROJECT,  # Only use project scope for isolation
                    category=random.choice(categories),
                    content=content,
                    tags=[random.choice(list(fillers["tool"]))],
                )

            # Benchmark search
            queries = ["deploy configuration", "testing pattern", "auth security",
                       "performance optimization", "database migration"]
            latencies = []

            for _ in range(5):
                for q in queries:
                    start = time.perf_counter()
                    hits = mgr.search(q, limit=20)
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append((elapsed, len(hits)))

            avg_latency = statistics.mean(l[0] for l in latencies)
            avg_hits = statistics.mean(l[1] for l in latencies)
            p99_latency = sorted(l[0] for l in latencies)[int(len(latencies) * 0.99)]

            print(f"  Corpus={corpus_size:>5}: avg={avg_latency:.2f}ms, "
                  f"p99={p99_latency:.2f}ms, hits={avg_hits:.1f}")

            results[corpus_size] = {
                "avg_latency_ms": round(avg_latency, 2),
                "p99_latency_ms": round(p99_latency, 2),
                "avg_hits": round(avg_hits, 1),
            }

    # Restore
    config_module.MINI_CODE_DIR = real_mini_code_dir
    mem_module.MemoryPaths.for_workspace = real_user_paths_func

    return results


# ---------------------------------------------------------------------------
# 2. Context Compression Ratio
# ---------------------------------------------------------------------------

def build_coding_task_scenarios() -> list[dict]:
    """Build labelled multi-turn coding-task transcripts for compression tests."""
    scenarios = []
    round_plan = [18, 20, 22, 24, 26, 28, 32, 40, 48, 56]
    task_catalog = [
        {
            "slug": "auth_session",
            "goal": "fix auth/session regression and produce a safe patch",
            "constraints": [
                "keep public API unchanged",
                "preserve chinese final response",
                "run focused tests before claiming success",
            ],
            "plan": [
                "inspect auth entrypoint",
                "trace session loader",
                "patch retry/expiry flow",
            ],
            "todos": [
                "verify failing pytest case",
                "confirm token refresh edge case",
            ],
            "results": [
                "pytest auth_flow failure token expired not redirected",
                "grep validate_token referenced by login handler",
            ],
            "read_path": "src/auth/module_{slot}.py",
            "read_body": "def authenticate(token):\n    return validate_token(token)\n",
            "grep_label": "validate_token",
            "grep_body": "src/auth/module_{slot}.py:{line} validate_token(token)\n",
            "shell_body": "FAILED tests/test_auth_flow.py::test_case_{slot}\n",
            "edit_path": "src/auth/module_{slot}.py",
        },
        {
            "slug": "context_compaction",
            "goal": "diagnose context overflow and stabilize compaction behavior",
            "constraints": [
                "do not regress compaction history",
                "keep context summary readable",
                "preserve recent critical tool outputs",
            ],
            "plan": [
                "inspect context compactor pipeline",
                "compare auto compact and reactive compact paths",
                "validate token accounting",
            ],
            "todos": [
                "confirm boundary metadata survives compaction",
                "re-run overflow reproduction script",
            ],
            "results": [
                "prompt too long reproduced after repeated tool_result growth",
                "auto compact boundary count increased during overflow path",
            ],
            "read_path": "minicode/context_compactor.py",
            "read_body": "class ContextCompactor:\n    def process_request(self, messages):\n        return messages\n",
            "grep_label": "compact",
            "grep_body": "minicode/context_compactor.py:{line} compact boundary summary\n",
            "shell_body": "FAILED tests/test_context_compactor.py::test_overflow_case_{slot}\n",
            "edit_path": "minicode/context_compactor.py",
        },
        {
            "slug": "memory_pipeline",
            "goal": "improve memory retrieval quality without breaking prompt injection",
            "constraints": [
                "keep memory scopes intact",
                "do not remove reranker fallback",
                "preserve write-back after reflection",
            ],
            "plan": [
                "inspect memory pipeline read path",
                "compare reranker and vector path",
                "verify write-back after task completion",
            ],
            "todos": [
                "confirm domain filtering still works",
                "check reflection output category mapping",
            ],
            "results": [
                "memory search returns noisy project entries for vague query",
                "reranker summary improves top candidates before injection",
            ],
            "read_path": "minicode/memory_pipeline.py",
            "read_body": "class MemoryPipeline:\n    def read(self, task_description, current_files=None):\n        return []\n",
            "grep_label": "memory",
            "grep_body": "minicode/memory_pipeline.py:{line} memory pipeline read inject write\n",
            "shell_body": "FAILED tests/test_memory_integration.py::test_memory_case_{slot}\n",
            "edit_path": "minicode/memory_pipeline.py",
        },
        {
            "slug": "permissions_tui",
            "goal": "fix permission prompt flow in tty interaction",
            "constraints": [
                "preserve approval semantics",
                "keep UI responsive during prompt handling",
                "avoid changing tool permission defaults",
            ],
            "plan": [
                "trace tty approval event flow",
                "inspect permission manager decision path",
                "validate feedback-mode interaction",
            ],
            "todos": [
                "replay deny-with-feedback path",
                "confirm pending approval state clears correctly",
            ],
            "results": [
                "tty approval dialog stays open after deny_with_feedback",
                "permission summary missing command details in transcript",
            ],
            "read_path": "minicode/tty_app.py",
            "read_body": "def run_tty_app(args=None):\n    return None\n",
            "grep_label": "approval",
            "grep_body": "minicode/tty_app.py:{line} approval_event approval_result pending_approval\n",
            "shell_body": "FAILED tests/test_tty_app.py::test_permission_case_{slot}\n",
            "edit_path": "minicode/tty_app.py",
        },
        {
            "slug": "routing_models",
            "goal": "improve model routing and fallback decisions under failure pressure",
            "constraints": [
                "preserve current provider config format",
                "keep retry classification stable",
                "do not break smart router learning hooks",
            ],
            "plan": [
                "inspect agent router and model registry",
                "trace smart router decision path",
                "validate fallback adapter selection",
            ],
            "todos": [
                "reproduce provider failure branch",
                "check model switch history recording",
            ],
            "results": [
                "router selected expensive model for moderate task complexity",
                "fallback path skipped expected adapter after API failure",
            ],
            "read_path": "minicode/agent_router.py",
            "read_body": "class AgentRouter:\n    def route(self, task_profile):\n        return None\n",
            "grep_label": "model",
            "grep_body": "minicode/agent_router.py:{line} selected_model routing tier model switch\n",
            "shell_body": "FAILED tests/test_agent_loop.py::test_model_route_case_{slot}\n",
            "edit_path": "minicode/model_registry.py",
        },
    ]

    for idx, num_rounds in enumerate(round_plan, start=1):
        task_template = task_catalog[(idx - 1) % len(task_catalog)]
        task_id = f"task_{idx:02d}"
        task_goal = f"GOAL::{task_id} {task_template['goal']}"
        hard_constraints = [
            f"CONSTRAINT::{task_id} {constraint}"
            for constraint in task_template["constraints"]
        ]
        current_plan = [
            f"PLAN::{task_id} {plan_item}"
            for plan_item in task_template["plan"]
        ]
        pending_todos = [
            f"TODO::{task_id} {todo}"
            for todo in task_template["todos"]
        ]
        recent_critical_results = [
            f"RESULT::{task_id} {result}"
            for result in task_template["results"]
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a coding assistant.\n"
                + task_goal + "\n"
                + "\n".join(hard_constraints),
            }
        ]

        for i in range(num_rounds):
            messages.append({
                "role": "user",
                "content": (
                    f"Round {i}: inspect auth and session bugs, explain the root cause, "
                    f"and prepare a patch with regression coverage. "
                ) + ("user-context " * (16 + (idx % 4))),
            })
            messages.append({
                "role": "assistant",
                "content": (
                    f"I will inspect the relevant files for round {i}, check failing tests, "
                    f"and compare current behavior against the expected flow. "
                ) + ("assistant-plan " * (8 + (i % 3))),
            })
            messages.append({
                "role": "tool_result",
                "toolUseId": f"{task_id}_read_{i}",
                "toolName": "read_file",
                "content": (
                    task_template["read_path"].format(slot=i % 5) + "\n"
                    + (task_template["read_body"] * 42)
                ),
            })
            messages.append({
                "role": "tool_result",
                "toolUseId": f"{task_id}_grep_{i}",
                "toolName": "grep_files",
                "content": (
                    f"grep results for {task_template['grep_label']} round {i}\n"
                    + (task_template["grep_body"].format(slot=i % 5, line=10 + i) * 30)
                ),
            })
            messages.append({
                "role": "tool_result",
                "toolUseId": f"{task_id}_shell_{i}",
                "toolName": "run_command",
                "content": (
                    f"pytest output for round {i}\n"
                    + (task_template["shell_body"].format(slot=i % 7) * 24)
                ),
            })
            if i % 4 == 0:
                messages.append({
                    "role": "tool_result",
                    "toolUseId": f"{task_id}_edit_{i}",
                    "toolName": "edit_file",
                    "content": (
                        f"Applied patch to {task_template['edit_path'].format(slot=i % 5)}\n"
                        + ("- old line\n+ new line\n" * 28)
                    ),
                })

        messages.append({
            "role": "assistant",
            "content": "\n".join(current_plan),
        })
        messages.append({
            "role": "assistant",
            "content": "\n".join(pending_todos),
        })
        messages.append({
            "role": "tool_result",
            "toolUseId": f"{task_id}_recent_results",
            "toolName": "run_command",
            "content": "\n".join(recent_critical_results),
        })

        scenarios.append({
            "task_id": task_id,
            "name": f"multiturn_rounds_{num_rounds}",
            "task_goal": task_goal,
            "hard_constraints": hard_constraints,
            "current_plan": current_plan,
            "pending_todos": pending_todos,
            "recent_critical_results": recent_critical_results,
            "messages": messages,
        })

    return scenarios


def _flatten_messages(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(str(content))
    return "\n".join(parts)


def build_real_session_scenarios(limit: int = 6) -> list[dict]:
    """Sample real saved sessions from the local MiniCode session store."""
    sessions_dir = MINI_CODE_DIR / "sessions"
    if not sessions_dir.exists():
        return []

    scenarios = []
    session_files = sorted(
        [path for path in sessions_dir.glob("*.json") if path.name != "new.json.gz"],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for session_file in session_files:
        if len(scenarios) >= limit:
            break
        try:
            payload = json.loads(session_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        workspace = str(payload.get("workspace", ""))
        if not workspace.startswith("D:\\Python\\agent"):
            continue

        messages = payload.get("messages", [])
        if len(messages) < 8:
            continue
        if not any(message.get("role") == "tool_result" for message in messages):
            continue

        user_messages = [
            str(message.get("content", "")).strip()
            for message in messages
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        if not user_messages:
            continue
        assistant_messages = [
            str(message.get("content", "")).strip()
            for message in messages
            if message.get("role") == "assistant" and str(message.get("content", "")).strip()
        ]
        tool_results = [
            str(message.get("content", "")).strip()
            for message in messages
            if message.get("role") == "tool_result" and str(message.get("content", "")).strip()
        ]

        task_id = f"real_{payload.get('session_id', session_file.stem)}"
        task_goal = f"GOAL::{task_id} {user_messages[0][:160]}"

        hard_constraints = []
        system_content = next(
            (str(message.get("content", "")) for message in messages if message.get("role") == "system"),
            "",
        )
        for line in system_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(marker in line.lower() for marker in ["current cwd", "keep", "prefer", "do not", "when making code changes"]):
                hard_constraints.append(f"CONSTRAINT::{task_id} {line[:180]}")
            if len(hard_constraints) >= 3:
                break
        if not hard_constraints:
            hard_constraints = [f"CONSTRAINT::{task_id} preserve sampled session context"]

        plan_lines = []
        for text in assistant_messages:
            for line in text.splitlines():
                stripped = line.strip("-* \t")
                if any(keyword in stripped.lower() for keyword in ["inspect", "check", "compare", "fix", "investigate", "look", "trace"]):
                    plan_lines.append(f"PLAN::{task_id} {stripped[:160]}")
                if len(plan_lines) >= 3:
                    break
            if len(plan_lines) >= 3:
                break
        if not plan_lines:
            plan_lines = [f"PLAN::{task_id} continue working on sampled session task"]

        pending_todos = [f"TODO::{task_id} review remaining sampled session work"]
        if len(user_messages) > 1:
            pending_todos.append(f"TODO::{task_id} {user_messages[-1][:160]}")

        recent_critical_results = [
            f"RESULT::{task_id} {result.splitlines()[0][:180]}"
            for result in tool_results[-2:]
        ]
        if not recent_critical_results:
            recent_critical_results = [f"RESULT::{task_id} sampled session has no retained tool result"]

        synthesized_messages = list(messages)
        synthesized_messages.append({"role": "assistant", "content": "\n".join(plan_lines)})
        synthesized_messages.append({"role": "assistant", "content": "\n".join(pending_todos)})
        synthesized_messages.append({
            "role": "tool_result",
            "toolUseId": f"{task_id}_recent_results",
            "toolName": "run_command",
            "content": "\n".join(recent_critical_results),
        })

        scenarios.append({
            "task_id": task_id,
            "name": f"real_session_{session_file.stem}",
            "task_goal": task_goal,
            "hard_constraints": hard_constraints,
            "current_plan": plan_lines,
            "pending_todos": pending_todos,
            "recent_critical_results": recent_critical_results,
            "messages": synthesized_messages,
            "source": "real_session",
            "workspace": workspace,
        })

    return scenarios


def _compute_layer_result(
    layer: str,
    before_tokens: int,
    after_messages: list[dict],
    applied: bool,
) -> dict:
    after_tokens = sum(estimate_message_tokens(m) for m in after_messages)
    saved_pct = (1 - after_tokens / before_tokens) * 100 if before_tokens else 0.0
    return {
        "applied": applied,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "compression_rate": round(saved_pct, 4),
    }


def run_context_compression_benchmark(workspace: str | Path | None = None) -> dict:
    """Run context compression benchmark with retention checks."""
    workspace_path = Path(workspace) if workspace else Path(tempfile.mkdtemp())
    synthetic_scenarios = build_coding_task_scenarios()
    real_session_scenarios = build_real_session_scenarios()
    scenarios = synthetic_scenarios + real_session_scenarios
    scenario_results = []
    constraints_kept_total = 0
    recent_results_kept_total = 0

    for scenario in scenarios:
        messages = scenario["messages"]
        tokens_before = sum(estimate_message_tokens(m) for m in messages)

        precompact_compactor = ContextCompactor(
            context_window=24000,
            workspace=workspace_path,
        )
        precompact_result = precompact_compactor.process_request(
            messages,
            enable_tool_budget=True,
            enable_read_dedup=True,
            enable_microcompact=True,
            enable_auto_compact=False,
        )
        precompact_compactor._microcompact._state.time_based_interval = 0.0
        precompact_compactor._microcompact._state.last_time_based_compact = 0.0
        micro_result = precompact_compactor._microcompact.run_time_based_microcompact(messages, now=time.time())
        tokens_after_precompact = sum(
            estimate_message_tokens(m) for m in precompact_result.messages
        )

        auto_compactor = ContextCompactor(
            context_window=9000,
            workspace=workspace_path,
        )
        auto_result = auto_compactor.process_request(
            messages,
            enable_tool_budget=True,
            enable_read_dedup=True,
            enable_microcompact=False,
            enable_auto_compact=True,
        )

        reactive_compactor = ContextCompactor(
            context_window=9000,
            workspace=workspace_path,
        )
        reactive_result = reactive_compactor.reactive_recover(messages, "prompt too long")

        context_manager = ContextManager(model="default", context_window=12000)
        context_manager.messages = list(precompact_result.messages)
        compacted_messages = context_manager.compact_messages()
        tokens_after_full_compact = sum(
            estimate_message_tokens(m) for m in compacted_messages
        )

        saved_pct_precompact = (
            (1 - tokens_after_precompact / tokens_before) * 100 if tokens_before else 0.0
        )
        saved_pct_full = (
            (1 - tokens_after_full_compact / tokens_before) * 100 if tokens_before else 0.0
        )
        flattened_compact = _flatten_messages(compacted_messages)
        goal_kept = scenario["task_goal"] in flattened_compact
        constraints_kept = all(item in flattened_compact for item in scenario["hard_constraints"])
        plan_kept = all(item in flattened_compact for item in scenario["current_plan"])
        todos_kept = all(item in flattened_compact for item in scenario["pending_todos"])
        recent_results_kept = all(item in flattened_compact for item in scenario["recent_critical_results"])
        constraints_kept_total += int(constraints_kept)
        recent_results_kept_total += int(recent_results_kept)

        scenario_results.append({
            "task_id": scenario["task_id"],
            "name": scenario["name"],
            "source": scenario.get("source", "synthetic"),
            "tokens_before": tokens_before,
            "tokens_after_precompact": tokens_after_precompact,
            "tokens_after_full_compact": tokens_after_full_compact,
            "saved_pct_precompact": round(saved_pct_precompact, 1),
            "saved_pct_full": round(saved_pct_full, 1),
            "used_real_compactor": True,
            "compactor_summary": precompact_result.summary_text,
            "goal_kept": goal_kept,
            "constraints_kept": constraints_kept,
            "plan_kept": plan_kept,
            "todos_kept": todos_kept,
            "recent_results_kept": recent_results_kept,
            "layer_results": {
                "microcompact": _compute_layer_result(
                    "microcompact",
                    tokens_before,
                    micro_result.messages if micro_result else messages,
                    bool(micro_result and micro_result.effective),
                ),
                "auto_compact": _compute_layer_result(
                    "auto_compact",
                    tokens_before,
                    auto_result.messages,
                    bool(auto_result and auto_result.effective),
                ),
                "reactive_compact": _compute_layer_result(
                    "reactive_compact",
                    tokens_before,
                    reactive_result.messages if reactive_result else messages,
                    bool(reactive_result and reactive_result.effective),
                ),
            },
        })

    average_saved = statistics.mean(
        scenario["saved_pct_full"] for scenario in scenario_results
    ) if scenario_results else 0.0
    total = len(scenario_results) or 1

    return {
        "average_saved_pct_multiturn": round(average_saved, 1),
        "constraint_retention_rate": round(constraints_kept_total / total, 4),
        "recent_result_retention_rate": round(recent_results_kept_total / total, 4),
        "synthetic_scenario_count": len(synthetic_scenarios),
        "real_session_scenario_count": len(real_session_scenarios),
        "scenarios": scenario_results,
    }

def _legacy_stress_context_compression_unused() -> dict:
    header("2. Context Compression Ratio")

    results = {}

    # Simulate different conversation lengths
    for num_rounds in [20, 50, 100]:
        # Build simulated conversation with tool results
        messages = [{"role": "system", "content": "You are a coding assistant."}]

        for i in range(num_rounds):
            # User message
            messages.append({
                "role": "user",
                "content": f"Please fix bug #{i} in the authentication module. "
                           f"The issue relates to token validation and session management. "
                           f"Users are reporting {i % 3} different symptoms." + "x" * 200,
            })
            # Assistant response
            messages.append({
                "role": "assistant",
                "content": f"I will investigate bug #{i} by reading the relevant files."
                           f"Let me check the auth module first." + "y" * 100,
            })
            # Tool call
            tool_names = ["read_file", "grep_files", "list_files", "run_command"]
            tool_outputs = [
                f"File content for auth.py showing {i} lines of code.\n" + "def authenticate(): ...\n" * 50,
                f"Found {i * 3} matches for security pattern in src/.\n" + f"  src/auth.py:{i}  def validate_token()\n" * 20,
                f"Directory listing for src/: {i * 5} files\n" + "  auth.py  models.py  views.py  utils.py\n" * 10,
                f"Command output: test suite {i % 5} failures\n" + f"FAILED test_auth_{i}\n" * 30,
            ]
            tool_name = tool_names[i % 4]
            tool_output = tool_outputs[i % 4]

            messages.append({
                "role": "tool_result",
                "toolUseId": f"tool_{i}",
                "toolName": tool_name,
                "content": tool_output,
            })

        # Measure tokens before compression
        tokens_before = sum(estimate_message_tokens(m) for m in messages)

        # Apply compression: Tool Result Budget + Read Dedup
        # 1) Budget large tool results
        budget_mgr = ToolResultBudgetManager(
            workspace=Path(tempfile.mkdtemp()),
            budget_per_message=4000,
            persist_threshold=2000,
        )
        budgeted_msgs, bytes_saved = budget_mgr.check_and_replace(messages)

        # 2) Read dedup
        dedup_mgr = ReadDedupManager()
        deduped_msgs = []
        for msg in budgeted_msgs:
            if msg.get("role") == "tool_result" and msg.get("toolName") == "read_file":
                file_path = f"src/auth_{hash(msg['content']) % 100}.py"
                dedup_mgr.register_read(file_path, msg["content"], len(deduped_msgs))
                if dedup_mgr.should_dedup(file_path, msg["content"]):
                    msg = {**msg, "content": "[File unchanged — refer to earlier read]"}
            deduped_msgs.append(msg)

        tokens_after = sum(estimate_message_tokens(m) for m in deduped_msgs)
        compression_ratio = (1 - tokens_after / tokens_before) * 100 if tokens_before > 0 else 0

        # Simulate reactive compact
        mock_compact_tokens_after = int(tokens_after * 0.55)
        reactive_ratio = (1 - mock_compact_tokens_after / tokens_before) * 100

        print(f"  Rounds={num_rounds:>4}: tokens_before={fmt(tokens_before)}, "
              f"after_budget={fmt(tokens_after)} "
              f"({compression_ratio:.1f}% saved), "
              f"after_summary={fmt(mock_compact_tokens_after)} "
              f"({reactive_ratio:.1f}% total saved)")

        results[num_rounds] = {
            "tokens_before": tokens_before,
            "tokens_after_budget_dedup": tokens_after,
            "budget_dedup_saved_pct": round(compression_ratio, 1),
            "tokens_after_full_compact": mock_compact_tokens_after,
            "total_saved_pct": round(reactive_ratio, 1),
        }

    return results


def stress_context_compression() -> dict:
    """Benchmark context compression using real precompact and full compact paths."""
    header("2. Context Compression Ratio")
    results = run_context_compression_benchmark()
    for scenario in results["scenarios"]:
        print(
            f"  {scenario['name']}: before={fmt(scenario['tokens_before'])}, "
            f"after_precompact={fmt(scenario['tokens_after_precompact'])} "
            f"({scenario['saved_pct_precompact']:.1f}% saved), "
            f"after_full={fmt(scenario['tokens_after_full_compact'])} "
            f"({scenario['saved_pct_full']:.1f}% total saved)"
        )
    print(f"  Average full-compaction saved: {results['average_saved_pct_multiturn']:.1f}%")
    print(f"  Constraint retention rate: {results['constraint_retention_rate']:.2%}")
    print(f"  Recent result retention rate: {results['recent_result_retention_rate']:.2%}")
    return results


# ---------------------------------------------------------------------------
# 3. Agent Loop Throughput (multi-turn stress)
# ---------------------------------------------------------------------------

def stress_agent_loop() -> dict:
    header("3. Agent Runtime Throughput")

    from minicode.tooling import ToolDefinition, ToolRegistry, ToolResult, ToolContext
    from minicode.context_manager import ContextManager
    import concurrent.futures

    results = {}

    # Benchmark: tool execution throughput under concurrency
    for num_tools in [10, 50, 100]:
        for concurrency in [1, 5, 10]:

            # Create tools of varying complexity
            tool_defs = []
            for i in range(num_tools):
                payload_size = 100 if i % 3 == 0 else (1000 if i % 3 == 1 else 5000)
                tool_defs.append(ToolDefinition(
                    name=f"tool_{i}",
                    description=f"Simulated tool {i} with {payload_size}-char output",
                    input_schema={"type": "object"},
                    validator=lambda v: v,
                    run=lambda inp, ctx, size=payload_size: ToolResult(
                        ok=True,
                        output=f"Tool output (size={size}):\n" + "x" * size,
                    ),
                ))
            registry = ToolRegistry(tool_defs)
            ctx = ToolContext(cwd=".")

            # Run tools concurrently
            latencies = []
            errors = 0
            total_chars = 0

            for batch in range(10):
                batch_calls = [
                    (f"tool_{(batch * concurrency + j) % num_tools}", {"arg": j})
                    for j in range(concurrency)
                ]

                start = time.perf_counter()
                if concurrency == 1:
                    for name, inp in batch_calls:
                        try:
                            result = registry.execute(name, inp, ctx)
                            total_chars += len(result.output)
                        except Exception:
                            errors += 1
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = {
                            pool.submit(registry.execute, name, inp, ctx): name
                            for name, inp in batch_calls
                        }
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                                total_chars += len(result.output)
                            except Exception:
                                errors += 1

                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            avg_latency = statistics.mean(latencies) if latencies else 0
            median_latency = statistics.median(latencies) if latencies else 0
            throughput = concurrency * 10 / (sum(latencies) / 1000)  # tools/sec

            print(f"  Tools={num_tools:>3} Concurrency={concurrency:>2}: "
                  f"avg_batch={avg_latency:.1f}ms, med={median_latency:.1f}ms, "
                  f"throughput={throughput:.0f} tools/s, errors={errors}")

            key = f"tools{num_tools}_concurrency{concurrency}"
            results[key] = {
                "avg_batch_latency_ms": round(avg_latency, 1),
                "median_batch_latency_ms": round(median_latency, 1),
                "throughput_tools_per_sec": round(throughput, 0),
                "total_chars_processed": total_chars,
                "errors": errors,
            }

    return results


# ---------------------------------------------------------------------------
# 4. Token Estimation Throughput
# ---------------------------------------------------------------------------

def stress_token_estimation() -> dict:
    header("4. Token Estimation Throughput")

    results = {}

    test_cases = [
        ("short_prompt", "You are a helpful coding assistant." * 5),
        ("medium_code", "def authenticate(token: str) -> bool:\n" * 200),
        ("long_tool_output", "Error: connection refused on port 5432\n" * 500),
        ("mixed_cjk", "用户认证模块 controller layer handler\n" * 300),
    ]

    for name, text in test_cases:
        # Warm up
        for _ in range(100):
            estimate_tokens(text)

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            estimate_tokens(text)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        chars_per_sec = (len(text) * iterations) / elapsed
        tokens = estimate_tokens(text)
        ratio = tokens / len(text) if len(text) > 0 else 0

        print(f"  {name:<20}: {ops_per_sec:>12,.0f} ops/s, "
              f"{fmt(chars_per_sec)} chars/s, "
              f"{len(text)} chars -> {tokens} tokens (ratio={ratio:.2f})")

        results[name] = {
            "ops_per_sec": round(ops_per_sec, 0),
            "chars": len(text),
            "tokens": tokens,
            "token_char_ratio": round(ratio, 3),
        }

    return results


# ---------------------------------------------------------------------------
# 5. Context Manager Overhead
# ---------------------------------------------------------------------------

def stress_context_manager() -> dict:
    header("5. Context Manager Overhead (Long Conversations)")

    results = {}

    for msg_count in [100, 500, 1000, 2000]:
        messages = []
        for i in range(msg_count):
            role = "user" if i % 2 == 0 else "assistant"
            size = "small" if i % 3 == 0 else ("medium" if i % 3 == 1 else "large")
            payloads = {
                "small": "Fix the auth bug in login.py",
                "medium": "Fix bug #123 in auth module. " * 10,
                "large": "Investigate the authentication failure in production. " * 50,
            }
            messages.append({"role": role, "content": payloads[size]})

        # Measure: get_stats
        cm = ContextManager(model="claude-sonnet-4-20250514")

        start = time.perf_counter()
        cm.messages = messages
        stats = cm.get_stats()
        elapsed = (time.perf_counter() - start) * 1000

        # Measure: should_auto_compact
        compact_start = time.perf_counter()
        for _ in range(100):
            cm.should_auto_compact()
        compact_elapsed = (time.perf_counter() - compact_start) * 1000 / 100

        # Measure: compact (simulated)
        compact_full_start = time.perf_counter()
        try:
            compacted = cm.compact_messages()
            compact_full_elapsed = (time.perf_counter() - compact_full_start) * 1000
            tokens_after = sum(estimate_message_tokens(m) for m in compacted)
            saved_pct = (1 - tokens_after / stats.total_tokens) * 100 if stats.total_tokens > 0 else 0
        except Exception:
            compact_full_elapsed = 0
            tokens_after = stats.total_tokens
            saved_pct = 0

        print(f"  Messages={msg_count:>5}: tokens={fmt(stats.total_tokens)}, "
              f"stats={elapsed:.2f}ms, "
              f"should_compact={compact_elapsed:.2f}ms/check, "
              f"compact={compact_full_elapsed:.0f}ms, "
              f"saved={saved_pct:.1f}%")

        results[msg_count] = {
            "total_tokens": stats.total_tokens,
            "usage_pct": round(stats.usage_percentage, 1),
            "get_stats_ms": round(elapsed, 2),
            "should_compact_ms": round(compact_elapsed, 2),
            "compact_ms": round(compact_full_elapsed, 0),
            "compact_saved_pct": round(saved_pct, 1),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results = {}

    all_results["memory_search"] = stress_memory_search()
    all_results["context_compression"] = stress_context_compression()
    all_results["agent_loop"] = stress_agent_loop()
    all_results["token_estimation"] = stress_token_estimation()
    all_results["context_manager"] = stress_context_manager()

    # ---- Summary ----
    header("SUMMARY — Resume-Ready Metrics")
    print()

    ms = all_results["memory_search"]
    print(f"  Memory BM25 Search (2000 entries): "
          f"avg {ms[2000]['avg_latency_ms']}ms, p99 {ms[2000]['p99_latency_ms']}ms")

    cc = all_results["context_compression"]
    print("  Context Compression (multi-turn coding scenarios):")
    print(f"    - Average full compact saved: {cc['average_saved_pct_multiturn']}%")
    print(f"    - Constraint retention: {cc['constraint_retention_rate']:.2%}")
    print(f"    - Recent result retention: {cc['recent_result_retention_rate']:.2%}")
    if cc["scenarios"]:
        first = cc["scenarios"][0]
        print(f"    - Example precompact saved: {first['saved_pct_precompact']}%")
        print(f"    - Example full compact saved: {first['saved_pct_full']}%")

    al = all_results["agent_loop"]
    k = "tools50_concurrency10"
    if k in al:
        print(f"  Agent Runtime (50 tools, concurrency=10): ")
        print(f"    - Batch latency: {al[k]['avg_batch_latency_ms']}ms")
        print(f"    - Throughput: {al[k]['throughput_tools_per_sec']:.0f} tools/s")
        print(f"    - Errors: {al[k]['errors']}")

    te = all_results["token_estimation"]
    print(f"  Token Estimation: {te['medium_code']['ops_per_sec']:,.0f} ops/s "
          f"(~{te['medium_code']['token_char_ratio']:.2f} tokens/char)")

    cm = all_results["context_manager"]
    c2k = cm[2000]
    print(f"  Context Manager (2000 messages): ")
    print(f"    - Stats check: {c2k['get_stats_ms']}ms")
    print(f"    - Compact saved: {c2k['compact_saved_pct']}% tokens")

    # Save to file
    output_path = Path(__file__).parent / "resume_metrics_results.json"
    # Convert sets to lists for JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
