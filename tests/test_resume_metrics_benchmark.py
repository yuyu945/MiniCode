"""Tests for resume metrics benchmark outputs."""

from __future__ import annotations

from benchmarks.resume_metrics_stress_test import (
    build_coding_task_scenarios,
    run_context_compression_benchmark,
)


def test_build_coding_task_scenarios_returns_multiple_multiturn_cases():
    scenarios = build_coding_task_scenarios()

    assert len(scenarios) >= 10
    assert all("name" in scenario for scenario in scenarios)
    assert all("messages" in scenario for scenario in scenarios)
    assert all("task_goal" in scenario for scenario in scenarios)
    assert all("hard_constraints" in scenario for scenario in scenarios)
    assert all("current_plan" in scenario for scenario in scenarios)
    assert all("pending_todos" in scenario for scenario in scenarios)
    assert all("recent_critical_results" in scenario for scenario in scenarios)
    assert all(
        any(message.get("role") == "tool_result" for message in scenario["messages"])
        for scenario in scenarios
    )


def test_run_context_compression_benchmark_reports_real_compactor_metrics(tmp_path):
    result = run_context_compression_benchmark(workspace=tmp_path)

    assert "average_saved_pct_multiturn" in result
    assert "constraint_retention_rate" in result
    assert "recent_result_retention_rate" in result
    assert "scenarios" in result
    assert len(result["scenarios"]) >= 10

    for scenario_result in result["scenarios"]:
        assert {
            "name",
            "task_id",
            "tokens_before",
            "tokens_after_precompact",
            "tokens_after_full_compact",
            "saved_pct_precompact",
            "saved_pct_full",
            "used_real_compactor",
            "goal_kept",
            "constraints_kept",
            "plan_kept",
            "todos_kept",
            "recent_results_kept",
            "layer_results",
        } <= set(scenario_result)
        assert scenario_result["used_real_compactor"] is True
        assert scenario_result["tokens_after_full_compact"] <= scenario_result["tokens_after_precompact"]
        assert {"microcompact", "auto_compact", "reactive_compact"} <= set(scenario_result["layer_results"])
