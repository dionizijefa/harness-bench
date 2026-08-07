#!/usr/bin/env python3
"""Plot token spend, outcomes, budgets, and version trends from results.sqlite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import tomllib
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch


SUCCESS = "#009E73"
UNSUCCESSFUL = "#D55E00"
MISSING = "#D0D4D8"
INK = "#202124"
MUTED = "#6B7280"
GRID = "#D8DADD"
ACCENT = "#0072B2"


@dataclass(frozen=True)
class Result:
    task_id: str
    version: str
    status: str
    passed: bool
    total_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    model_call_count: int | None
    cost_usd: float | None
    output_dir: str
    max_output_tokens_per_call: int | None = None
    context_window: int | None = None


def version_key(version: str) -> tuple[tuple[int, Any], ...]:
    """Sort ordinary semantic versions without adding a packaging dependency."""
    pieces: list[tuple[int, Any]] = []
    for piece in version.replace("-", ".").split("."):
        pieces.append((0, int(piece)) if piece.isdigit() else (1, piece.lower()))
    return tuple(pieces)


def finite(values: Iterable[int | float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def median(values: Iterable[int | float | None]) -> float | None:
    usable = finite(values)
    return statistics.median(usable) if usable else None


def mean(values: Iterable[int | float | None]) -> float | None:
    usable = finite(values)
    return statistics.fmean(usable) if usable else None


def sample_std(values: Iterable[int | float | None]) -> float | None:
    usable = finite(values)
    return statistics.stdev(usable) if len(usable) >= 2 else None


def linear_fit(values: list[float | None]) -> dict[str, float | None]:
    points = [(index, value) for index, value in enumerate(values) if value is not None]
    if len(points) < 2:
        return {"slope": None, "intercept": None, "r_squared": None, "pct_per_step": None}
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total == 0 and residual == 0 else (1 - residual / total if total else 0.0)
    average = float(np.mean(y))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "pct_per_step": float(100 * slope / average) if average else None,
    }


def load_results(
    database: Path,
    harness: str,
    model: str | None,
    dataset: str | None,
) -> tuple[list[Result], str, str]:
    if not database.is_file():
        raise SystemExit(f"results database does not exist: {database}")
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    where = ["harness = ?"]
    parameters: list[str] = [harness]
    if model:
        where.append("model = ?")
        parameters.append(model)
    if dataset:
        where.append("dataset = ?")
        parameters.append(dataset)
    sql = f"""
        SELECT task_id, harness_version, status, passed, total_tokens,
               input_tokens, output_tokens, model_call_count, cost_usd, output_dir,
               model, dataset
        FROM rollout_results
        WHERE {' AND '.join(where)}
        ORDER BY task_id, harness_version, rollout, timestamp
    """
    rows = connection.execute(sql, parameters).fetchall()
    connection.close()
    if not rows:
        raise SystemExit("no rows matched the requested harness/model/dataset")

    models = sorted({row["model"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    if len(models) > 1:
        raise SystemExit("multiple models matched; select one with --model: " + ", ".join(models))
    if len(datasets) > 1:
        raise SystemExit(
            "multiple datasets matched; select one with --dataset: " + ", ".join(datasets)
        )
    results = [
        Result(
            task_id=row["task_id"],
            version=row["harness_version"],
            status=row["status"],
            passed=bool(row["passed"]),
            total_tokens=row["total_tokens"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            model_call_count=row["model_call_count"],
            cost_usd=row["cost_usd"],
            output_dir=row["output_dir"],
        )
        for row in rows
    ]
    return results, models[0], datasets[0]


def add_configured_budgets(results: list[Result], database: Path) -> list[Result]:
    repo_root = database.resolve().parent.parent
    cache: dict[Path, tuple[int | None, int | None]] = {}
    enriched: list[Result] = []
    for result in results:
        output_path = Path(result.output_dir)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        config_path = output_path / "config.toml"
        if config_path not in cache:
            max_tokens = context_window = None
            if config_path.is_file():
                try:
                    harness = tomllib.loads(config_path.read_text(encoding="utf-8")).get(
                        "harness", {}
                    )
                    max_tokens = harness.get("max_tokens")
                    context_window = harness.get("context_window")
                except (OSError, tomllib.TOMLDecodeError):
                    pass
            cache[config_path] = (max_tokens, context_window)
        max_tokens, context_window = cache[config_path]
        enriched.append(
            replace(
                result,
                max_output_tokens_per_call=max_tokens,
                context_window=context_window,
            )
        )
    return enriched


def group_results(results: list[Result]) -> dict[tuple[str, str], list[Result]]:
    grouped: dict[tuple[str, str], list[Result]] = defaultdict(list)
    for result in results:
        grouped[(result.task_id, result.version)].append(result)
    return grouped


def all_version_successes(
    results: list[Result], versions: list[str]
) -> tuple[list[str], list[str]]:
    by_task: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result)
    complete: list[str] = []
    successful: list[str] = []
    expected = set(versions)
    for task, task_results in by_task.items():
        if {result.version for result in task_results} != expected:
            continue
        complete.append(task)
        if all(result.status == "passed" and result.passed for result in task_results):
            successful.append(task)
    return sorted(complete), sorted(successful)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "axes.grid": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_outcome_matrix(
    results: list[Result], versions: list[str], successful: list[str], path: Path
) -> None:
    by_task: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result)
    successful_set = set(successful)

    def task_sort(task: str) -> tuple[int, int, str]:
        passed_versions = sum(
            any(row.status == "passed" and row.passed for row in by_task[task] if row.version == v)
            for v in versions
        )
        return (0 if task in successful_set else 1, -passed_versions, task)

    tasks = sorted(by_task, key=task_sort)
    matrix = np.zeros((len(tasks), len(versions)), dtype=int)
    for y, task in enumerate(tasks):
        for x, version in enumerate(versions):
            rows = [row for row in by_task[task] if row.version == version]
            if not rows:
                state = 0
            elif all(row.status == "passed" and row.passed for row in rows):
                state = 2
            else:
                state = 1
            matrix[y, x] = state

    fig, ax = plt.subplots(figsize=(10.5, max(12, 0.245 * len(tasks) + 2.5)))
    cmap = ListedColormap([MISSING, UNSUCCESSFUL, SUCCESS])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto")
    for y in range(len(tasks)):
        for x in range(len(versions)):
            symbol = {0: "·", 1: "×", 2: "✓"}[int(matrix[y, x])]
            ax.text(x, y, symbol, ha="center", va="center", fontsize=7, color="white" if matrix[y, x] else MUTED)
    ax.set_xticks(range(len(versions)), [f"v{version}" for version in versions])
    ax.set_yticks(range(len(tasks)), tasks, fontsize=6.8)
    for label, task in zip(ax.get_yticklabels(), tasks, strict=True):
        if task in successful_set:
            label.set_fontweight("bold")
            label.set_color(SUCCESS)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("OpenCode version (oldest → newest)")
    ax.set_title(
        f"Task outcomes across OpenCode versions\n{len(successful)} tasks passed in every one of {len(versions)} versions",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=48,
    )
    ax.legend(
        handles=[
            Patch(facecolor=SUCCESS, label="Successful"),
            Patch(facecolor=UNSUCCESSFUL, label="Unsuccessful / error"),
            Patch(facecolor=MISSING, label="Missing"),
        ],
        ncol=3,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.012),
        fontsize=8,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, path)


def aggregate_metric(
    grouped: dict[tuple[str, str], list[Result]], task: str, version: str, field: str
) -> float | None:
    return median(getattr(result, field) for result in grouped.get((task, version), []))


def plot_successful_spend_trends(
    grouped: dict[tuple[str, str], list[Result]],
    versions: list[str],
    tasks: list[str],
    trends: dict[str, dict[str, float | None]],
    path: Path,
) -> None:
    columns = 3
    rows = math.ceil(len(tasks) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.55), squeeze=False)
    x = np.arange(len(versions), dtype=float)
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            -(trends[task]["r_squared"] or 0),
            task,
        ),
    )
    for ax, task in zip(axes.flat, ordered_tasks, strict=False):
        values = [aggregate_metric(grouped, task, version, "total_tokens") for version in versions]
        y = np.asarray([value if value is not None else np.nan for value in values], dtype=float)
        ax.plot(x, y, color=SUCCESS, marker="o", linewidth=2, markersize=4, label="Observed")
        trend = trends[task]
        if trend["slope"] is not None and trend["intercept"] is not None:
            fitted = float(trend["slope"]) * x + float(trend["intercept"])
            ax.plot(x, fitted, color=INK, linestyle="--", linewidth=1, label="Linear fit")
        pct = trend["pct_per_step"]
        r_squared = trend["r_squared"]
        subtitle = (
            f"{pct:+.1f}% / version step · R² {r_squared:.2f}"
            if pct is not None and r_squared is not None
            else "insufficient token data"
        )
        ax.set_title(f"{task}\n{subtitle}", loc="left", fontsize=8.2, fontweight="bold")
        ax.set_xticks(x, versions, rotation=35, ha="right", fontsize=6.8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(6)
    for ax in axes.flat[len(ordered_tasks) :]:
        ax.remove()
    fig.suptitle(
        "Total-token spend for tasks successful in every version\nObserved spend and linear fit over historical version order",
        x=0.06,
        y=1.002,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("OpenCode version (oldest → newest)", fontsize=9)
    fig.supylabel("Total tokens per task rollout", fontsize=9)
    fig.tight_layout()
    save_figure(fig, path)


def outcome_name(result: Result) -> str:
    return "Successful" if result.status == "passed" and result.passed else "Unsuccessful"


def plot_version_variance(
    results: list[Result], versions: list[str], common_tasks: list[str], path: Path
) -> None:
    outcomes = ["Successful", "Unsuccessful"]
    colors = [SUCCESS, UNSUCCESSFUL]
    data: dict[tuple[str, str], list[float]] = defaultdict(list)
    common_task_set = set(common_tasks)
    for result in results:
        if result.task_id not in common_task_set:
            continue
        if result.total_tokens is not None and result.total_tokens > 0:
            data[(result.version, outcome_name(result))].append(float(result.total_tokens))

    fig, (ax_box, ax_cv) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True
    )
    base = np.arange(len(versions), dtype=float)
    offsets = [-0.19, 0.19]
    for outcome, color, offset in zip(outcomes, colors, offsets, strict=True):
        values = [data[(version, outcome)] for version in versions]
        positions = base + offset
        box = ax_box.boxplot(
            values,
            positions=positions,
            widths=0.31,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.45, "markeredgecolor": color},
            medianprops={"color": INK, "linewidth": 1.3},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
            patch.set_edgecolor(color)
        cvs = []
        for values_for_version in values:
            avg = mean(values_for_version)
            std = sample_std(values_for_version)
            cvs.append(std / avg if avg and std is not None else np.nan)
        ax_cv.bar(positions, cvs, width=0.31, color=color, alpha=0.76, label=outcome)

    ax_box.set_yscale("log")
    ax_box.set_ylabel("Total tokens per rollout (log scale)")
    ax_box.set_title(
        f"Token-spend distributions and variance by OpenCode version\nCommon-task cohort (n={len(common_tasks)})",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax_box.grid(axis="y", color=GRID, linewidth=0.7, which="both")
    ax_box.legend(handles=[Patch(facecolor=c, label=o) for o, c in zip(outcomes, colors, strict=True)], frameon=False, ncol=2)
    ax_cv.set_ylabel("Coefficient of variation\n(sample SD ÷ mean)")
    ax_cv.set_xlabel("OpenCode version (oldest → newest)")
    ax_cv.set_xticks(base, [f"v{version}" for version in versions])
    ax_cv.grid(axis="y", color=GRID, linewidth=0.7)
    ax_cv.legend(frameon=False, ncol=2)
    for ax in (ax_box, ax_cv):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)


def make_matrix(
    grouped: dict[tuple[str, str], list[Result]],
    tasks: list[str],
    versions: list[str],
    value,
) -> np.ndarray:
    matrix = np.full((len(tasks), len(versions)), np.nan)
    for y, task in enumerate(tasks):
        for x, version in enumerate(versions):
            values = finite(value(result) for result in grouped.get((task, version), []))
            if values:
                matrix[y, x] = statistics.median(values)
    return matrix


def plot_cost_and_budget(
    grouped: dict[tuple[str, str], list[Result]], versions: list[str], tasks: list[str], path: Path
) -> None:
    cost = make_matrix(grouped, tasks, versions, lambda result: result.cost_usd)

    def utilization(result: Result) -> float | None:
        if not result.output_tokens or not result.model_call_count or not result.max_output_tokens_per_call:
            return None
        return 100 * result.output_tokens / (
            result.model_call_count * result.max_output_tokens_per_call
        )

    budget = make_matrix(grouped, tasks, versions, utilization)
    order = np.argsort(-np.nanmedian(cost, axis=1))
    ordered_tasks = [tasks[index] for index in order]
    cost = cost[order]
    budget = budget[order]

    fig, (ax_cost, ax_budget) = plt.subplots(
        1, 2, figsize=(16, max(10, len(tasks) * 0.37 + 2.8)), sharey=True
    )
    green_cmap = LinearSegmentedColormap.from_list("success_cost", ["#F1F8F6", SUCCESS])
    blue_cmap = LinearSegmentedColormap.from_list("budget_use", ["#EEF6FA", ACCENT])
    cost_image = ax_cost.imshow(cost, aspect="auto", cmap=green_cmap)
    budget_image = ax_budget.imshow(budget, aspect="auto", cmap=blue_cmap, vmin=0)
    for ax, matrix, formatter in (
        (ax_cost, cost, lambda value: f"${value:.3f}"),
        (ax_budget, budget, lambda value: f"{value:.1f}%"),
    ):
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                if math.isnan(value):
                    label = "—"
                else:
                    label = formatter(value)
                ax.text(x, y, label, ha="center", va="center", fontsize=6.2, color=INK)
        ax.set_xticks(range(len(versions)), [f"v{version}" for version in versions], rotation=35, ha="right")
        ax.tick_params(axis="both", length=0, labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_cost.set_yticks(range(len(ordered_tasks)), ordered_tasks, fontsize=7)
    ax_cost.set_title("Provider-reported cost per task", loc="left", fontsize=11, fontweight="bold")
    ax_budget.set_title(
        "Output-token budget used\noutput tokens ÷ (calls × configured max tokens)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.colorbar(cost_image, ax=ax_cost, fraction=0.028, pad=0.02, label="USD")
    fig.colorbar(budget_image, ax=ax_budget, fraction=0.028, pad=0.02, label="Percent")
    fig.suptitle(
        "Cost and token-budget use for all-version-successful tasks",
        x=0.04,
        y=1.002,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_mean_tokens_and_cost(
    grouped: dict[tuple[str, str], list[Result]],
    versions: list[str],
    common_tasks: list[str],
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | None]]]:
    paired_tasks = [
        task
        for task in common_tasks
        if all(
            aggregate_metric(grouped, task, version, "total_tokens") is not None
            and aggregate_metric(grouped, task, version, "cost_usd") is not None
            for version in versions
        )
    ]
    rows: list[dict[str, Any]] = []
    for index, version in enumerate(versions):
        token_values = [
            aggregate_metric(grouped, task, version, "total_tokens")
            for task in paired_tasks
        ]
        cost_values = [
            aggregate_metric(grouped, task, version, "cost_usd")
            for task in paired_tasks
        ]
        rows.append(
            {
                "harness_version": version,
                "version_index": index,
                "tasks": len(paired_tasks),
                "mean_total_tokens_per_task": mean(token_values),
                "mean_cost_usd_per_task": mean(cost_values),
            }
        )

    tokens = [float(row["mean_total_tokens_per_task"]) for row in rows]
    costs = [float(row["mean_cost_usd_per_task"]) for row in rows]
    fits = {"tokens": linear_fit(tokens), "cost_usd": linear_fit(costs)}
    x = np.arange(len(versions), dtype=float)

    fig, tokens_axis = plt.subplots(figsize=(11.5, 6.5))
    cost_axis = tokens_axis.twinx()
    tokens_axis.plot(
        x,
        tokens,
        color=SUCCESS,
        marker="o",
        markersize=7,
        linewidth=2.6,
        label="Mean total tokens / task",
    )
    cost_axis.plot(
        x,
        costs,
        color=ACCENT,
        marker="D",
        markersize=6,
        linewidth=2.6,
        label="Mean cost / task",
    )
    for position, token_value, cost_value in zip(x, tokens, costs, strict=True):
        tokens_axis.annotate(
            f"{token_value / 1000:.0f}k",
            (position, token_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=SUCCESS,
            fontsize=8,
            fontweight="bold",
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
        )
        cost_axis.annotate(
            f"${cost_value:.3f}",
            (position, cost_value),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            color=ACCENT,
            fontsize=8,
            fontweight="bold",
            zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
        )

    token_fit = fits["tokens"]
    cost_fit = fits["cost_usd"]
    tokens_axis.set_title(
        "Mean token spend and cost per task by OpenCode version\n"
        f"Complete token-and-cost cohort (n={len(paired_tasks)} tasks); "
        f"linear trend R²: tokens {token_fit['r_squared']:.2f}, cost {cost_fit['r_squared']:.2f}",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    tokens_axis.set_xlabel("OpenCode version (oldest → newest)")
    tokens_axis.set_ylabel("Mean total tokens per task", color=SUCCESS)
    cost_axis.set_ylabel("Mean provider-reported cost per task (USD)", color=ACCENT)
    tokens_axis.set_xticks(x, [f"v{version}" for version in versions])
    tokens_axis.tick_params(axis="y", colors=SUCCESS)
    cost_axis.tick_params(axis="y", colors=ACCENT)
    tokens_axis.grid(axis="y", color=GRID, linewidth=0.7)
    tokens_axis.spines[["top"]].set_visible(False)
    cost_axis.spines[["top"]].set_visible(False)
    lines = tokens_axis.get_lines() + cost_axis.get_lines()
    tokens_axis.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        ncol=2,
        loc="upper left",
    )
    fig.tight_layout()
    save_figure(fig, path)
    return rows, fits


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(
    results: list[Result],
    grouped: dict[tuple[str, str], list[Result]],
    versions: list[str],
    common_tasks: list[str],
    successful: list[str],
    trends: dict[str, dict[str, float | None]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    for task in successful:
        for version_index, version in enumerate(versions):
            rows = grouped[(task, version)]
            max_tokens = median(row.max_output_tokens_per_call for row in rows)
            calls = median(row.model_call_count for row in rows)
            output_tokens = median(row.output_tokens for row in rows)
            cumulative_budget = max_tokens * calls if max_tokens is not None and calls is not None else None
            detail_rows.append(
                {
                    "task_id": task,
                    "harness_version": version,
                    "version_index": version_index,
                    "rollouts": len(rows),
                    "median_total_tokens": median(row.total_tokens for row in rows),
                    "median_input_tokens": median(row.input_tokens for row in rows),
                    "median_output_tokens": output_tokens,
                    "median_model_calls": calls,
                    "median_cost_usd": median(row.cost_usd for row in rows),
                    "max_output_tokens_per_call": max_tokens,
                    "cumulative_output_budget": cumulative_budget,
                    "output_budget_utilization": (
                        output_tokens / cumulative_budget
                        if output_tokens is not None and cumulative_budget
                        else None
                    ),
                    "context_window": median(row.context_window for row in rows),
                    "token_slope_per_version_step": trends[task]["slope"],
                    "token_slope_pct_of_mean": trends[task]["pct_per_step"],
                    "token_trend_r_squared": trends[task]["r_squared"],
                }
            )
    detail_fields = list(detail_rows[0]) if detail_rows else []
    if detail_fields:
        write_csv(output_dir / "all_version_successful_spend.csv", detail_rows, detail_fields)

    variance_rows: list[dict[str, Any]] = []
    common_task_set = set(common_tasks)
    for version in versions:
        version_results = [
            result
            for result in results
            if result.version == version and result.task_id in common_task_set
        ]
        for outcome in ("Successful", "Unsuccessful", "All"):
            subset = (
                version_results
                if outcome == "All"
                else [result for result in version_results if outcome_name(result) == outcome]
            )
            tokens = finite(result.total_tokens for result in subset if result.total_tokens and result.total_tokens > 0)
            avg = mean(tokens)
            std = sample_std(tokens)
            variance_rows.append(
                {
                    "harness_version": version,
                    "outcome": outcome,
                    "rows": len(subset),
                    "rows_with_positive_tokens": len(tokens),
                    "mean_total_tokens": avg,
                    "median_total_tokens": median(tokens),
                    "sample_std_total_tokens": std,
                    "sample_variance_total_tokens": std * std if std is not None else None,
                    "coefficient_of_variation": std / avg if std is not None and avg else None,
                }
            )
    write_csv(output_dir / "version_variance.csv", variance_rows, list(variance_rows[0]))
    return detail_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("outputs/results.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots/opencode"))
    parser.add_argument("--harness", default="opencode")
    parser.add_argument("--model", help="required when the selected harness has multiple models")
    parser.add_argument("--dataset", help="required when the selected harness has multiple datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results, model, dataset = load_results(args.db, args.harness, args.model, args.dataset)
    results = add_configured_budgets(results, args.db)
    versions = sorted({result.version for result in results}, key=version_key)
    complete, successful = all_version_successes(results, versions)
    grouped = group_results(results)
    trends = {
        task: linear_fit(
            [aggregate_metric(grouped, task, version, "total_tokens") for version in versions]
        )
        for task in successful
    }

    apply_plot_style()
    plot_outcome_matrix(results, versions, successful, output_dir / "task_outcomes.png")
    plot_successful_spend_trends(
        grouped, versions, successful, trends, output_dir / "successful_task_token_trends.png"
    )
    plot_version_variance(
        results, versions, complete, output_dir / "version_token_variance.png"
    )
    plot_cost_and_budget(
        grouped, versions, successful, output_dir / "successful_task_cost_and_budget.png"
    )
    mean_rows, mean_fits = plot_mean_tokens_and_cost(
        grouped,
        versions,
        complete,
        output_dir / "mean_tokens_and_cost_by_version.png",
    )
    write_csv(
        output_dir / "mean_tokens_and_cost_by_version.csv",
        mean_rows,
        list(mean_rows[0]),
    )
    detail_rows = export_tables(
        results, grouped, versions, complete, successful, trends, output_dir
    )

    version_medians = [
        median(
            aggregate_metric(grouped, task, version, "total_tokens")
            for task in successful
        )
        for version in versions
    ]
    aggregate_trend = linear_fit(version_medians)
    summary = {
        "database": str(args.db.resolve()),
        "harness": args.harness,
        "model": model,
        "dataset": dataset,
        "rows": len(results),
        "versions": versions,
        "tasks": len({result.task_id for result in results}),
        "tasks_present_in_all_versions": len(complete),
        "tasks_successful_in_all_versions": len(successful),
        "all_version_successful_tasks": successful,
        "median_token_trend": {
            "values_by_version": dict(zip(versions, version_medians, strict=True)),
            **aggregate_trend,
        },
        "mean_tokens_and_cost_complete_cohort": {
            "tasks": mean_rows[0]["tasks"],
            "values_by_version": mean_rows,
            "linear_fits": mean_fits,
        },
        "configured_max_output_tokens_per_call": sorted(
            {result.max_output_tokens_per_call for result in results if result.max_output_tokens_per_call}
        ),
        "context_windows": sorted(
            {result.context_window for result in results if result.context_window}
        ),
        "detail_rows": len(detail_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Selected {len(results)} rows for {args.harness} / {model}")
    print(f"Versions ({len(versions)}): {', '.join(versions)}")
    print(f"Tasks present in every version: {len(complete)}")
    print(f"Tasks successful in every version: {len(successful)}")
    if aggregate_trend["pct_per_step"] is not None:
        print(
            "Median successful-task token trend: "
            f"{aggregate_trend['pct_per_step']:+.1f}% per version step "
            f"(R²={aggregate_trend['r_squared']:.2f})"
        )
    print(f"Wrote plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
