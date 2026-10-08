#!/usr/bin/env python3
"""SVD and task-subset reconstruction of the retained historical bench cohorts.

Uses the committed aggregate input; does not launch benchmark or model runs.
Reproduce with:
    uv run --group dev python scripts/analyze_task_svd.py
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "analysis/task-selection-input.json"
SEED = 20261008
RCOND = 1e-10
HARNESS_LABELS = {
    "claude_code_agent": "Claude Code", "codex_agent": "Codex",
    "deepseek_harness": "DeepSeek Harness", "hermes_agent": "Hermes",
    "omp_agent": "OMP", "opencode": "OpenCode", "pi_agent": "Pi Agent",
    "pi_rlm_runtime": "Pi RLM", "prime_agent": "PrimeAgent",
}


def load_aggregate(matrix_path: Path):
    """Reproduce selection without the private rollout database or export code."""
    contents = matrix_path.read_bytes()
    data = json.loads(contents)
    assert data["schema_version"] == 1
    tasks, harnesses, models = data["tasks"], data["harnesses"], data["models"]
    assert tasks == sorted(set(tasks)) and len(tasks) == 78
    assert len(set(harnesses)) == len(harnesses) == 9
    assert models == ["deepseek", "glm", "muse"]
    matrix = np.asarray(data["rates"], dtype=float)
    cube = np.asarray(data["per_model_rates"], dtype=float)
    assert matrix.shape == (78, 9) and cube.shape == (3, 78, 9)
    assert np.isfinite(cube).all() and ((cube >= 0) & (cube <= 1)).all()
    assert np.allclose(matrix, cube.mean(axis=0), rtol=0, atol=1e-12)
    audit = dict(data["provenance"])
    assert hashlib.sha256("\n".join(tasks).encode()).hexdigest() == audit["task_set_sha256"]
    counts = cube * np.array([audit["repetitions_per_model"][model] for model in models])[:, None, None]
    assert np.allclose(counts, np.rint(counts), rtol=0, atol=1e-12)
    assert int(np.rint(counts).sum()) == audit["status_counts"]["passed"]
    assert 78 * 9 * sum(audit["repetitions_per_model"].values()) == audit["trial_count"] == 3510
    assert sum(audit["status_counts"].values()) == audit["trial_count"]
    audit.update(
        aggregate_input=matrix_path.name,
        aggregate_input_sha256=hashlib.sha256(contents).hexdigest(),
        verification="Aggregate shape, task-set fingerprint, repetition counts, probability bounds and equal model weighting rechecked. The original trial-level audit is preserved separately; raw rollout records are not bundled.",
    )
    return matrix, cube, tasks, harnesses, models, audit


def load_matrix(snapshot_path: Path, reference_path: Path):
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    reference = json.loads(reference_path.read_bytes())
    tasks = sorted({row["task_id"] for row in snapshot["rows"]})
    schema = json.loads(DEFAULT_MATRIX.read_bytes())
    harnesses, model_keys = schema["harnesses"], schema["models"]
    groups = defaultdict(list)
    trial_keys = set()
    versions = defaultdict(set)
    expected = {}
    for source in schema["provenance"]["sources"]:
        for harness in source["harnesses"]:
            expected[source["model_key"], harness] = (source["column"], source["source_id"], source["model_id"], source["repetitions"])
    for row in snapshot["rows"]:
        key = (row["model_key"], row["harness"])
        assert key in expected, f"Unexpected cohort: {key}"
        column, source_id, model_id, _ = expected[key]
        assert row[column] == source_id and row["model"] == model_id
        assert row["dataset"] == "terminal-bench/terminal-bench-2-1"
        assert row["status"] in {"passed", "failed", "error"}
        trial_key = (*key, row["task_id"], row["rollout"])
        assert trial_key not in trial_keys, f"Duplicate trial: {trial_key}"
        trial_keys.add(trial_key)
        groups[*key, row["task_id"]].append(float(row["status"] == "passed"))
        versions[key].add(row["harness_version"])
    assert len(tasks) == 78 and len(snapshot["rows"]) == 3510
    for key, (_, _, _, repetitions) in expected.items():
        assert len(versions[key]) == 1
        for task in tasks:
            assert len(groups[*key, task]) == repetitions, (key, task)
    cube = np.array([
        [[np.mean(groups[model, harness, task]) for harness in harnesses] for task in tasks]
        for model in model_keys
    ])
    for dataset in reference["datasets"]:
        model_index = model_keys.index(dataset["id"])
        for harness in dataset["harnesses"]:
            index = harnesses.index(harness["id"])
            assert np.isclose(
                cube[model_index, :, index].mean(), harness["mean_success_rate_per_task"],
                rtol=0, atol=1e-12,
            ), f"Public aggregate mismatch: {dataset['id']}/{harness['id']}"
    assert hashlib.sha256("\n".join(tasks).encode()).hexdigest() == reference["task_set_sha256"]
    audit = {
        "snapshot": str(snapshot_path.resolve()),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "retrieved_at": snapshot["retrieved_at"],
        "source_database": snapshot["source_database"],
        "sources": snapshot["sources"],
        "reference": str(reference_path.resolve()),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "task_set_sha256": reference["task_set_sha256"],
        "trial_count": len(snapshot["rows"]),
        "status_counts": dict(Counter(row["status"] for row in snapshot["rows"])),
        "first_result": min(row["timestamp"] for row in snapshot["rows"]),
        "last_result": max(row["timestamp"] for row in snapshot["rows"]),
        "harness_versions": {
            f"{model}/{harness}": next(iter(value)) for (model, harness), value in versions.items()
        },
        "verification": "All 27 per-model/harness means match the retained website snapshot within 1e-12. "
                        "Exact sources/models, shared 78-task set, repetitions, unique trials and terminal statuses verified.",
    }
    return cube.mean(axis=0), cube, tasks, harnesses, model_keys, audit


def spectrum(matrix):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    assert np.allclose((u * s) @ vh, matrix, rtol=1e-10, atol=1e-10)
    assert np.isclose(np.sum(s**2), np.sum(matrix**2), rtol=1e-12)
    cumulative = np.cumsum(s**2) / np.sum(s**2)
    return {
        "singular_values": s.tolist(),
        "explained_fraction": (s**2 / np.sum(s**2)).tolist(),
        "cumulative_explained_fraction": cumulative.tolist(),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "components_for_90_percent": int(np.searchsorted(cumulative, 0.9) + 1),
        "components_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        "task_loadings": u.tolist(),
        "harness_loadings": vh.T.tolist(),
    }


def task_order(contrasts):
    """Greedy row interpolation, followed by A-optimal sampling of the SVD basis.

    Each initial pivot gives the largest one-step decrease in Frobenius error
    after projecting onto the selected row span. After spanning that space,
    additional rows minimize the trace of the inverse SVD information matrix.
    This provides one nested ordering, not a proof of the globally best subset.
    """
    u, _, _ = np.linalg.svd(contrasts, full_matrices=False)
    rank = int(np.linalg.matrix_rank(contrasts))
    basis = u[:, :rank]
    residual = contrasts.copy()
    selected = []
    for _ in range(rank):
        energy = np.sum(residual**2, axis=1)
        gram = residual @ residual.T
        scores = np.sum(gram**2, axis=0) / np.where(energy > 1e-20, energy, np.inf)
        scores[selected] = -np.inf
        pivot = int(np.argmax(scores))
        selected.append(pivot)
        direction = residual[pivot] / np.linalg.norm(residual[pivot])
        residual -= np.outer(residual @ direction, direction)
    assert np.linalg.norm(residual) < 1e-8
    for _ in range(rank, len(contrasts)):
        inverse = np.linalg.inv(basis[selected].T @ basis[selected])
        projected = basis @ inverse
        scores = np.sum(projected**2, axis=1) / (1 + np.sum(projected * basis, axis=1))
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    assert len(set(selected)) == len(contrasts)
    return selected


def predict_fold(train, observed, selected):
    baseline = train.mean(axis=1)
    contrasts = train - baseline[:, None]
    prediction = baseline.copy()
    if len(selected):
        coefficients = np.linalg.lstsq(
            contrasts[selected], observed[selected] - baseline[selected], rcond=RCOND,
        )[0]
        prediction += contrasts @ coefficients
    # Completion probabilities must be within [0, 1]. Clipping is fixed in advance,
    # and used for all held-out predictions including the random control.
    prediction = np.clip(prediction, 0, 1)
    prediction[selected] = observed[selected]
    return prediction


def cross_validate(matrix, orders):
    predictions = np.zeros((len(matrix) + 1, *matrix.shape))
    unmeasured = np.ones_like(predictions, dtype=bool)
    folds = []
    for harness, order in enumerate(orders):
        train = np.delete(matrix, harness, axis=1)
        assert train.shape[1] == matrix.shape[1] - 1
        for budget in range(len(matrix) + 1):
            selected = order[:budget]
            predictions[budget, :, harness] = predict_fold(train, matrix[:, harness], selected)
            unmeasured[budget, selected, harness] = False
        folds.append({"held_out_harness_index": harness, "task_order_indices": order})
    errors = (predictions - matrix[None, :, :])**2
    centered = matrix - matrix.mean(axis=0)
    denominator = np.sum(centered**2)
    unmeasured_denominator = np.sum(centered[None, :, :]**2 * unmeasured, axis=(1, 2))
    unseen_r2 = np.full(len(matrix) + 1, np.nan)
    np.divide(np.sum(errors * unmeasured, axis=(1, 2)), unmeasured_denominator,
              out=unseen_r2, where=unmeasured_denominator > 1e-14)
    unseen_r2 = 1 - unseen_r2
    unmeasured_count = unmeasured.sum(axis=(1, 2))
    unseen_rmse = np.full(len(matrix) + 1, np.nan)
    np.divide(np.sum(errors * unmeasured, axis=(1, 2)), unmeasured_count,
              out=unseen_rmse, where=unmeasured_count > 0)
    return {
        "r2": 1 - errors.sum(axis=(1, 2)) / denominator,
        "unmeasured_r2": unseen_r2,
        "rmse": np.sqrt(errors.mean(axis=(1, 2))),
        "unmeasured_rmse": np.sqrt(unseen_rmse),
        "predictions": predictions,
        "folds": folds,
    }


def first_threshold(values, target):
    return next((index for index, value in enumerate(values) if value >= target - 1e-12), None)


def make_plots(output, matrix, tasks, harnesses, order, centered_svd, interactions_svd,
               historical, cv, random_summary):
    cache = Path(tempfile.gettempdir()) / "harness-bench-svd-plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
        "axes.labelcolor": "#253047", "text.color": "#182237",
        "xtick.color": "#475569", "ytick.color": "#475569",
    })
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.23, wspace=0.24)
    fig.text(0.07, 0.94, "How many tasks capture the harness benchmark?", fontsize=22, weight="bold")
    fig.text(0.07, 0.88, "78 tasks × 9 harnesses · equal average of DeepSeek, GLM and Muse · 3,510 retained trials", fontsize=12, color="#475569")
    ax = axes[0]
    components = np.arange(1, len(harnesses) + 1)
    ax.plot(components, centered_svd["cumulative_explained_fraction"], "o-", lw=2.4,
            color="#2563eb", label="Full matrix variance")
    ax.plot(components, interactions_svd["cumulative_explained_fraction"], "o-", lw=2.4,
            color="#a855f7", label="Harness differences, difficulty removed")
    ax.axhline(0.9, color="#94a3b8", ls="--", lw=1.2)
    ax.text(8.95, 0.905, "90%", ha="right", va="bottom", fontsize=9, color="#64748b")
    ax.annotate("4 components\n91.0%", (4, centered_svd["cumulative_explained_fraction"][3]),
                xytext=(4.6, 0.72), arrowprops={"arrowstyle": "-", "color": "#2563eb"}, color="#2563eb")
    ax.annotate("7 components\n95.3%", (7, interactions_svd["cumulative_explained_fraction"][6]),
                xytext=(6.4, 0.48), arrowprops={"arrowstyle": "-", "color": "#a855f7"}, color="#9333ea")
    ax.set(title="SVD: latent components", xlabel="Number of components", ylabel="Cumulative variance explained",
           xlim=(0.8, 9.2), ylim=(0, 1.04), xticks=components)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e2e8f0")
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    ax = axes[1]
    budgets = np.arange(len(tasks) + 1)
    ax.fill_between(budgets, random_summary["p10"], random_summary["p90"], color="#94a3b8", alpha=0.16)
    ax.plot(budgets, random_summary["median"], color="#94a3b8", lw=1.7,
            label="Random subsets: median / 10–90% range")
    ax.plot(budgets, historical, color="#ea580c", lw=2.2, label="Historical fit: learned task baseline")
    ax.plot(budgets, cv["r2"], color="#0f766e", lw=2.5, label="Held-out harness: measured + predicted")
    ax.axhline(0.9, color="#94a3b8", ls="--", lw=1.2)
    k = first_threshold(cv["r2"], 0.9)
    ax.scatter([3, k], [historical[3], cv["r2"][k]], color=["#ea580c", "#0f766e"], zorder=5)
    ax.annotate(f"3 tasks: {historical[3]:.1%}\nhistorical fit", (3, historical[3]),
                xytext=(11, 0.53), arrowprops={"arrowstyle": "-", "color": "#ea580c"}, color="#c2410c", fontsize=9)
    ax.annotate(f"{k} tasks: {cv['r2'][k]:.1%}\nheld-out reconstruction", (k, cv["r2"][k]),
                xytext=(38, 0.67), arrowprops={"arrowstyle": "-", "color": "#0f766e"}, color="#0f766e", fontsize=9)
    ax.set(title="Actual task subsets", xlabel="Number of tasks measured", ylabel="Full matrix reconstructed (R²)",
           xlim=(0, len(tasks)), ylim=(min(-0.01, np.floor(10 * min(random_summary["p10"])) / 10), 1.04),
           xticks=[0, 10, 20, 30, 40, 50, 60, 70, 78])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e2e8f0")
    ax.legend(loc="lower right", fontsize=8.3, frameon=False)
    fig.text(0.07, 0.105, "Historical compression is not a guarantee for a new harness. Each validation fold selects tasks and learns weights from the other 8 harnesses.",
             fontsize=10, color="#475569")
    fig.text(0.07, 0.06, "The green curve includes tasks already measured. The report separately scores only unmeasured tasks; random bands are sampling ranges, not confidence intervals.",
             fontsize=9, color="#64748b")
    for extension in ("png", "svg"):
        fig.savefig(output / f"task-reconstruction.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(budgets[:-1], cv["unmeasured_r2"][:-1], lw=2.3, color="#0f766e", label="Unmeasured tasks only")
    ax.plot(budgets, cv["r2"], lw=2, color="#2563eb", label="Measured + predicted tasks")
    ax.axhline(0.9, color="#94a3b8", ls="--", lw=1)
    ax.set(title="Held-out harnesses: prediction accuracy versus measurement coverage",
           xlabel="Number of tasks measured", ylabel="Reconstruction R²", xlim=(0, 78), ylim=(-0.05, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e2e8f0")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "unmeasured-task-validation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.6, 19))
    view_order = order[:8] + [i for i in range(len(tasks)) if i not in order[:8]]
    heat = ax.imshow(matrix[view_order], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(harnesses)), [HARNESS_LABELS[h] for h in harnesses], rotation=42, ha="right")
    labels = [f"{i + 1}. {tasks[t]}" if i < 8 else tasks[t] for i, t in enumerate(view_order)]
    ax.set_yticks(range(len(tasks)), labels, fontsize=8)
    ax.axhline(7.5, color="white", lw=2)
    for label in ax.get_yticklabels()[:8]:
        label.set_weight("bold")
    ax.set_title("Completion-rate matrix: first 8 selected tasks at the top\nEqual model weights; repetitions averaged within model", pad=15)
    colorbar = fig.colorbar(heat, ax=ax, fraction=0.033, pad=0.035)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("Observed completion rate")
    fig.tight_layout()
    fig.savefig(output / "completion-matrix.png", dpi=170)
    plt.close(fig)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def write_report(output, result):
    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
        ) + "</tbody></table>"
    def image_data(name):
        return "data:image/png;base64," + base64.b64encode((output / name).read_bytes()).decode()
    def percent(value):
        return "—" if value is None else f"{value:.1%}"
    cv = result["validation"]
    curve = cv["curve"]
    historical = result["historical_reconstruction"]["curve"]
    k90 = cv["tasks_for_90_percent_full_reconstruction"]
    selected_table = table(["Order", "Task", "Historical matrix R²", "Harness contrasts retained"], [
        [item["selection_order"], item["task_id"], percent(item["historical_r2"]), percent(item["historical_contrasts_explained"])]
        for item in result["task_selection"][:8]
    ])
    validation_table = table(["Tasks", "Historical fit", "Held-out full matrix", "Only unmeasured tasks", "Unmeasured RMSE (pp)"], [
        [k, percent(historical[k]["r2"]), percent(curve[k]["r2"]), percent(curve[k]["unmeasured_r2"]),
         "—" if curve[k]["unmeasured_rmse"] is None else f"{100 * curve[k]['unmeasured_rmse']:.1f}"]
        for k in sorted({0, 1, 2, 3, 4, 5, 8, 10, 20, 30, k90, 40, 50, 60, 78})
    ])
    recommendation = ", ".join(html.escape(item["task_id"]) for item in result["task_selection"][:3])
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness Bench: SVD and task reconstruction</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#182237;background:#f8fafc;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 28px}}
h1{{font-size:34px;line-height:1.15}}h2{{margin-top:40px}}p{{max-width:960px}}img{{max-width:100%;background:white;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #e2e8f0}}
th{{background:#e2e8f0}}code{{font:13px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}}.note{{padding:15px 20px;background:#e0f2fe;border-left:4px solid #0284c7}}
.scroll{{overflow:auto}}a{{color:#2563eb}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}
</style><main><h1>How many tasks capture the harness benchmark?</h1>
<p>78 shared tasks × 9 harnesses × 3 models, from 3,510 retained trials. Models receive equal weight;
DeepSeek's three repetitions are averaged within each task first, while GLM and Muse each have one repetition.
These are the saved past results retrieved on 9 September 2026, not a newly executed benchmark.</p>
<div class="note"><strong>Four SVD components explain 91.0% of the full matrix variance.</strong>
Three selected tasks give a 90.3% historical reconstruction with a learned per-task baseline.
In a stricter held-out-harness test, this selection procedure first exceeds 90% full-matrix reconstruction at {k90} measured tasks.
That score includes measured tasks; only the remaining tasks have to be predicted.</div>
<p><img alt="SVD explained variance and task-subset reconstruction curves" src="{image_data('task-reconstruction.png')}"></p>
<h2>What the SVD says</h2><p>The input A has tasks as rows, harnesses as columns, and mean observed completion rate as its entries.
For the main SVD, subtract each harness's mean across tasks: X = A − column_mean(A). Decompose X = U S Vᵀ.
Cumulative explained variance is sum(s₁²…sₖ²) / sum(s²), consistent with the
<a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html">NumPy SVD definition</a>.
The first component explains 78.2%; four explain 91.0%, and six explain 96.0%.</p>
<p>Shared task difficulty dominates that compression. With both task and harness means removed,
X_interaction = A − row_mean(A) − column_mean(A) + grand_mean(A), four components explain only 70.9% of the residual harness differences.
Seven are needed to pass 90% (95.3%). An uncentered SVD gives 93.2% energy in one component,
but that includes the nonzero average completion level and is not the centered variance result.</p>
<h2>Which actual tasks were selected?</h2><p>The first three candidate probes are <strong>{recommendation}</strong>.
The complete 78-task nested order and each task's harness completion rates are included in <code>task-selection.json</code>.
These are informative probes for a learned reconstruction model; an unweighted average of their completion rates is not a calibrated benchmark score.</p>
<div class="scroll">{selected_table}</div>
<p>Greedy row selection chooses the task giving the largest one-step reduction in remaining squared reconstruction error.
It is a deterministic heuristic, not a proof that the subset is globally optimal. The first eight independent task contrasts span the nine historical harness columns
after task centering. Beyond that, additional tasks are chosen to improve conditioning using an A-optimal criterion on the SVD task basis.</p>
<p>Historical reconstruction uses μ = row_mean(A), B = A − μ and selected task rows B_S:
Â = μ + B B_S⁺ B_S. The superscript ⁺ is the
<a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html">SVD pseudoinverse</a>.
The stored weights and task baseline can be applied to a new harness's selected-task observations.
Even zero selected tasks explains 77.5% of centered historical variance because the learned task-difficulty baseline is already known.
Three tasks retain only {result['task_selection'][2]['historical_contrasts_explained']:.1%} of the within-task harness contrasts.
Exact reconstruction at eight tasks follows from the small column count, so it does not establish an eight-task benchmark for unseen harnesses.</p>
<h2>Does a small subset generalize?</h2><p>Leave out one complete harness column, average the same three models,
and learn task selection, task baselines and reconstruction weights using only the other eight harnesses.
For the held-out harness, reveal only the selected task outcomes. Predict the rest, clip predicted rates to [0,1],
and retain revealed measurements exactly. Repeat for all nine harnesses. Each fold has its own training-only selected subset.</p>
<div class="scroll">{validation_table}</div>
<p>The full-matrix score is 1 − Σ(A − Â)² / Σ(A − column_mean(A))².
The unmeasured-only score uses those same per-harness centering constants but restricts both sums to entries that were not revealed.
At all 78 measured tasks, the full score is necessarily 100% and the unmeasured score is undefined.</p>
<p><img alt="Held-out reconstruction scored on all tasks and on unmeasured tasks only" src="{image_data('unmeasured-task-validation.png')}"></p>
<p>At {k90} measured tasks, the full-matrix score is {curve[k90]['r2']:.1%}, the unmeasured-only score is {curve[k90]['unmeasured_r2']:.1%},
and unmeasured completion-rate RMSE is {100 * curve[k90]['unmeasured_rmse']:.1f} percentage points.
With very few tasks, fitting all available contrast directions is unstable and can perform worse than the learned task baseline.
The random control uses {result['random_control']['permutations']} fixed-seed random task orders, the same prediction method and the same folds;
its band is the 10th–90th percentile over task samples, not a statistical confidence interval.</p>
<h2>Completion-rate matrix</h2><p>The first eight selected tasks are listed at the top; all remaining tasks are alphabetical.
Cells are observed completion rates averaged equally across models.</p>
<p><img alt="All 78 tasks by nine harnesses, colored by completion rate" src="{image_data('completion-matrix.png')}"></p>
<h2>Scope and limits</h2><p>Normal scored failures and retained execution errors count as unsuccessful trials, matching the existing benchmark's aggregation.
The discarded Muse cohort and invalid original GLM/Muse Codex request-failure cohorts are excluded;
their corrected Codex reruns are included. Harness versions match across the retained models, but historical reasoning settings and DeepSeek model IDs differ between source cohorts.
Single GLM/Muse trials and only three DeepSeek trials do not estimate noise-free success probabilities.
The held-out results validate this particular reconstruction procedure on nine existing harnesses; they are not a lower bound on necessary task count
or a guarantee for future models, versions or harnesses. No benchmark or model calls were run for this analysis.</p>
<h2>Reproduce and inspect</h2><p><code>uv run --group dev python scripts/analyze_task_svd.py</code></p>
<p>The default input is the committed aggregate <code>analysis/task-selection-input.json</code>.
Raw rollout records and general cost/export data are excluded. Optional <code>--snapshot</code> and <code>--reference</code>
arguments rerun the original trial-level audit when both original files are available.</p>
<p>Output files: <code>analysis.json</code> (all spectra, curves and provenance), <code>completion-matrix.json</code>,
<code>task-selection.json</code>, <code>reconstruction-weights-3-tasks.json</code>, <code>reconstruction-weights-5-tasks.json</code>,
<code>reconstruction-weights-8-tasks.json</code>, and standalone PNG/SVG charts.</p>
<details><summary>Data provenance and checks</summary><p>{html.escape(result['provenance']['verification'])}</p>
<p>Raw snapshot SHA-256: <code>{result['provenance']['snapshot_sha256']}</code><br>
Source: <code>{html.escape(result['provenance']['source_database'])}</code><br>
Snapshot retrieved: {html.escape(result['provenance']['retrieved_at'])}<br>
Trial statuses: {html.escape(str(result['provenance']['status_counts']))}</p></details>
</main></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="Committed aggregate input for offline reproduction")
    parser.add_argument("--snapshot", type=Path, help="Optional original raw rollout snapshot; requires --reference")
    parser.add_argument("--reference", type=Path, help="Optional original aggregate reference; requires --snapshot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/task-svd-2026-10-08")
    parser.add_argument("--random-subsets", type=int, default=100)
    args = parser.parse_args()
    if bool(args.snapshot) != bool(args.reference):
        parser.error("--snapshot and --reference must be supplied together")
    assert args.random_subsets >= 1
    args.output.mkdir(parents=True, exist_ok=True)
    matrix, cube, tasks, harnesses, model_keys, audit = load_matrix(args.snapshot, args.reference) if args.snapshot else load_aggregate(args.matrix)
    centered = matrix - matrix.mean(axis=0)
    baseline = matrix.mean(axis=1)
    contrasts = matrix - baseline[:, None]
    interactions = centered - centered.mean(axis=1, keepdims=True)
    centered_svd = spectrum(centered)
    interaction_svd = spectrum(interactions)
    raw_svd = spectrum(matrix)
    contrast_svd = spectrum(contrasts)
    order = task_order(contrasts)
    denominator = np.sum(centered**2)
    historical, contrast_scores = [], []
    for budget in range(len(tasks) + 1):
        selected = order[:budget]
        fitted = baseline[:, None] + contrasts @ np.linalg.pinv(contrasts[selected], rcond=RCOND) @ contrasts[selected] if budget else baseline[:, None]
        squared_error = np.sum((matrix - fitted)**2)
        historical.append(1 - squared_error / denominator)
        contrast_scores.append(1 - squared_error / np.sum(contrasts**2))
    assert np.all(np.diff(historical) >= -1e-10)
    assert np.isclose(historical[contrast_svd["rank"]], 1, atol=1e-12)
    fold_orders = [task_order(np.delete(matrix, h, axis=1) - np.delete(matrix, h, axis=1).mean(axis=1, keepdims=True))
                   for h in range(len(harnesses))]
    cv = cross_validate(matrix, fold_orders)
    assert np.allclose(cv["predictions"][-1], matrix, atol=0)
    rng = np.random.default_rng(SEED)
    random_curves = []
    for _ in range(args.random_subsets):
        random_order = rng.permutation(len(tasks)).tolist()
        random_curves.append(cross_validate(matrix, [random_order] * len(harnesses))["r2"])
    random_curves = np.array(random_curves)
    random_summary = {"median": np.median(random_curves, axis=0),
                      "p10": np.percentile(random_curves, 10, axis=0),
                      "p90": np.percentile(random_curves, 90, axis=0)}
    selection = [{
        "selection_order": i + 1, "task_id": tasks[t],
        "historical_r2": historical[i + 1], "historical_contrasts_explained": contrast_scores[i + 1],
        "mean_completion_rate": matrix[t].mean(), "harness_completion_rates": dict(zip(harnesses, matrix[t])),
        "selection_phase": "greedy reconstruction" if i < contrast_svd["rank"] else "A-optimal conditioning",
    } for i, t in enumerate(order)]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": audit,
        "shape": {"tasks": len(tasks), "harnesses": len(harnesses), "models": len(model_keys)},
        "aggregation": "Per model/task/harness: mean(status == passed) over repetitions; then equal mean over the three models. Retained execution errors are unsuccessful trials.",
        "tasks": tasks, "harnesses": harnesses, "models": model_keys,
        "model_mean_completion_rates": dict(zip(model_keys, cube.mean(axis=(1, 2)))),
        "harness_mean_completion_rates": dict(zip(harnesses, matrix.mean(axis=0))),
        "svd": {
            "column_centered_full_variance": centered_svd,
            "double_centered_harness_interactions": interaction_svd,
            "uncentered_energy_not_variance": raw_svd,
            "task_centered_harness_contrasts": contrast_svd,
        },
        "historical_reconstruction": {
            "formula": "mu = row_mean(A); B = A - mu; A_hat = mu + B @ pinv(B[selected]) @ B[selected]",
            "baseline_source": "Per-task means learned from all historical harnesses, not estimated from the selected tasks alone.",
            "tasks_for_90_percent": first_threshold(historical, 0.9),
            "tasks_for_95_percent": first_threshold(historical, 0.95),
            "curve": [{"tasks": k, "r2": v, "harness_contrasts_explained": contrast_scores[k]} for k, v in enumerate(historical)],
        },
        "task_selection": selection,
        "validation": {
            "method": "Leave-one-harness-out. Selection, task means and reconstruction weights refit from the other eight columns in each fold. Selected held-out observations are revealed; all others are predicted. Predictions clipped to [0,1].",
            "score": "Pooled R2: 1-SSE/sum((A-column_mean(A))**2). Full score includes measured tasks. Unmeasured-only numerator and denominator restricted to unseen entries, retaining the same per-harness centering constants.",
            "tasks_for_90_percent_full_reconstruction": first_threshold(cv["r2"], 0.9),
            "tasks_for_95_percent_full_reconstruction": first_threshold(cv["r2"], 0.95),
            "tasks_for_90_percent_unmeasured_prediction": first_threshold(cv["unmeasured_r2"], 0.9),
            "curve": [{"tasks": k, "r2": cv["r2"][k], "unmeasured_r2": cv["unmeasured_r2"][k],
                       "rmse": cv["rmse"][k], "unmeasured_rmse": cv["unmeasured_rmse"][k]}
                      for k in range(len(tasks) + 1)],
            "folds": [{"held_out_harness": harnesses[h], "training_task_order": [tasks[t] for t in fold_orders[h]]}
                      for h in range(len(harnesses))],
        },
        "random_control": {"seed": SEED, "permutations": args.random_subsets, "band": "10th–90th percentile over random subsets, not confidence intervals",
                           "curve": [{"tasks": k, "median_r2": random_summary["median"][k],
                                      "p10_r2": random_summary["p10"][k], "p90_r2": random_summary["p90"][k]}
                                     for k in range(len(tasks) + 1)]},
        "limitations": [
            "Historical fit is not out-of-sample prediction. Any eight independent task contrasts can span nine centered historical harness columns.",
            "Selected task means alone are not an unbiased estimate of harness completion rates. Use learned calibration/reconstruction weights.",
            "Reported task thresholds are results for this deterministic greedy procedure, not proven minimum task counts.",
            "Only nine harness columns and three model cohorts; GLM/Muse have one trial per task and DeepSeek has three.",
            "Model averaging can hide model-specific failures; historical reasoning settings and DeepSeek submitted IDs differ across source cohorts.",
            "Full validation scores include revealed task observations, so measure-only coverage increases the score; consult unmeasured-only metrics as well.",
        ],
        "software": {"numpy_version": np.__version__, "random_seed": SEED, "pseudoinverse_rcond": RCOND},
    }
    result = plain(result)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "completion-matrix.json").write_text(json.dumps({"tasks": tasks, "harnesses": harnesses, "models": model_keys,
                                                               "rates": matrix.tolist(), "per_model_rates": cube.tolist()}, indent=2) + "\n")
    (args.output / "task-selection.json").write_text(json.dumps(plain(selection), indent=2) + "\n")
    for count in (3, 5, 8):
        selected = order[:count]
        weights = contrasts @ np.linalg.pinv(contrasts[selected], rcond=RCOND)
        assert np.allclose(baseline[:, None] + weights @ contrasts[selected],
                           baseline[:, None] + contrasts @ np.linalg.pinv(contrasts[selected], rcond=RCOND) @ contrasts[selected])
        model = {"training_tasks": tasks, "selected_tasks": [tasks[t] for t in selected],
                 "task_baseline": baseline.tolist(), "weights": weights.tolist(),
                 "predict_formula": "clip(task_baseline + weights @ (new_harness_selected_rates - task_baseline[selected_indices]), 0, 1); retain measured entries exactly",
                 "selected_indices": selected, "training_harnesses": harnesses,
                 "warning": "Trained on all nine historical harnesses. Historical reconstruction is not a guarantee for a new harness."}
        (args.output / f"reconstruction-weights-{count}-tasks.json").write_text(json.dumps(model, indent=2) + "\n")
    make_plots(args.output, matrix, tasks, harnesses, order, centered_svd, interaction_svd, historical, cv, random_summary)
    write_report(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()),
                      "svd_components_90_percent": centered_svd["components_for_90_percent"],
                      "interaction_components_90_percent": interaction_svd["components_for_90_percent"],
                      "historical_tasks_90_percent": first_threshold(historical, 0.9),
                      "held_out_tasks_90_percent_full": first_threshold(cv["r2"], 0.9),
                      "held_out_tasks_90_percent_unmeasured": first_threshold(cv["unmeasured_r2"], 0.9),
                      "first_eight_tasks": [tasks[t] for t in order[:8]]}, indent=2))


if __name__ == "__main__":
    main()
