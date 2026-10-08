#!/usr/bin/env python3
"""Build a portable HTML explanation of the five-task multiagent pilot.

Reads the existing SVD analysis; never starts benchmark or model runs.
    uv run --group dev python scripts/build_multiagent_pilot_report.py
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "outputs/analysis/task-svd-2026-10-08"
EXPECTED_TASKS = [
    "torch-pipeline-parallelism", "pypi-server", "fix-code-vulnerability",
    "qemu-startup", "sanitize-git-repo",
]
COLORS = {"matrix": "#176582", "contrasts": "#8462a6", "prediction": "#b85c3b"}


def pct(value):
    return f"{100 * value:.1f}%"


def chart_svg(analysis):
    width, height = 930, 355
    left, right, top, bottom = 63, 30, 20, 48
    def x(k):
        return left + k * (width - left - right) / 8
    def y(value):
        return top + (1 - value) * (height - top - bottom)
    historical = analysis["historical_reconstruction"]["curve"][:9]
    validation = analysis["validation"]["curve"][:9]
    series = {
        "matrix": [row["r2"] for row in historical],
        "contrasts": [row["harness_contrasts_explained"] for row in historical],
        "prediction": [row["unmeasured_r2"] for row in validation],
    }
    elements = [f'<svg id="budget-chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-description">',
                '<title id="chart-title">Reconstruction quality for zero to eight measured tasks</title>',
                '<desc id="chart-description">Historical full matrix reconstruction increases to 95.3 percent at five tasks. Harness contrast reconstruction reaches 79.1 percent. Held-out prediction on unmeasured tasks is 48.3 percent at five tasks. The historical fit includes a learned per-task baseline.</desc>']
    for value in (0, 0.2, 0.4, 0.6, 0.8, 1):
        elements.append(f'<line x1="{left}" x2="{width-right}" y1="{y(value)}" y2="{y(value)}" class="grid-line"/><text x="{left-12}" y="{y(value)+4}" text-anchor="end" class="axis-text">{value:.0%}</text>')
    elements.append(f'<line x1="{left}" x2="{width-right}" y1="{y(.9)}" y2="{y(.9)}" class="target-line"/>')
    elements.append(f'<text x="{width-right}" y="{y(.9)-7}" text-anchor="end" class="axis-text">90% reference</text>')
    for k in range(9):
        elements.append(f'<text x="{x(k)}" y="{height-bottom+23}" text-anchor="middle" class="axis-text">{k}</text>')
    elements.append(f'<text x="{width/2}" y="{height-3}" text-anchor="middle" class="axis-text">Number of selected tasks</text>')
    elements.append(f'<line id="focus-line" x1="{x(5)}" x2="{x(5)}" y1="{top}" y2="{height-bottom}" class="focus-line"/>')
    for name, values in series.items():
        coordinates = " ".join(f"{x(k):.2f},{y(v):.2f}" for k, v in enumerate(values))
        dashed = ' stroke-dasharray="7 5"' if name == "prediction" else ""
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{COLORS[name]}" stroke-width="3"{dashed}/>')
        for k, value in enumerate(values):
            elements.append(f'<circle cx="{x(k)}" cy="{y(value)}" r="3.3" fill="{COLORS[name]}"><title>{name}: {k} tasks, {pct(value)}</title></circle>')
        elements.append(f'<circle id="focus-{name}" cx="{x(5)}" cy="{y(values[5])}" r="6" fill="{COLORS[name]}" stroke="white" stroke-width="2"/>')
    elements.append('</svg>')
    return "".join(elements)


def export_chart(directory, analysis):
    cache = Path(tempfile.gettempdir()) / "harness-bench-svd-plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(top=.8, bottom=.26, left=.09, right=.97)
    fig.text(.09, .94, "Five tasks for a multiagent harness pilot", fontsize=20, weight="bold", color="#173c4f")
    fig.text(.09, .87, "Historical evidence: 78 tasks × 9 harnesses · three models averaged equally", fontsize=11, color="#536775")
    historical = analysis["historical_reconstruction"]["curve"][:9]
    validation = analysis["validation"]["curve"][:9]
    series = [("matrix", [r["r2"] for r in historical], "Historical full matrix fit"),
              ("contrasts", [r["harness_contrasts_explained"] for r in historical], "Historical harness contrasts"),
              ("prediction", [r["unmeasured_r2"] for r in validation], "Held-out prediction: unmeasured tasks")]
    for key, values, label in series:
        ax.plot(range(9), values, "o--" if key == "prediction" else "o-", color=COLORS[key], lw=2.3, label=label)
        ax.scatter([5], [values[5]], color=COLORS[key], s=80, zorder=5)
        ax.annotate(pct(values[5]), (5, values[5]), xytext=(10, 7), textcoords="offset points", color=COLORS[key], fontsize=10)
    ax.axvline(5, color="#a8b5bc", ls="--", lw=1)
    ax.axhline(.9, color="#bdc7ce", ls=":", lw=1)
    ax.set(xlim=(-.1, 8.1), ylim=(-.02, 1.04), xticks=range(9), xlabel="Number of selected tasks", ylabel="Reconstruction R² / fraction of contrasts retained")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.text(.09, .12, "Historical fits reuse learned task baselines and weights. Validation refits the selection separately in each held-out harness fold.", fontsize=9, color="#536775")
    fig.text(.09, .065, "These results support a screening pilot. A new main-agent/subagent protocol still needs benchmark measurements and validation.", fontsize=9, color="#536775")
    for extension in ("png", "svg"):
        fig.savefig(directory / f"five-task-pilot-reconstruction.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def task_table(analysis):
    rows = []
    for task in analysis["task_selection"][:5]:
        rows.append(f'<tr><td class="order">0{task["selection_order"]}</td><td><code>{html.escape(task["task_id"])}</code></td><td>{pct(task["historical_r2"])}</td><td>{pct(task["historical_contrasts_explained"])}</td></tr>')
    return "".join(rows)


def heatmap_table(analysis):
    labels = ["Claude Code", "Codex", "DeepSeek Harness", "Hermes", "OMP", "OpenCode", "Pi Agent", "Pi RLM", "PrimeAgent"]
    header = '<tr><th scope="col">Task</th>' + "".join(f'<th scope="col">{label}</th>' for label in labels) + '</tr>'
    rows = []
    for task in analysis["task_selection"][:5]:
        cells = []
        for rate in task["harness_completion_rates"].values():
            # A pale sequential scale keeps cell labels readable without encoding rank.
            intensity = rate * .75
            red = round(244 - intensity * 88)
            green = round(247 - intensity * 46)
            blue = round(245 - intensity * 37)
            cells.append(f'<td style="background:rgb({red},{green},{blue})">{rate:.0%}</td>')
        rows.append(f'<tr><th scope="row"><code>{html.escape(task["task_id"])}</code></th>{"".join(cells)}</tr>')
    return '<table class="heatmap"><thead>' + header + '</thead><tbody>' + "".join(rows) + '</tbody></table>'


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="How five tasks were selected from historical Harness Bench results, what they reconstruct, and how to use them for a controlled main-agent/subagent pilot.">
<title>Five-task pilot · Multiagent harness benchmarking</title>
<style>
:root{--ink:#18394c;--muted:#576b77;--paper:#fbfaf6;--line:#dde3e3;--blue:#176582;--purple:#8462a6;--rust:#b85c3b;--soft:#edf2ef;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:78px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--blue);text-underline-offset:3px}a:hover{color:var(--rust)}button,input,a{touch-action:manipulation}button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #e39f4e;outline-offset:4px}
.skip{position:absolute;left:-9999px}.skip:focus{left:16px;top:12px;z-index:10;background:white;padding:10px}.nav{position:sticky;top:0;z-index:5;background:rgba(251,250,246,.97);border-bottom:1px solid var(--line)}
.nav-inner{max-width:1180px;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:20px;padding:17px 28px}.brand{font-size:12px;letter-spacing:.14em;font-weight:750}.nav-links{display:flex;gap:22px;flex-wrap:wrap}.nav a{text-decoration:none;font-size:13px;color:var(--muted)}
main{max-width:1180px;margin:auto;padding:0 28px 55px}.hero{padding:61px 0 37px;max-width:990px}.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--rust);font-weight:750}.hero h1{font-size:clamp(34px,4.3vw,56px);line-height:1.1;letter-spacing:-.035em;margin:17px 0 24px;font-weight:650}.lead{font-size:20px;line-height:1.55;color:var(--muted);max-width:920px}.meta{font-size:12px;color:var(--muted);margin-top:24px}
section{padding:34px 0 40px;border-top:1px solid var(--line)}h2{font-size:28px;line-height:1.25;letter-spacing:-.02em;margin:0 0 16px;font-weight:650}h3{font-size:17px;line-height:1.35;margin:0 0 10px}p{margin:0 0 16px;max-width:980px}.section-intro{color:var(--muted);max-width:940px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;margin:29px 0}.stat{padding:19px 0;border-top:2px solid currentColor}.stat.matrix{color:var(--blue)}.stat.contrasts{color:var(--purple)}.stat.prediction{color:var(--rust)}.stat strong{display:block;font-size:43px;line-height:1.12;font-weight:650;letter-spacing:-.04em}.stat .stat-label{display:block;font-size:14px;font-weight:700;margin:12px 0 6px}.stat p{font-size:13px;color:var(--muted);line-height:1.55;margin:0}.status{padding:17px 21px;border-left:3px solid var(--rust);background:#f6eee7;font-size:14px}.status strong{color:var(--ink)}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:5px}table{width:100%;border-collapse:collapse;font-size:14px;background:#fff}th,td{padding:13px 16px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}th{font-weight:650;background:var(--soft);font-size:12px}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}td.order{color:var(--rust);font:13px var(--mono)}td code{font-size:13px}code{font-family:var(--mono);font-size:.87em;overflow-wrap:anywhere}.table-note,.caption{font-size:12px;line-height:1.65;color:var(--muted);margin-top:12px}.heatmap{min-width:940px;font-variant-numeric:tabular-nums}.heatmap th:first-child{min-width:240px}.heatmap td{text-align:center;font-size:13px}.heatmap th{text-align:center}.heatmap th[scope=row]{text-align:left;background:white;font-weight:500}
.figure{background:white;border:1px solid var(--line);border-radius:6px;padding:24px;margin:26px 0 18px}.figure-title{font-size:16px;font-weight:650;margin-bottom:8px}.legend{display:flex;gap:21px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend span{display:flex;align-items:center;gap:8px}.legend i{width:23px;height:3px;background:currentColor}.legend .matrix{color:var(--blue)}.legend .contrasts{color:var(--purple)}.legend .prediction{color:var(--rust)}.legend .prediction i{background:repeating-linear-gradient(to right,currentColor 0 7px,transparent 7px 10px)}svg{display:block;max-width:100%;height:auto;width:100%}.axis-text{font-family:system-ui,sans-serif;font-size:12px;fill:#647580}.grid-line{stroke:#e4e9eb;stroke-width:1}.target-line{stroke:#b9c2c7;stroke-dasharray:4 5}.focus-line{stroke:#bcc5c9;stroke-dasharray:3 4}
.budget-control{display:grid;grid-template-columns:190px 1fr 170px;align-items:center;gap:22px;margin-top:16px;font-size:14px}.budget-control label{font-weight:650}.budget-control input{width:100%;accent-color:var(--blue)}.budget-control output{font-family:var(--mono);font-size:13px;text-align:right}.mini-scores{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:17px}.mini-score{border:1px solid var(--line);padding:12px 15px;background:var(--paper);border-radius:4px}.mini-score b{display:block;font-size:22px;font-weight:650}.mini-score span{font-size:11px;color:var(--muted)}.selected-list{font:12px/1.7 var(--mono);color:var(--muted);margin-top:12px;overflow-wrap:anywhere}.figure-actions{display:flex;gap:16px;font-size:12px;margin-top:17px}
.steps{counter-reset:steps;display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:25px 0}.step{position:relative;padding:22px 23px 22px 59px;border:1px solid var(--line);background:#fff;border-radius:5px}.step::before{counter-increment:steps;content:counter(steps,decimal-leading-zero);position:absolute;top:22px;left:20px;font:13px var(--mono);color:var(--rust)}.step p{font-size:14px;color:var(--muted);margin:0}.equation{padding:18px 21px;border:1px solid var(--line);background:var(--soft);font:13px/1.9 var(--mono);overflow:auto;border-radius:4px;margin:20px 0}.two-cols{display:grid;grid-template-columns:1fr 1fr;gap:27px}.inset{background:white;padding:25px;border:1px solid var(--line);border-radius:5px}.inset p{font-size:14px;color:var(--muted);margin-bottom:0}.inset ul,.protocol ul{padding-left:19px;font-size:14px;color:var(--muted);margin:10px 0 0}.inset li,.protocol li{margin:8px 0}
.architecture{padding:27px 23px;background:#edf2ef;border:1px solid #dbe2df;border-radius:6px;margin:25px 0}.architecture-label{text-align:center;font-size:12px;color:var(--muted);margin-bottom:14px}.coordinator{max-width:560px;margin:auto;padding:20px 24px;border:1px solid #a9bebf;border-radius:5px;background:#fff;text-align:center}.coordinator strong{font-size:17px}.coordinator span{display:block;color:var(--muted);font-size:12px;margin-top:6px}.connector{text-align:center;padding:8px;color:#81969b;font-size:22px}.worker-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.worker{padding:15px 13px;background:white;border:1px solid #c9d6d3;border-radius:4px}.worker .worker-label{display:block;font:10px var(--mono);color:var(--rust);margin-bottom:8px}.worker code{font-size:11px;line-height:1.6;display:block;min-height:40px}.worker small{display:block;margin-top:11px;padding-top:10px;border-top:1px solid var(--line);font-size:10px;color:var(--muted)}.grader{margin-top:17px;padding:13px;border-top:1px dashed #b1c4c0;text-align:center;font-size:12px;color:var(--muted)}
.protocol{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:23px 0}.protocol>div{border:1px solid var(--line);padding:22px;background:#fff;border-radius:5px}.comparison{width:100%}.comparison td:first-child{width:24%;font-weight:650}.measure-list{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.measure-list>div{padding:20px;border-top:2px solid #9eb3b9;background:#fff}.measure-list p{font-size:13px;color:var(--muted);margin:0}.decision{background:#eaf0f4;padding:22px 25px;border-left:3px solid var(--blue);border-radius:3px;font-size:14px}
details{margin:15px 0;border:1px solid var(--line);border-radius:4px;background:white}summary{cursor:pointer;padding:15px 20px;font-size:14px;font-weight:650}details .detail-content{padding:0 20px 20px;font-size:13px;color:var(--muted)}.detail-content code{word-break:break-all}.source-table th{width:25%}.downloads{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.button{font:13px system-ui,sans-serif;padding:11px 16px;background:var(--ink);color:white;border:1px solid var(--ink);border-radius:4px;text-decoration:none;cursor:pointer}.button.secondary{background:transparent;color:var(--ink);border-color:#bac7ce}.button:hover{background:var(--blue);color:white}.footer{font-size:12px;color:var(--muted);padding-top:20px;border-top:1px solid var(--line)}.no-js{font-size:13px;background:#fff1d4;padding:12px}
@media(max-width:760px){.nav-inner{padding:13px 20px;align-items:flex-start;gap:12px}.nav-links{gap:10px 16px;justify-content:flex-end}.brand{font-size:10px;min-width:105px}main{padding:0 20px 35px}.hero{padding-top:40px}.lead{font-size:17px}.stats{gap:17px}.stat strong{font-size:34px}.stat .stat-label{font-size:12px}.stat p{font-size:12px}h2{font-size:25px}.steps,.two-cols,.protocol{grid-template-columns:1fr}.figure{padding:17px}.budget-control{grid-template-columns:1fr auto;gap:9px}.budget-control input{grid-column:1/-1;grid-row:2}.budget-control output{grid-column:2;grid-row:1}.mini-score{padding:10px}.mini-score b{font-size:19px}.worker-grid{grid-template-columns:repeat(2,1fr)}.worker code{min-height:0}.measure-list{grid-template-columns:1fr}.legend{gap:10px 16px}.comparison td{padding:12px}.comparison td:first-child{width:32%}}
@media(max-width:420px){.nav-links{gap:7px 12px}.nav a{font-size:11px}.stats{grid-template-columns:1fr;gap:10px}.stat{padding:15px 0;display:grid;grid-template-columns:90px 1fr;column-gap:14px}.stat strong{grid-row:1/3}.stat .stat-label{margin:0 0 5px}.stat p{grid-column:2}.mini-scores{grid-template-columns:1fr}.mini-score{display:flex;align-items:center;gap:15px}.hero h1{font-size:35px}.heatmap th:first-child{min-width:210px}}
@media print{body{background:white;font-size:11pt}.nav,.downloads,.figure-actions,.budget-control,.no-print{display:none}main{padding:0;max-width:none}.hero{padding-top:0}h1{font-size:28pt!important}.lead{font-size:14pt}.stat strong{font-size:28pt}section{padding:23px 0}a{color:inherit;text-decoration:none}.figure,.table-wrap,.architecture,.inset,.step{break-inside:avoid}details .detail-content{display:block}.worker-grid{grid-template-columns:repeat(5,1fr)}.steps,.two-cols,.protocol{grid-template-columns:1fr 1fr}.table-wrap{overflow:visible}.heatmap{min-width:0;font-size:8pt}.heatmap th,.heatmap td{padding:7px 5px}.heatmap th:first-child{min-width:0}svg{max-height:250px}summary{display:none}}
</style>
</head>
<body>
<a class="skip" href="#overview">Skip to report</a>
<nav class="nav" aria-label="Report sections"><div class="nav-inner"><span class="brand">HARNESS BENCH</span><div class="nav-links"><a href="#tasks">Five tasks</a><a href="#method">Method</a><a href="#protocol">Multiagent pilot</a><a href="#validity">Validation</a><a href="#sources">Sources</a></div></div></nav>
<main>
<header class="hero" id="overview"><div class="eyebrow">Research note / proposed experiment</div><h1>Multiagent harness benchmarking:<br>the five-task pilot</h1><p class="lead">Use five tasks selected from past benchmark runs to screen how well a main agent delegates work to subagents. The historical data explain why these probes were selected; the new experiment will measure the team configurations themselves.</p><p class="meta">8 October 2026 · Analysis of saved runs from 16 August–7 September 2026 · New multiagent runs have not been launched.</p></header>
<section aria-labelledby="evidence-heading"><div class="eyebrow">What five tasks reconstruct</div><h2 id="evidence-heading">Three different numbers, three different meanings.</h2><p class="section-intro">All figures below are calculated from the saved 78-task × 9-harness matrix, with DeepSeek, GLM and Muse weighted equally.</p>
<div class="stats"><div class="stat matrix"><strong>__MATRIX_FIVE__</strong><span class="stat-label">Historical matrix reconstruction</span><p>Fit to the existing configurations, using task baselines and reconstruction weights learned from the full historical matrix.</p></div><div class="stat contrasts"><strong>__CONTRAST_FIVE__</strong><span class="stat-label">Historical harness contrasts retained</span><p>Variation between harnesses after removing the shared easy-or-hard effect of each task. This is the more relevant historical measure for comparing harnesses.</p></div><div class="stat prediction"><strong>__PREDICTION_FIVE__</strong><span class="stat-label">Held-out prediction R²</span><p>Accuracy on tasks still unmeasured when a whole harness is held out. Each fold refits its own five-task selection using only the other eight harnesses.</p></div></div>
<div class="status"><strong>The five tasks form a screening pilot.</strong> The 95.3% figure is an in-sample reconstruction score. The minimum task count needed to preserve rankings under the new main-agent/subagent protocol has not yet been established.</div></section>
<section id="tasks" aria-labelledby="tasks-heading"><div class="eyebrow">The proposed workload</div><h2 id="tasks-heading">The first five selected tasks.</h2><p class="section-intro">Use this exact ordered list for every harness and model in the pilot. Each row shows cumulative reconstruction after adding that task to the preceding rows.</p>
<div class="table-wrap"><table><thead><tr><th scope="col">Order</th><th scope="col">Task ID</th><th scope="col">Historical full matrix R²</th><th scope="col">Historical harness contrasts</th></tr></thead><tbody>__TASK_ROWS__</tbody></table></div>
<p class="table-note">Five tasks are 6.4% of the 78-task workload. They were selected for their historical response patterns across harnesses, rather than from measurements of multiagent suitability. An unweighted mean over these five tasks is a pilot score; it is not a calibrated estimate of the full 78-task score.</p>
<div class="figure"><div class="figure-title">Explore the historical task budget</div><div class="legend" aria-label="Chart legend"><span class="matrix"><i></i>Historical full matrix fit</span><span class="contrasts"><i></i>Historical harness contrasts</span><span class="prediction"><i></i>Held-out prediction: unmeasured tasks</span></div>__CHART__
<div class="budget-control"><label for="task-budget">Inspect task count</label><input id="task-budget" type="range" min="1" max="8" step="1" value="5" aria-describedby="budget-caption"><output id="budget-output" for="task-budget">5 of 78 tasks</output></div>
<div class="mini-scores" aria-live="polite" aria-atomic="true"><div class="mini-score"><b id="score-matrix" style="color:var(--blue)">__MATRIX_FIVE__</b><span>Historical full matrix R²</span></div><div class="mini-score"><b id="score-contrasts" style="color:var(--purple)">__CONTRAST_FIVE__</b><span>Historical harness contrasts</span></div><div class="mini-score"><b id="score-prediction" style="color:var(--rust)">__PREDICTION_FIVE__</b><span>Held-out, unmeasured-task R²</span></div></div>
<p id="selected-preview" class="selected-list">__SELECTED_PREVIEW__</p><p id="budget-caption" class="caption">The slider explores the historical analysis; the proposed pilot remains fixed to the first five tasks. The held-out curve scores the training-only selection procedure, with a different subset in each fold. It is not an independent validation of this fixed five-task list.</p>
<div class="figure-actions"><a id="download-chart" href="data:image/png;base64,__CHART_PNG__" download="five-task-pilot-reconstruction.png">Download chart PNG</a><button class="no-print" id="reset-budget" type="button" style="font:inherit;color:var(--blue);background:none;border:0;padding:0;text-decoration:underline;cursor:pointer">Reset to five</button></div>
<noscript><p class="no-js">The five-task results and charts are available without JavaScript. The budget slider and JSON downloads require JavaScript.</p></noscript></div>
<details><summary>The observed completion patterns of the five tasks</summary><div class="detail-content"><p>Cells are equally weighted model averages, rounded to whole percentages here. These are historical observations; explanations for individual successes or failures require trajectory inspection.</p><div class="table-wrap">__HEATMAP__</div></div></details></section>
<section id="method" aria-labelledby="method-heading"><div class="eyebrow">How we arrived here</div><h2 id="method-heading">From past trials to a task subset.</h2>
<div class="steps"><div class="step"><h3>Keep the comparable historical cohorts.</h3><p>3,510 retained trials cover the same 78 non-image Terminal-Bench 2.1 tasks and nine harnesses. Include the corrected GLM/Muse Codex reruns. Exclude the discarded Muse attempt and the invalid original Codex request-failure cohorts.</p></div><div class="step"><h3>Average repetitions, then average models.</h3><p>For each model/task/harness, average binary passes across repetitions: three for DeepSeek, one for GLM and one for Muse. Then give the three model-level rates equal weight. Retained execution errors count as unsuccessful trials, matching the existing benchmark convention.</p></div><div class="step"><h3>Use SVD to measure matrix structure.</h3><p>Tasks are rows, harnesses are columns, and cell values are completion rates. Subtract each harness’s mean before SVD. Four components explain 91.0% of centered variance. Much of that is shared task difficulty; seven components explain 95.3% of the double-centered harness interaction variance.</p></div><div class="step"><h3>Select actual task rows.</h3><p>Remove each task’s mean across harnesses. Greedily add the task whose residual response pattern gives the largest one-step reduction in squared reconstruction error. This yields a nested task ordering. The method is a deterministic heuristic, with no claim of a globally optimal subset.</p></div><div class="step"><h3>Reconstruct with learned baselines and weights.</h3><p>The selected rows predict residual task outcomes through an SVD pseudoinverse; add the historical task baselines back. Five tasks give 95.3% total R² and retain 79.1% of task-centered harness contrasts. The learned baseline alone already gives 77.5% historical R².</p></div><div class="step"><h3>Check an unseen harness column.</h3><p>Leave each of the nine harnesses out in turn. Refit baselines, task selection and weights from the other eight, reveal only the selected held-out task outcomes, and predict the rest. With five tasks, full-matrix R² is 51.3%; unmeasured-only R² is 48.3% and RMSE is 27.9 percentage points.</p></div></div>
<details><summary>Exact matrix and reconstruction definitions</summary><div class="detail-content"><div class="equation">A[t,h] = mean over models m of mean over repeats r of passed[m,t,h,r]<br>X = A − column_mean(A);&nbsp; X = U S Vᵀ<br>SVD variance captured by k components = sum(s[1:k]²) / sum(s²)<br>μ = row_mean(A);&nbsp; B = A − μ<br>Historical reconstruction: Â = μ + B · pinv(B[selected,:]) · B[selected,:]<br>Full-matrix R² = 1 − sum((A − Â)²) / sum((A − column_mean(A))²)</div><p>Harness-contrast retention uses sum(B²) as its denominator. The double-centered SVD uses A − row_mean(A) − column_mean(A) + grand_mean(A), which also removes each harness’s overall mean effect. These are distinct measures. For held-out prediction, learn μ and B only from training harnesses, solve for coefficients from revealed tasks, predict the remaining task rates and clip them to [0,1]. Unmeasured-only R² restricts both numerator and denominator to unrevealed entries, using the same per-harness centering constants as the full score.</p><p>SVD components are linear combinations of task outcomes. The SVD component count and the number of selected tasks are different quantities. The reconstruction equation also reuses information learned from past full-matrix outcomes; five task measurements alone do not establish the quoted historical fit.</p></div></details>
<div class="two-cols"><div class="inset"><h3>What reconstruction accuracy answers</h3><p>How closely the learned model recovers historical cell values in the completion-rate matrix. It measures squared error relative to a specified baseline.</p></div><div class="inset"><h3>What the multiagent study must answer</h3><p>Which harness delivers higher task success with a coordinator and subagents, at a matched total budget; whether the ranking and delegation benefit repeat across models and trials.</p></div></div></section>
<section id="protocol" aria-labelledby="protocol-heading"><div class="eyebrow">What we are using the tasks for</div><h2 id="protocol-heading">A coordinator-and-workers harness pilot.</h2><p class="section-intro">For each harness, model and repetition, one main agent receives the same five-task batch and delegates task work to subagents. Every task has an isolated sandbox and its own unchanged benchmark grader. This design measures orchestration of independent tasks, including delegation overhead and parallel throughput.</p>
<div class="architecture" role="group" aria-label="Proposed main-agent and subagent architecture"><div class="architecture-label">One harness × one fixed model × one repetition</div><div class="coordinator"><strong>Main agent / coordinator</strong><span>Plans the batch, delegates work, collects results and records completion.</span></div><div class="connector" aria-hidden="true">↓</div><div class="worker-grid">__WORKERS__</div><div class="grader">Separate task graders → five verified task outcomes + pooled team cost + batch wall time</div></div>
<div class="table-wrap"><table class="comparison"><thead><tr><th scope="col">Comparison arm</th><th scope="col">What runs</th><th scope="col">Purpose</th></tr></thead><tbody><tr><td>Main agent + subagents</td><td>A coordinator delegates the five tasks under a predeclared worker limit.</td><td>Measure verified completion, coordination failures, cost and parallel batch time.</td></tr><tr><td>Main agent only</td><td>The same harness and model solve the same five-task batch with delegation disabled.</td><td>Measure the change associated with enabling delegation at a matched total budget.</td></tr></tbody></table></div>
<div class="protocol"><div><h3>Hold the comparison conditions fixed.</h3><ul><li>Use the same model for main agents and workers within an arm; keep reasoning settings and tool permissions explicit.</li><li>Set the same total batch token or USD ceiling across arms, counting all coordinator and worker calls. Record actual usage.</li><li>Predeclare the worker cap, task/batch deadlines and per-task CPU/RAM allocation. Use equivalent limits for every harness.</li><li>Keep task sandboxes independent; track resource allocation so simultaneous tasks receive comparable capacity.</li></ul></div><div><h3>Make the measurements repeatable.</h3><ul><li>Verify each harness can create workers under the intended configuration; record versions and unsupported configurations.</li><li>Use paired repeated trials for the two arms. Predeclare task order, seeds where supported, and the trial count before scoring.</li><li>Log actual spawn calls, worker lifetimes and overlap. A configured subagent feature alone does not establish that delegation occurred.</li><li>Report failures of the integration separately from scored task failures while keeping the primary unsuccessful-trial policy explicit.</li></ul></div></div>
<div class="measure-list"><div><h3>Primary: verified task success</h3><p>Grade each task’s final state with its existing verifier. Average repeats within each model/task, then tasks; report per-model scores and an equal-model aggregate.</p></div><div><h3>Secondary: team efficiency</h3><p>Record total coordinator + worker tokens and cost, batch wall time, per-task time and resource allocation. Report success alongside efficiency.</p></div><div><h3>Mechanism: real delegation</h3><p>Record workers spawned, concurrency, coordinator overhead and worker failures. Compare the paired pass-rate difference between the delegation and main-only arms.</p></div></div>
<p class="caption">The architecture shown assigns independent tasks to workers. A later study of cooperation within one task would need its own protocol and evidence. Compute-budget matching and wall-time improvement measure different aspects of performance; report both.</p></section>
<section id="validity" aria-labelledby="validity-heading"><div class="eyebrow">Scientific scope</div><h2 id="validity-heading">Use the pilot to learn; validate the ranking broadly.</h2><div class="decision"><strong>The pilot answers a focused question:</strong> how these five historically informative tasks respond to the proposed delegation protocol. Calling a harness “best” across the full task distribution requires broader multiagent measurements and ranking uncertainty.</div>
<div class="protocol"><div><h3>Evidence already available</h3><ul><li>Exact source/model allowlists, a shared task-set hash and complete task coverage were checked.</li><li>All 27 model/harness aggregate success rates match the retained website snapshot within 10⁻¹².</li><li>The first five task rows reconstruct 95.3% of historical total variance, with 79.1% of harness contrasts retained.</li><li>Historical held-out prediction is weak at this budget. The task-baseline-only unmeasured score is 71.6%, compared with 48.3% for five-task interpolation.</li></ul></div><div><h3>Measurements still needed</h3><ul><li>Run the new main-agent/subagent protocol and its main-only baseline; no new team results are included here.</li><li>Use the full 78-task multiagent workload to calibrate and assess how the five-task ranking compares with the full ranking.</li><li>Evaluate a reduced set on held-out multiagent configurations or models with task selection confined to training data.</li><li>Report paired score differences and their uncertainty. Choose trial counts for the desired resolution using pilot variability.</li></ul></div></div>
<p>Changing the team protocol can change which tasks discriminate between harnesses. The previous 36-task threshold for full historical reconstruction and 43-task threshold for unmeasured prediction belong to that historical procedure; they do not set a validated minimum for the new multiagent experiment. High matrix R² also does not guarantee the same winning harness when score differences are small.</p>
<p>Historical limitations remain: only nine harness columns and three model cohorts; one GLM/Muse trial per task versus three for DeepSeek; differing historical reasoning settings; and two DeepSeek submitted model IDs across source cohorts. A model average can hide model-specific behavior, so preserve those results separately.</p></section>
<section id="sources" aria-labelledby="sources-heading"><div class="eyebrow">Traceability and reuse</div><h2 id="sources-heading">Data, formulas and the proposed plan.</h2><div class="table-wrap"><table class="source-table"><tbody><tr><th scope="row">Saved rollout snapshot</th><td>Retrieved __RETRIEVED__. 3,510 trials: 2,013 passed, 1,369 failed and 128 retained execution errors.</td></tr><tr><th scope="row">Models</th><td>DeepSeek V4 Flash: three repetitions. GLM 5.3 Flash and Muse Spark 1.3 Contributor: one repetition each. Models receive equal weight.</td></tr><tr><th scope="row">Benchmark</th><td>Terminal-Bench 2.1, the same 78 runnable non-image tasks for all 27 retained model/harness combinations.</td></tr><tr><th scope="row">Analysis</th><td><code>scripts/analyze_task_svd.py</code> → <code>outputs/analysis/task-svd-2026-10-08/analysis.json</code></td></tr><tr><th scope="row">Build this report</th><td><code>uv run --group dev python scripts/build_multiagent_pilot_report.py</code></td></tr></tbody></table></div>
<p class="caption">Method references: <a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html">NumPy SVD</a>, <a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html">SVD pseudoinverse</a>, and <a href="https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html">SciPy’s explanation of interpolative matrix decomposition</a>. The numerical results and selection procedure are calculated from this study’s own saved data.</p>
<div class="downloads"><button id="download-plan" class="button" type="button">Download proposed pilot plan</button><button id="download-data" class="button secondary" type="button">Download historical matrix + scores</button></div>
<details><summary>Provenance fingerprints and reproduction checks</summary><div class="detail-content"><p>Rollout snapshot SHA-256:<br><code>__SNAPSHOT_HASH__</code></p><p>Analysis JSON SHA-256:<br><code>__ANALYSIS_HASH__</code></p><p>Task-set SHA-256:<br><code>__TASK_SET_HASH__</code></p><p>Task-level successes for the original six-harness cohort were independently checked against the local read-only results database. The SVD spectrum was checked through eigenvalues of XᵀX. An invariance check confirmed that unmeasured held-out task outcomes do not enter the prediction model. This report reads existing outputs; it performs no benchmark or paid model calls.</p></div></details></section>
<footer class="footer">Harness Bench · Five-task multiagent pilot · Historical evidence and proposed protocol, with their scope stated separately.<br>Portable HTML: charts, explorer and downloadable JSON are embedded; no external scripts, fonts or data requests.</footer>
</main>
<script id="report-data" type="application/json">__DATA__</script>
<script>
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const slider = document.getElementById('task-budget');
  const percent = value => (value * 100).toFixed(1) + '%';
  const x = k => 63 + k * (930 - 63 - 30) / 8;
  const y = value => 20 + (1 - value) * (355 - 20 - 48);
  function render() {
    const k = Number(slider.value), h = data.historical[k], v = data.validation[k];
    document.getElementById('budget-output').textContent = `${k} of 78 tasks`;
    document.getElementById('score-matrix').textContent = percent(h.r2);
    document.getElementById('score-contrasts').textContent = percent(h.harness_contrasts_explained);
    document.getElementById('score-prediction').textContent = percent(v.unmeasured_r2);
    document.getElementById('selected-preview').textContent = data.task_selection.slice(0,k).map(task => task.task_id).join(' · ');
    const focus = document.getElementById('focus-line');
    focus.setAttribute('x1', x(k)); focus.setAttribute('x2', x(k));
    for (const [key, value] of Object.entries({matrix:h.r2, contrasts:h.harness_contrasts_explained, prediction:v.unmeasured_r2})) {
      const circle = document.getElementById('focus-' + key);
      circle.setAttribute('cx', x(k)); circle.setAttribute('cy', y(value));
    }
    document.getElementById('chart-description').textContent = `${k} selected tasks: historical full matrix reconstruction ${percent(h.r2)}, historical harness contrasts ${percent(h.harness_contrasts_explained)}, held-out prediction on unmeasured tasks ${percent(v.unmeasured_r2)}. Historical fits include learned task baselines; the held-out selection is refitted in each fold.`;
  }
  function download(name, object) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(object,null,2) + '\n'], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = name;
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  slider.addEventListener('input', render);
  document.getElementById('reset-budget').addEventListener('click', () => {slider.value=5;render();});
  document.getElementById('download-plan').addEventListener('click', () => download('multiagent-five-task-pilot-plan.json',data.plan));
  document.getElementById('download-data').addEventListener('click', () => download('five-task-pilot-historical-data.json',data.historical_data));
  let beforePrintBudget = 5;
  window.addEventListener('beforeprint', () => {beforePrintBudget=slider.value;slider.value=5;render();});
  window.addEventListener('afterprint', () => {slider.value=beforePrintBudget;render();});
  render();
})();
</script>
</body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.analysis_dir
    analysis_path = directory / "analysis.json"
    analysis_bytes = analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes)
    matrix = json.loads((directory / "completion-matrix.json").read_bytes())
    assert [task["task_id"] for task in analysis["task_selection"][:5]] == EXPECTED_TASKS
    assert analysis["shape"] == {"tasks": 78, "harnesses": 9, "models": 3}
    assert abs(analysis["historical_reconstruction"]["curve"][5]["r2"] - .9530082300840466) < 1e-12
    plan = {
        "name": "Five-task multiagent harness screening pilot",
        "status": "proposed_not_launched",
        "task_ids": EXPECTED_TASKS,
        "benchmark": "terminal-bench/terminal-bench-2-1",
        "purpose": "Compare a main agent delegating independent task work to subagents against a matched main-only arm.",
        "experiment_unit": "harness x model x paired repetition; coordinator receives the same five-task batch",
        "arms": ["main_agent_with_subagents", "main_agent_only"],
        "sandbox_policy": "One isolated sandbox and unchanged benchmark grader per task.",
        "model_policy": "Same model for coordinator/workers within a configuration. Report each model and an equal-model average.",
        "budget_policy": "Predeclare equal total batch token or USD ceilings, including coordinator and all worker calls; report actual consumption.",
        "limits_to_predeclare": ["worker cap", "task and batch deadlines", "CPU/RAM allocation", "reasoning settings", "tool permissions", "paired repetition count", "seeds where supported"],
        "primary_measure": "Task verification pass rate, with repetitions averaged within task/model before aggregation.",
        "secondary_measures": ["paired pass-rate change with delegation", "pooled token usage and USD cost", "batch and per-task wall time", "worker spawn calls", "observed overlap and concurrency", "worker failures", "coordinator overhead"],
        "interpretation": "Independent-task delegation/throughput pilot; historical reconstruction is not validation of new multiagent rankings or within-task cooperation.",
        "follow_up": "Calibrate the new protocol on all 78 tasks and validate reduced-set rankings on held-out configurations/models.",
        "historical_evidence": {
            "five_task_full_matrix_r2": analysis["historical_reconstruction"]["curve"][5]["r2"],
            "five_task_harness_contrasts_retained": analysis["historical_reconstruction"]["curve"][5]["harness_contrasts_explained"],
            "five_task_procedure_held_out_unmeasured_r2": analysis["validation"]["curve"][5]["unmeasured_r2"],
            "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
        },
    }
    historical_data = {
        "provenance": analysis["provenance"], "matrix": matrix,
        "historical_reconstruction": analysis["historical_reconstruction"],
        "validation": analysis["validation"], "task_selection": analysis["task_selection"],
        "limitations": analysis["limitations"],
    }
    payload = {
        "historical": analysis["historical_reconstruction"]["curve"][:9],
        "validation": analysis["validation"]["curve"][:9],
        "task_selection": analysis["task_selection"][:8],
        "plan": plan, "historical_data": historical_data,
    }
    export_chart(directory, analysis)
    workers = "".join(f'<div class="worker"><span class="worker-label">WORKER / TASK 0{i+1}</span><code>{html.escape(task)}</code><small>Isolated sandbox<br>Own task verification</small></div>' for i, task in enumerate(EXPECTED_TASKS))
    replacements = {
        "__MATRIX_FIVE__": pct(plan["historical_evidence"]["five_task_full_matrix_r2"]),
        "__CONTRAST_FIVE__": pct(plan["historical_evidence"]["five_task_harness_contrasts_retained"]),
        "__PREDICTION_FIVE__": pct(plan["historical_evidence"]["five_task_procedure_held_out_unmeasured_r2"]),
        "__TASK_ROWS__": task_table(analysis), "__HEATMAP__": heatmap_table(analysis),
        "__SELECTED_PREVIEW__": html.escape(" · ".join(EXPECTED_TASKS)),
        "__CHART__": chart_svg(analysis),
        "__CHART_PNG__": base64.b64encode((directory / "five-task-pilot-reconstruction.png").read_bytes()).decode(),
        "__WORKERS__": workers,
        "__RETRIEVED__": html.escape(analysis["provenance"]["retrieved_at"]),
        "__SNAPSHOT_HASH__": analysis["provenance"]["snapshot_sha256"],
        "__ANALYSIS_HASH__": hashlib.sha256(analysis_bytes).hexdigest(),
        "__TASK_SET_HASH__": analysis["provenance"]["task_set_sha256"],
        "__DATA__": json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
    }
    report = HTML_TEMPLATE
    for key, value in replacements.items():
        assert key in report
        report = report.replace(key, value)
    output = args.output or directory / "multiagent-five-task-pilot.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    (directory / "multiagent-five-task-pilot-plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"html": str(output.resolve()), "plan": str((directory / 'multiagent-five-task-pilot-plan.json').resolve()),
                      "five_task_scores": plan["historical_evidence"], "created_at": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
