#!/usr/bin/env python3
"""Export the combined cost/success chart data for reuse in a browser.

The original six harnesses come from the historical Dagster rollout. The
three newer harnesses come from their later benchmark campaign. For both
cohorts, repeated rollouts are averaged within each task before harness-level
means and confidence intervals are calculated.

The exporter emits JSON by default, flat CSV, or a browser-ready ES module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sqlite3
import statistics
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "outputs" / "results.sqlite"
DEFAULT_OLD_RUN_ID = "8d604156"
DEFAULT_NEW_CAMPAIGN_ID = "terminal-bench-new-harnesses-2026-08-24-a"
DEFAULT_REMOTE_CONFIG = REPO_ROOT / ".dagster" / "remote.json"
DEFAULT_REMOTE_DB = "outputs/results.sqlite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"
T_CRITICAL_95 = 1.991

NEW_HARNESS_IDS = {"deepseek_harness", "pi_agent", "pi_rlm_runtime"}

HARNESS_PRESENTATION = {
    "claude_code_agent": {"label": "Claude Code", "logo": "anthropic", "order": 0},
    "codex_agent": {"label": "Codex", "logo": "openai", "order": 1},
    "deepseek_harness": {"label": "DeepSeek Harness", "logo": "deepseek", "order": 2},
    "hermes_agent": {"label": "Hermes", "logo": "nous", "order": 3},
    "omp_agent": {"label": "OMP", "logo": "omp", "order": 4},
    "opencode": {"label": "OpenCode", "logo": "opencode", "order": 5},
    "pi_agent": {"label": "Pi Agent", "logo": "pi", "order": 6},
    "pi_rlm_runtime": {"label": "Pi RLM", "logo": "pi", "order": 7},
    "prime_agent": {"label": "PrimeAgent", "logo": "prime-intellect", "order": 8},
}

TASK_MEANS_SQL = """
    SELECT harness, harness_version, dataset, model, task_id,
           AVG(cost_usd) AS task_cost_usd,
           AVG(passed) AS task_success_rate,
           GROUP_CONCAT(DISTINCT dagster_run_id) AS source_run_ids
    FROM rollout_results
    WHERE {source_column} = ?
    GROUP BY harness, harness_version, dataset, model, task_id
    ORDER BY harness, task_id
"""

REMOTE_QUERY_CODE = r"""
import json
import sqlite3
import sys

database, campaign_id = sys.argv[1:3]
connection = sqlite3.connect(database)
connection.row_factory = sqlite3.Row
rows = [
    dict(row)
    for row in connection.execute(
        '''
        SELECT harness, harness_version, dataset, model, task_id,
               AVG(cost_usd) AS task_cost_usd,
               AVG(passed) AS task_success_rate,
               GROUP_CONCAT(DISTINCT dagster_run_id) AS source_run_ids
        FROM rollout_results
        WHERE campaign_id = ?
        GROUP BY harness, harness_version, dataset, model, task_id
        ORDER BY harness, task_id
        ''',
        (campaign_id,),
    )
]
campaign_row = connection.execute(
    '''
    SELECT campaign_id, planner_run_id, state, task_count, combination_count,
           task_execution_count, dataset
    FROM benchmark_campaigns
    WHERE campaign_id = ?
    ''',
    (campaign_id,),
).fetchone()
connection.close()
print(json.dumps({
    'rows': rows,
    'campaign': dict(campaign_row) if campaign_row else None,
}))
"""


def resolve_run_id(connection: sqlite3.Connection, run_id: str) -> str:
    matches = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT dagster_run_id FROM rollout_results "
            "WHERE dagster_run_id LIKE ? ORDER BY dagster_run_id",
            (f"{run_id}%",),
        )
    ]
    if not matches:
        raise SystemExit(f"No rollout found with Dagster run ID prefix {run_id!r}")
    if len(matches) > 1:
        joined = ", ".join(matches)
        raise SystemExit(f"Run ID prefix {run_id!r} is ambiguous: {joined}")
    return matches[0]


def query_task_means(
    connection: sqlite3.Connection,
    source_column: str,
    source_id: str,
) -> list[dict[str, Any]]:
    if source_column not in {"dagster_run_id", "campaign_id"}:
        raise ValueError(f"Unsupported source column: {source_column}")
    sql = TASK_MEANS_SQL.format(source_column=source_column)
    return [dict(row) for row in connection.execute(sql, (source_id,))]


def load_old_rollout(db_path: Path, run_id_prefix: str) -> tuple[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        run_id = resolve_run_id(connection, run_id_prefix)
        rows = query_task_means(connection, "dagster_run_id", run_id)
    finally:
        connection.close()
    if not rows:
        raise SystemExit(f"Run {run_id!r} contains no rollout rows")
    return run_id, rows


def load_campaign_from_db(
    db_path: Path,
    campaign_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = query_task_means(connection, "campaign_id", campaign_id)
        campaign_row = connection.execute(
            """
            SELECT campaign_id, planner_run_id, state, task_count,
                   combination_count, task_execution_count, dataset
            FROM benchmark_campaigns
            WHERE campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()
    finally:
        connection.close()
    return rows, dict(campaign_row) if campaign_row else None


def load_campaign_from_remote(
    remote_config_path: Path,
    remote_db: str,
    campaign_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    remote_config = json.loads(remote_config_path.read_text(encoding="utf-8"))
    host = remote_config["host"]
    project_dir = remote_config["project_dir"]
    remote_command = (
        f"cd {shlex.quote(project_dir)} && "
        f"python3 -c {shlex.quote(REMOTE_QUERY_CODE)} "
        f"{shlex.quote(remote_db)} {shlex.quote(campaign_id)}"
    )
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, remote_command],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise SystemExit(f"Could not read newer campaign from {host}: {detail}") from error
    payload = json.loads(completed.stdout)
    return payload["rows"], payload["campaign"]


def load_new_campaign(
    primary_db: Path,
    new_db: Path | None,
    campaign_id: str,
    remote_config: Path,
    remote_db: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    candidate_db = new_db or primary_db
    rows, campaign = load_campaign_from_db(candidate_db, campaign_id)
    if rows:
        return rows, campaign, str(candidate_db)
    rows, campaign = load_campaign_from_remote(remote_config, remote_db, campaign_id)
    if not rows:
        raise SystemExit(f"Campaign {campaign_id!r} contains no rollout rows")
    return rows, campaign, f"remote:{remote_config}"


def ci_half_width(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return T_CRITICAL_95 * statistics.stdev(values) / math.sqrt(len(values))


def single_value(values: set[str], description: str) -> str:
    if len(values) != 1:
        joined = ", ".join(sorted(values)) or "none"
        raise SystemExit(f"Expected one {description}, found: {joined}")
    return next(iter(values))


def split_run_ids(value: str | None) -> set[str]:
    return {item for item in (value or "").split(",") if item}


def build_payload(
    db_path: Path,
    old_run_id_prefix: str,
    new_db: Path | None,
    new_campaign_id: str,
    remote_config: Path,
    remote_db: str,
) -> dict[str, Any]:
    old_run_id, old_rows = load_old_rollout(db_path, old_run_id_prefix)
    new_rows, campaign, _new_source_location = load_new_campaign(
        db_path,
        new_db,
        new_campaign_id,
        remote_config,
        remote_db,
    )
    old_rows = [row for row in old_rows if row["harness"] not in NEW_HARNESS_IDS]
    new_rows = [row for row in new_rows if row["harness"] in NEW_HARNESS_IDS]
    observed_new_ids = {row["harness"] for row in new_rows}
    if observed_new_ids != NEW_HARNESS_IDS:
        missing = ", ".join(sorted(NEW_HARNESS_IDS - observed_new_ids))
        raise SystemExit(f"New campaign is missing expected harness data: {missing}")

    for row in old_rows:
        row["source_kind"] = "dagster_run"
        row["source_id"] = old_run_id
    for row in new_rows:
        row["source_kind"] = "campaign"
        row["source_id"] = new_campaign_id
    rows = old_rows + new_rows

    dataset = single_value({row["dataset"] for row in rows}, "dataset")
    harness_ids = sorted(
        {row["harness"] for row in rows},
        key=lambda item: HARNESS_PRESENTATION.get(item, {}).get("order", 999),
    )

    harnesses = []
    for harness_id in harness_ids:
        harness_rows = [row for row in rows if row["harness"] == harness_id]
        version = single_value(
            {row["harness_version"] for row in harness_rows},
            f"version for {harness_id}",
        )
        source_kind = single_value(
            {row["source_kind"] for row in harness_rows},
            f"source kind for {harness_id}",
        )
        source_id = single_value(
            {row["source_id"] for row in harness_rows},
            f"source ID for {harness_id}",
        )
        source_run_ids: set[str] = set()
        for row in harness_rows:
            source_run_ids.update(split_run_ids(row["source_run_ids"]))
        costs = [float(row["task_cost_usd"]) for row in harness_rows if row["task_cost_usd"] is not None]
        successes = [
            float(row["task_success_rate"])
            for row in harness_rows
            if row["task_success_rate"] is not None
        ]
        presentation = HARNESS_PRESENTATION.get(
            harness_id,
            {"label": harness_id, "logo": harness_id},
        )
        harnesses.append(
            {
                "id": harness_id,
                "label": presentation["label"],
                "version": version,
                "logo": presentation["logo"],
                "source": {
                    "kind": source_kind,
                    "id": source_id,
                    "dagster_run_ids": sorted(source_run_ids),
                },
                "mean_cost_usd_per_task": sum(costs) / len(costs),
                "cost_95ci_half_width": ci_half_width(costs),
                "mean_success_rate_per_task": sum(successes) / len(successes),
                "success_95ci_half_width": ci_half_width(successes),
                "tasks": len(harness_rows),
                "tasks_with_cost": len(costs),
                "tasks_with_success": len(successes),
            }
        )

    old_harnesses = sorted({row["harness"] for row in old_rows})
    new_harnesses = sorted({row["harness"] for row in new_rows})
    old_model = single_value({row["model"] for row in old_rows}, "old rollout model")
    new_model = single_value({row["model"] for row in new_rows}, "new campaign model")
    new_run_ids = sorted(
        {
            run_id
            for row in new_rows
            for run_id in split_run_ids(row["source_run_ids"])
        }
    )

    return {
        "schema_version": 2,
        "benchmark": {
            "id": dataset,
            "label": "Terminal-Bench 2.1",
        },
        "sources": [
            {
                "kind": "dagster_run",
                "id": old_run_id,
                "model": old_model,
                "harnesses": old_harnesses,
            },
            {
                "kind": "campaign",
                "id": new_campaign_id,
                "planner_run_id": campaign.get("planner_run_id") if campaign else None,
                "campaign_state": campaign.get("state") if campaign else None,
                "model": new_model,
                "harnesses": new_harnesses,
                "dagster_run_ids": new_run_ids,
                "task_count": campaign.get("task_count") if campaign else None,
                "combination_count": campaign.get("combination_count") if campaign else None,
                "task_execution_count": campaign.get("task_execution_count") if campaign else None,
            },
        ],
        "aggregation": "mean of task-level means across repeated rollouts",
        "confidence_interval": "95% CI half-width using t critical 1.991",
        "presentation": {
            "model": "deepseek-v4-flash-0731",
            "reasoning_label": "High reasoning",
            "x": {"field": "mean_cost_usd_per_task", "label": "Mean cost per task", "unit": "USD"},
            "y": {"field": "mean_success_rate_per_task", "label": "Mean task success rate", "domain": [0, 1]},
        },
        "harnesses": harnesses,
    }


def write_json(payload: dict[str, Any], output: Path) -> None:
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_js(payload: dict[str, Any], output: Path) -> None:
    serialized = json.dumps(payload, indent=2)
    output.write_text(
        f"export const costSuccessChartData = {serialized};\n\n"
        "export default costSuccessChartData;\n",
        encoding="utf-8",
    )


def write_csv(payload: dict[str, Any], output: Path) -> None:
    fieldnames = [
        "benchmark",
        "model",
        "reasoning_label",
        "source_kind",
        "source_id",
        "source_run_ids",
        "id",
        "label",
        "version",
        "logo",
        "mean_cost_usd_per_task",
        "cost_95ci_half_width",
        "mean_success_rate_per_task",
        "success_95ci_half_width",
        "tasks",
        "tasks_with_cost",
        "tasks_with_success",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for harness in payload["harnesses"]:
            source = harness["source"]
            writer.writerow(
                {
                    "benchmark": payload["benchmark"]["label"],
                    "model": payload["presentation"]["model"],
                    "reasoning_label": payload["presentation"]["reasoning_label"],
                    "source_kind": source["kind"],
                    "source_id": source["id"],
                    "source_run_ids": ",".join(source["dagster_run_ids"]),
                    **{key: value for key, value in harness.items() if key != "source"},
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database containing the old rollout")
    parser.add_argument("--old-run-id", default=DEFAULT_OLD_RUN_ID, help="old Dagster run ID or unique prefix")
    parser.add_argument("--new-db", type=Path, help="optional local SQLite database containing the newer campaign")
    parser.add_argument("--new-campaign-id", default=DEFAULT_NEW_CAMPAIGN_ID)
    parser.add_argument("--remote-config", type=Path, default=DEFAULT_REMOTE_CONFIG)
    parser.add_argument("--remote-db", default=DEFAULT_REMOTE_DB)
    parser.add_argument("--format", choices=("json", "csv", "js"), default="json")
    parser.add_argument("--output", type=Path, help="output path; defaults under outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUT_DIR / f"cost-success-chart-data.{args.format}"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        args.db,
        args.old_run_id,
        args.new_db,
        args.new_campaign_id,
        args.remote_config,
        args.remote_db,
    )
    {"json": write_json, "csv": write_csv, "js": write_js}[args.format](payload, output)
    print(output)


if __name__ == "__main__":
    main()
