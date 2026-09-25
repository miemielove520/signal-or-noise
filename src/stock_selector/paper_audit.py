from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .json_io import json_safe, write_json


RUN_COLUMNS = [
    "run_id",
    "started_at",
    "finished_at",
    "status",
    "state_updated",
    "as_of_date",
    "analysis_row_count",
    "target_count",
    "order_count",
    "pre_trade_value",
    "post_trade_value",
    "post_trade_cash",
    "cash_weight",
    "total_spread_cost",
    "total_slippage_cost",
    "total_commission_cost",
    "total_transaction_cost",
    "unpriced_positions",
    "scan_csv",
    "state_csv",
    "output_dir",
    "snapshot_dir",
    "git_commit",
    "git_dirty",
    "error",
]

ORDER_COLUMNS = [
    "run_id",
    "started_at",
    "finished_at",
    "as_of_date",
    "state_updated",
    "order_sequence",
    "ticker",
    "action",
    "quantity",
    "signed_quantity",
    "close_price",
    "fill_price",
    "notional",
    "spread_cost",
    "slippage_cost",
    "commission_cost",
    "estimated_cost",
    "total_transaction_cost",
    "target_weight",
    "current_weight",
    "target_value",
    "current_value",
    "position_quantity_before",
    "position_quantity_after",
    "cash_before",
    "cash_after",
]

POSITION_COLUMNS = [
    "run_id",
    "finished_at",
    "as_of_date",
    "state_updated",
    "stage",
    "ticker",
    "quantity",
    "mark_price",
    "market_value",
    "portfolio_weight",
    "price_source",
]

VALUATION_COLUMNS = [
    "valuation_id",
    "executed_at",
    "date",
    "status",
    "equity",
    "benchmark_ticker",
    "benchmark_close",
    "current_price_count",
    "fallback_price_count",
    "missing_price_count",
    "warning_count",
    "total_return",
    "benchmark_return",
    "excess_return",
    "max_drawdown",
    "readiness",
    "backtest_gate",
    "snapshot_dir",
    "error",
]


def new_audit_identity(prefix: str, now: datetime | None = None) -> tuple[str, str]:
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    instant = instant.astimezone(timezone.utc)
    timestamp = instant.isoformat().replace("+00:00", "Z")
    run_id = f"{prefix}_{instant.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    return run_id, timestamp


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_rebalance_success(
    history_dir: str | Path,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    state_updated: bool,
    analysis: pd.DataFrame,
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    orders: pd.DataFrame,
    pre_trade_state: pd.DataFrame,
    post_trade_state: pd.DataFrame,
    result_payload: dict[str, Any],
    report: str,
    config_snapshot: dict[str, Any],
    source_snapshot: dict[str, Any],
    project_root: str | Path,
    status: str = "success",
) -> Path:
    root = Path(history_dir)
    run_dir = _new_snapshot_dir(root / "runs", run_id)
    as_of_date = _date_text(result_payload.get("as_of_date"))

    snapshots = {
        "analysis_input.csv": analysis,
        "prices_used.csv": prices,
        "paper_targets.csv": targets,
        "orders.csv": orders,
        "pre_trade_state.csv": pre_trade_state,
        "post_trade_state.csv": post_trade_state,
    }
    for name, frame in snapshots.items():
        frame.to_csv(run_dir / name, index=False)

    price_positions = _valued_positions(
        pre_trade_state,
        prices,
        run_id=run_id,
        finished_at=finished_at,
        as_of_date=as_of_date,
        state_updated=state_updated,
        stage="pre_trade",
    )
    post_positions = _valued_positions(
        post_trade_state,
        prices,
        run_id=run_id,
        finished_at=finished_at,
        as_of_date=as_of_date,
        state_updated=state_updated,
        stage="post_trade",
    )
    price_positions.to_csv(run_dir / "pre_trade_positions_valued.csv", index=False)
    post_positions.to_csv(run_dir / "post_trade_positions_valued.csv", index=False)
    (run_dir / "paper_trade_report.md").write_text(report, encoding="utf-8")
    write_json(run_dir / "paper_trade_result.json", result_payload)

    git_metadata = _git_metadata(Path(project_root))
    summary = result_payload.get("summary") or {}
    manifest = {
        "schema_version": 1,
        "record_type": "paper_rebalance",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "state_updated": bool(state_updated),
        "as_of_date": as_of_date,
        "config": config_snapshot,
        "sources": source_snapshot,
        "code": git_metadata,
        "counts": {
            "analysis_rows": int(len(analysis)),
            "targets": int(len(targets)),
            "orders": int(len(orders)),
        },
        "summary": summary,
        "files": _snapshot_hashes(run_dir),
    }
    write_json(run_dir / "run_manifest.json", manifest)

    run_row = _run_row(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        state_updated=state_updated,
        as_of_date=as_of_date,
        analysis_count=len(analysis),
        target_count=len(targets),
        order_count=len(orders),
        summary=summary,
        source_snapshot=source_snapshot,
        snapshot_dir=run_dir,
        git_metadata=git_metadata,
        error=None,
    )
    _append_records(root / "paper_runs.csv", pd.DataFrame([run_row]), RUN_COLUMNS)
    _append_jsonl(root / "paper_runs.jsonl", run_row)

    order_ledger = _order_ledger_rows(
        orders,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        as_of_date=as_of_date,
        state_updated=state_updated,
    )
    _append_records(root / "paper_orders.csv", order_ledger, ORDER_COLUMNS)
    _append_records(
        root / "paper_positions.csv",
        pd.concat([price_positions, post_positions], ignore_index=True),
        POSITION_COLUMNS,
    )
    render_paper_journal(root)
    return run_dir


def record_rebalance_failure(
    history_dir: str | Path,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    state_updated: bool,
    error: str,
    config_snapshot: dict[str, Any],
    source_snapshot: dict[str, Any],
    project_root: str | Path,
    analysis: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
) -> Path:
    root = Path(history_dir)
    run_dir = _new_snapshot_dir(root / "runs", run_id)
    for name, frame in [
        ("analysis_input.csv", analysis),
        ("prices_used.csv", prices),
        ("paper_targets.csv", targets),
    ]:
        if frame is not None:
            frame.to_csv(run_dir / name, index=False)

    git_metadata = _git_metadata(Path(project_root))
    manifest = {
        "schema_version": 1,
        "record_type": "paper_rebalance",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "failed",
        "state_updated": False,
        "error": str(error),
        "config": config_snapshot,
        "sources": source_snapshot,
        "code": git_metadata,
        "files": _snapshot_hashes(run_dir),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    run_row = _run_row(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status="failed",
        state_updated=False,
        as_of_date="",
        analysis_count=len(analysis) if analysis is not None else 0,
        target_count=len(targets) if targets is not None else 0,
        order_count=0,
        summary={},
        source_snapshot=source_snapshot,
        snapshot_dir=run_dir,
        git_metadata=git_metadata,
        error=str(error),
    )
    _append_records(root / "paper_runs.csv", pd.DataFrame([run_row]), RUN_COLUMNS)
    _append_jsonl(root / "paper_runs.jsonl", run_row)
    render_paper_journal(root)
    return run_dir


def record_valuation(
    history_dir: str | Path,
    *,
    valuation_id: str,
    executed_at: str,
    date: str,
    status: str,
    state: pd.DataFrame,
    current_prices: dict[str, float],
    fallback_prices: dict[str, float],
    benchmark_ticker: str,
    benchmark_close: float | None,
    warnings: list[str],
    performance: dict[str, Any] | None = None,
    backtest_gate: dict[str, Any] | None = None,
    equity: float | None = None,
    error: str | None = None,
) -> Path:
    root = Path(history_dir)
    snapshot_dir = _new_snapshot_dir(root / "valuations", valuation_id)
    valuation_prices = _valuation_positions(state, current_prices, fallback_prices)
    state.to_csv(snapshot_dir / "paper_state.csv", index=False)
    valuation_prices.to_csv(snapshot_dir / "valuation_prices.csv", index=False)
    payload = {
        "schema_version": 1,
        "record_type": "paper_valuation",
        "valuation_id": valuation_id,
        "executed_at": executed_at,
        "date": date,
        "status": status,
        "equity": equity,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_close": benchmark_close,
        "warnings": warnings,
        "error": error,
        "performance": performance or {},
        "backtest_gate": backtest_gate or {},
        "files": _snapshot_hashes(snapshot_dir),
    }
    write_json(snapshot_dir / "valuation_manifest.json", payload)

    statuses = valuation_prices.get("price_status", pd.Series(dtype="string"))
    perf = performance or {}
    gate = backtest_gate or {}
    row = {
        "valuation_id": valuation_id,
        "executed_at": executed_at,
        "date": date,
        "status": status,
        "equity": equity,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_close": benchmark_close,
        "current_price_count": int((statuses == "current").sum()),
        "fallback_price_count": int((statuses == "fallback").sum()),
        "missing_price_count": int((statuses == "missing").sum()),
        "warning_count": len(warnings),
        "total_return": perf.get("total_return"),
        "benchmark_return": perf.get("benchmark_return"),
        "excess_return": perf.get("excess_return"),
        "max_drawdown": perf.get("max_drawdown"),
        "readiness": perf.get("readiness"),
        "backtest_gate": gate.get("gate"),
        "snapshot_dir": str(snapshot_dir),
        "error": error or "",
    }
    _append_records(root / "paper_valuations.csv", pd.DataFrame([row]), VALUATION_COLUMNS)
    _append_jsonl(root / "paper_valuations.jsonl", row)
    render_paper_journal(root)
    return snapshot_dir


def render_paper_journal(history_dir: str | Path) -> Path:
    root = Path(history_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs = _read_csv(root / "paper_runs.csv")
    valuations = _read_csv(root / "paper_valuations.csv")
    lines = [
        "# Paper Trading Audit Journal / 模拟盘完整审计日志",
        "",
        "Every rebalance and valuation has an immutable snapshot folder.",
        "每次再平衡和估值都保存独立快照，后续运行不会覆盖。",
        "",
        "## Rebalance Runs / 再平衡运行",
        "",
    ]
    run_columns = [
        "run_id",
        "finished_at",
        "status",
        "state_updated",
        "as_of_date",
        "order_count",
        "post_trade_value",
        "post_trade_cash",
        "total_transaction_cost",
        "error",
    ]
    lines.extend(_markdown_table(runs, run_columns))
    lines.extend(["", "## Valuation Runs / 估值运行", ""])
    valuation_columns = [
        "valuation_id",
        "executed_at",
        "date",
        "status",
        "equity",
        "benchmark_close",
        "fallback_price_count",
        "warning_count",
        "error",
    ]
    lines.extend(_markdown_table(valuations, valuation_columns))
    output = root / "paper_journal.md"
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def _run_row(
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    status: str,
    state_updated: bool,
    as_of_date: str,
    analysis_count: int,
    target_count: int,
    order_count: int,
    summary: dict[str, Any],
    source_snapshot: dict[str, Any],
    snapshot_dir: Path,
    git_metadata: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "state_updated": bool(state_updated),
        "as_of_date": as_of_date,
        "analysis_row_count": int(analysis_count),
        "target_count": int(target_count),
        "order_count": int(order_count),
        "pre_trade_value": summary.get("pre_trade_value"),
        "post_trade_value": summary.get("post_trade_value"),
        "post_trade_cash": summary.get("post_trade_cash"),
        "cash_weight": summary.get("cash_weight"),
        "total_spread_cost": summary.get("total_spread_cost"),
        "total_slippage_cost": summary.get("total_slippage_cost"),
        "total_commission_cost": summary.get("total_commission_cost"),
        "total_transaction_cost": summary.get("total_transaction_cost"),
        "unpriced_positions": summary.get("unpriced_positions", ""),
        "scan_csv": source_snapshot.get("scan_csv", ""),
        "state_csv": source_snapshot.get("state_csv", ""),
        "output_dir": source_snapshot.get("output_dir", ""),
        "snapshot_dir": str(snapshot_dir),
        "git_commit": git_metadata.get("commit", "unknown"),
        "git_dirty": git_metadata.get("dirty"),
        "error": error or "",
    }


def _order_ledger_rows(
    orders: pd.DataFrame,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    as_of_date: str,
    state_updated: bool,
) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    rows: list[dict[str, Any]] = []
    for sequence, order in enumerate(orders.to_dict(orient="records"), start=1):
        rows.append(
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "as_of_date": as_of_date,
                "state_updated": bool(state_updated),
                "order_sequence": sequence,
                "ticker": order.get("ticker"),
                "action": order.get("action"),
                "quantity": order.get("quantity"),
                "signed_quantity": order.get("signed_quantity"),
                "close_price": order.get("close_price"),
                "fill_price": order.get("price"),
                "notional": order.get("notional"),
                "spread_cost": order.get("spread_cost"),
                "slippage_cost": order.get("slippage_cost"),
                "commission_cost": order.get("commission_cost"),
                "estimated_cost": order.get("estimated_cost"),
                "total_transaction_cost": order.get("total_transaction_cost"),
                "target_weight": order.get("target_weight"),
                "current_weight": order.get("current_weight"),
                "target_value": order.get("target_value"),
                "current_value": order.get("current_value"),
                "position_quantity_before": order.get("position_quantity_before"),
                "position_quantity_after": order.get("position_quantity_after"),
                "cash_before": order.get("cash_before"),
                "cash_after": order.get("cash_after"),
            }
        )
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def _valued_positions(
    state: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    run_id: str,
    finished_at: str,
    as_of_date: str,
    state_updated: bool,
    stage: str,
) -> pd.DataFrame:
    if state is None or state.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    price_map, source_map = _price_maps(prices)
    provisional: list[dict[str, Any]] = []
    for row in state.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").upper().strip()
        quantity = _float_or_none(row.get("quantity"))
        if not ticker or quantity is None:
            continue
        if ticker == "CASH":
            mark_price = 1.0
            price_source = "cash"
        else:
            mark_price = price_map.get(ticker)
            price_source = source_map.get(ticker, "missing")
        market_value = quantity * mark_price if mark_price is not None else None
        provisional.append(
            {
                "run_id": run_id,
                "finished_at": finished_at,
                "as_of_date": as_of_date,
                "state_updated": bool(state_updated),
                "stage": stage,
                "ticker": ticker,
                "quantity": quantity,
                "mark_price": mark_price,
                "market_value": market_value,
                "portfolio_weight": None,
                "price_source": price_source,
            }
        )
    total_value = sum(
        float(row["market_value"])
        for row in provisional
        if row["market_value"] is not None and np.isfinite(float(row["market_value"]))
    )
    for row in provisional:
        if row["market_value"] is not None and total_value > 0:
            row["portfolio_weight"] = float(row["market_value"]) / total_value
    return pd.DataFrame(provisional, columns=POSITION_COLUMNS)


def _valuation_positions(
    state: pd.DataFrame,
    current_prices: dict[str, float],
    fallback_prices: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in state.to_dict(orient="records"):
        ticker = str(item.get("ticker") or "").upper().strip()
        quantity = _float_or_none(item.get("quantity"))
        if not ticker or quantity is None:
            continue
        current = _float_or_none(current_prices.get(ticker))
        fallback = _float_or_none(fallback_prices.get(ticker))
        if ticker == "CASH":
            used = 1.0
            status = "cash"
        elif current is not None and current > 0:
            used = current
            status = "current"
        elif fallback is not None and fallback > 0:
            used = fallback
            status = "fallback"
        else:
            used = None
            status = "missing"
        rows.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "current_price": current,
                "fallback_price": fallback,
                "used_price": used,
                "price_status": status,
                "market_value": quantity * used if used is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _price_maps(prices: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    if prices is None or prices.empty:
        return {}, {}
    frame = prices.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    frame = frame.drop_duplicates(subset=["ticker"], keep="last")
    price_map: dict[str, float] = {}
    source_map: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        ticker = str(row.get("ticker") or "").upper().strip()
        price = _float_or_none(row.get("adj_close"))
        if ticker and price is not None and price > 0:
            price_map[ticker] = price
            source_map[ticker] = str(row.get("price_source") or "paper_price_input")
    return price_map, source_map


def _new_snapshot_dir(parent: Path, identifier: str) -> Path:
    path = parent / identifier
    path.mkdir(parents=True, exist_ok=False)
    return path


def _append_records(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reindex(columns=columns)
    output.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def _snapshot_hashes(snapshot_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(snapshot_dir.iterdir()):
        if path.is_file() and path.name not in {"run_manifest.json", "valuation_manifest.json"}:
            hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    return hashes


def _git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame is None or frame.empty:
        return ["No records / 暂无记录。"]
    available = [column for column in columns if column in frame.columns]
    display = frame[available].tail(100).fillna("")
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_display(value) for value in row) + " |")
    return lines


def _display(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _date_text(value: object) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value or "")
