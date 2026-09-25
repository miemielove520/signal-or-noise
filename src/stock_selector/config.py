from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DataConfig:
    prices_csv: Path
    fundamentals_csv: Path | None = None
    macro_csv: Path | None = None
    metadata_csv: Path | None = None
    date_column: str = "date"
    ticker_column: str = "ticker"
    price_column: str = "adj_close"
    volume_column: str = "volume"


@dataclass(frozen=True)
class UniverseConfig:
    min_history_days: int = 252
    min_avg_dollar_volume: float = 5_000_000.0


@dataclass(frozen=True)
class FactorConfig:
    momentum_windows: tuple[int, ...] = (20, 60, 126)
    volatility_window: int = 20
    trend_window: int = 50
    liquidity_window: int = 20
    money_flow_window: int = 20


@dataclass(frozen=True)
class ScoringConfig:
    top_n: int = 20
    rebalance_frequency: str = "M"
    factor_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskConfig:
    weighting_method: str = "equal"
    volatility_lookback_days: int = 60
    target_annual_volatility: float | None = None
    max_position_weight: float = 1.0
    max_sector_weight: float | None = None
    sector_column: str = "sector"
    min_position_weight: float = 0.0
    annualization_days: int = 252


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    annualization_days: int = 252
    risk_free_rate: float = 0.0
    transaction_cost_bps: float = 5.0
    execution_lag_days: int = 1


@dataclass(frozen=True)
class MLConfig:
    feature_columns: tuple[str, ...] = ()
    label_forward_days: int = 20
    train_window_days: int = 252
    min_train_rows: int = 500
    min_prediction_assets: int = 5
    model_type: str = "ridge"
    random_state: int = 42
    ridge_alpha: float = 1.0
    n_estimators: int = 200


@dataclass(frozen=True)
class PaperTradingConfig:
    state_csv: Path | None = None
    initial_cash: float = 100_000.0
    min_trade_value: float = 100.0
    trade_buffer_pct: float = 0.001
    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    # Half bid-ask spread applied to the fill: buys fill above the close, sells
    # below it. Makes paper fills realistic instead of assuming the mid/close.
    spread_bps: float = 10.0
    allow_fractional_shares: bool = True


@dataclass(frozen=True)
class MonitorConfig:
    max_data_age_days: int = 7
    max_average_feature_missing_rate: float = 0.10
    max_single_feature_missing_rate: float = 0.25
    drift_lookback_days: int = 60
    max_feature_drift_zscore: float = 3.0
    min_candidate_overlap: float = 0.50
    max_score_concentration: float = 0.80
    min_rank_ic_mean: float = 0.0


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig
    universe: UniverseConfig
    factors: FactorConfig
    scoring: ScoringConfig
    risk: RiskConfig
    backtest: BacktestConfig
    ml: MLConfig
    paper: PaperTradingConfig
    monitor: MonitorConfig


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent

    data_raw = raw.get("data", {})
    prices_csv = _resolve_optional_path(data_raw.get("prices_csv", "data/prices.csv"), base_dir)
    if prices_csv is None:
        raise ValueError("data.prices_csv is required.")

    data = DataConfig(
        prices_csv=prices_csv,
        fundamentals_csv=_resolve_optional_path(data_raw.get("fundamentals_csv"), base_dir),
        macro_csv=_resolve_optional_path(data_raw.get("macro_csv"), base_dir),
        metadata_csv=_resolve_optional_path(data_raw.get("metadata_csv"), base_dir),
        date_column=data_raw.get("date_column", "date"),
        ticker_column=data_raw.get("ticker_column", "ticker"),
        price_column=data_raw.get("price_column", "adj_close"),
        volume_column=data_raw.get("volume_column", "volume"),
    )
    universe_raw = raw.get("universe", {})
    factors_raw = raw.get("factors", {})
    scoring_raw = raw.get("scoring", {})
    risk_raw = raw.get("risk", {})
    backtest_raw = raw.get("backtest", {})
    ml_raw = raw.get("ml", {})
    paper_raw = raw.get("paper", {})
    monitor_raw = raw.get("monitor", {})

    return ResearchConfig(
        data=data,
        universe=UniverseConfig(
            min_history_days=int(universe_raw.get("min_history_days", 252)),
            min_avg_dollar_volume=float(
                universe_raw.get("min_avg_dollar_volume", 5_000_000.0)
            ),
        ),
        factors=FactorConfig(
            momentum_windows=tuple(
                int(window) for window in factors_raw.get("momentum_windows", [20, 60, 126])
            ),
            volatility_window=int(factors_raw.get("volatility_window", 20)),
            trend_window=int(factors_raw.get("trend_window", 50)),
            liquidity_window=int(factors_raw.get("liquidity_window", 20)),
            money_flow_window=int(factors_raw.get("money_flow_window", 20)),
        ),
        scoring=ScoringConfig(
            top_n=int(scoring_raw.get("top_n", 20)),
            rebalance_frequency=scoring_raw.get("rebalance_frequency", "M"),
            factor_weights={
                str(key): float(value)
                for key, value in scoring_raw.get("factor_weights", {}).items()
            },
        ),
        risk=RiskConfig(
            weighting_method=str(risk_raw.get("weighting_method", "equal")),
            volatility_lookback_days=int(risk_raw.get("volatility_lookback_days", 60)),
            target_annual_volatility=_optional_float(
                risk_raw.get("target_annual_volatility")
            ),
            max_position_weight=float(risk_raw.get("max_position_weight", 1.0)),
            max_sector_weight=_optional_float(risk_raw.get("max_sector_weight")),
            sector_column=str(risk_raw.get("sector_column", "sector")),
            min_position_weight=float(risk_raw.get("min_position_weight", 0.0)),
            annualization_days=int(risk_raw.get("annualization_days", 252)),
        ),
        backtest=BacktestConfig(
            initial_capital=float(backtest_raw.get("initial_capital", 100_000.0)),
            annualization_days=int(backtest_raw.get("annualization_days", 252)),
            risk_free_rate=float(backtest_raw.get("risk_free_rate", 0.0)),
            transaction_cost_bps=float(backtest_raw.get("transaction_cost_bps", 5.0)),
            execution_lag_days=int(backtest_raw.get("execution_lag_days", 1)),
        ),
        ml=MLConfig(
            feature_columns=tuple(str(column) for column in ml_raw.get("feature_columns", [])),
            label_forward_days=int(ml_raw.get("label_forward_days", 20)),
            train_window_days=int(ml_raw.get("train_window_days", 252)),
            min_train_rows=int(ml_raw.get("min_train_rows", 500)),
            min_prediction_assets=int(ml_raw.get("min_prediction_assets", 5)),
            model_type=str(ml_raw.get("model_type", "ridge")),
            random_state=int(ml_raw.get("random_state", 42)),
            ridge_alpha=float(ml_raw.get("ridge_alpha", 1.0)),
            n_estimators=int(ml_raw.get("n_estimators", 200)),
        ),
        paper=PaperTradingConfig(
            state_csv=_resolve_optional_path(paper_raw.get("state_csv"), base_dir),
            initial_cash=float(paper_raw.get("initial_cash", 100_000.0)),
            min_trade_value=float(paper_raw.get("min_trade_value", 100.0)),
            trade_buffer_pct=float(paper_raw.get("trade_buffer_pct", 0.001)),
            slippage_bps=float(paper_raw.get("slippage_bps", 5.0)),
            commission_bps=float(paper_raw.get("commission_bps", 0.0)),
            spread_bps=float(paper_raw.get("spread_bps", 10.0)),
            allow_fractional_shares=bool(paper_raw.get("allow_fractional_shares", True)),
        ),
        monitor=MonitorConfig(
            max_data_age_days=int(monitor_raw.get("max_data_age_days", 7)),
            max_average_feature_missing_rate=float(
                monitor_raw.get("max_average_feature_missing_rate", 0.10)
            ),
            max_single_feature_missing_rate=float(
                monitor_raw.get("max_single_feature_missing_rate", 0.25)
            ),
            drift_lookback_days=int(monitor_raw.get("drift_lookback_days", 60)),
            max_feature_drift_zscore=float(
                monitor_raw.get("max_feature_drift_zscore", 3.0)
            ),
            min_candidate_overlap=float(monitor_raw.get("min_candidate_overlap", 0.50)),
            max_score_concentration=float(
                monitor_raw.get("max_score_concentration", 0.80)
            ),
            min_rank_ic_mean=float(monitor_raw.get("min_rank_ic_mean", 0.0)),
        ),
    )


def _resolve_optional_path(value: object, base_dir: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)
