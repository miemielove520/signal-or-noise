from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FactorConfig, UniverseConfig


def add_basic_factors(
    prices: pd.DataFrame,
    factor_config: FactorConfig,
    universe_config: UniverseConfig,
    fundamentals: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = prices.copy()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False)

    frame["daily_return"] = grouped["adj_close"].pct_change()
    frame["dollar_volume"] = frame["adj_close"] * frame["volume"]
    frame["history_days"] = grouped.cumcount() + 1

    for window in factor_config.momentum_windows:
        frame[f"momentum_{window}"] = grouped["adj_close"].pct_change(window)

    vol_window = factor_config.volatility_window
    frame[f"volatility_{vol_window}"] = grouped["daily_return"].rolling(vol_window).std().reset_index(
        level=0, drop=True
    )
    frame[f"low_volatility_{vol_window}"] = -frame[f"volatility_{vol_window}"]

    trend_window = factor_config.trend_window
    moving_average = grouped["adj_close"].rolling(trend_window).mean().reset_index(
        level=0, drop=True
    )
    frame[f"trend_{trend_window}"] = frame["adj_close"] / moving_average - 1.0

    liquidity_window = factor_config.liquidity_window
    frame[f"liquidity_{liquidity_window}"] = grouped["dollar_volume"].rolling(
        liquidity_window
    ).mean().reset_index(level=0, drop=True)

    money_flow_window = factor_config.money_flow_window
    high_low_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    money_flow_multiplier = (
        (frame["close"] - frame["low"]) - (frame["high"] - frame["close"])
    ) / high_low_range
    frame["money_flow_volume"] = money_flow_multiplier.fillna(0.0) * frame["volume"]
    money_flow_volume_sum = grouped["money_flow_volume"].rolling(
        money_flow_window
    ).sum().reset_index(level=0, drop=True)
    volume_sum = grouped["volume"].rolling(money_flow_window).sum().reset_index(
        level=0, drop=True
    )
    frame[f"money_flow_{money_flow_window}"] = money_flow_volume_sum / volume_sum.replace(
        0, np.nan
    )

    frame["passes_universe"] = (
        (frame["history_days"] >= universe_config.min_history_days)
        & (frame[f"liquidity_{liquidity_window}"] >= universe_config.min_avg_dollar_volume)
    )

    if fundamentals is not None:
        frame = add_fundamental_factors(frame, fundamentals)
    if macro is not None:
        frame = add_macro_features(frame, macro)

    return frame


def add_fundamental_factors(frame: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    fund = fundamentals.copy().sort_values(["ticker", "report_date"]).reset_index(drop=True)
    if "fundamentals_source" not in fund.columns:
        fund["fundamentals_source"] = "yfinance_restated"
    else:
        fund["fundamentals_source"] = (
            fund["fundamentals_source"].fillna("").astype(str).str.strip()
        )
        fund.loc[fund["fundamentals_source"] == "", "fundamentals_source"] = "yfinance_restated"
    fund = _prefer_point_in_time_fundamentals(fund)
    flow_columns = [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "operating_cash_flow",
        "capital_expenditure",
    ]
    grouped = fund.groupby("ticker", group_keys=False)
    for column in flow_columns:
        fund[f"{column}_ttm"] = grouped[column].rolling(4, min_periods=4).sum().reset_index(
            level=0, drop=True
        )

    fund["free_cash_flow_ttm"] = (
        fund["operating_cash_flow_ttm"] - fund["capital_expenditure_ttm"]
    )
    fund["revenue_growth_yoy"] = grouped["revenue_ttm"].pct_change(4)

    feature_columns = [
        "report_date",
        "period_end",
        "ticker",
        "revenue_ttm",
        "gross_profit_ttm",
        "operating_income_ttm",
        "net_income_ttm",
        "free_cash_flow_ttm",
        "revenue_growth_yoy",
        "book_value",
        "total_assets",
        "total_liabilities",
        "shares_outstanding",
        "fundamentals_source",
    ]
    fund = fund[feature_columns]

    merged_parts: list[pd.DataFrame] = []
    for ticker, ticker_frame in frame.sort_values(["ticker", "date"]).groupby("ticker"):
        ticker_fund = fund[fund["ticker"] == ticker].sort_values("report_date")
        ticker_prices = ticker_frame.sort_values("date")
        if ticker_fund.empty:
            merged_parts.append(ticker_prices)
            continue

        merged = pd.merge_asof(
            ticker_prices,
            ticker_fund.drop(columns=["ticker"]),
            left_on="date",
            right_on="report_date",
            direction="backward",
        )
        merged["ticker"] = ticker
        merged_parts.append(merged)

    enriched = pd.concat(merged_parts, ignore_index=True).sort_values(["ticker", "date"])
    enriched = enriched.reset_index(drop=True)

    enriched["fundamental_age_days"] = (
        enriched["date"] - enriched["report_date"]
    ).dt.days
    market_cap = enriched["adj_close"] * enriched["shares_outstanding"]
    enriched["market_cap"] = market_cap
    enriched["earnings_yield"] = safe_divide(enriched["net_income_ttm"], market_cap)
    enriched["book_to_market"] = safe_divide(enriched["book_value"], market_cap)
    enriched["free_cash_flow_yield"] = safe_divide(
        enriched["free_cash_flow_ttm"], market_cap
    )
    enriched["roe"] = safe_divide(enriched["net_income_ttm"], enriched["book_value"])
    enriched["gross_margin"] = safe_divide(
        enriched["gross_profit_ttm"], enriched["revenue_ttm"]
    )
    enriched["debt_to_equity"] = safe_divide(
        enriched["total_liabilities"], enriched["book_value"]
    )
    enriched["low_debt_to_equity"] = -enriched["debt_to_equity"]
    return enriched


def _prefer_point_in_time_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    frame = fundamentals.copy()
    frame["_source_priority"] = np.where(frame["fundamentals_source"] == "sec_pit", 0, 1)
    frame = frame.sort_values(["ticker", "period_end", "_source_priority", "report_date"])
    frame = frame.drop_duplicates(["ticker", "period_end"], keep="first")
    return frame.drop(columns=["_source_priority"]).sort_values(["ticker", "report_date"]).reset_index(drop=True)


def add_macro_features(frame: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in macro.columns if column != "date"]
    rename_map = {column: f"macro_{column}" for column in value_columns}
    macro_features = macro.rename(columns=rename_map).sort_values("date")
    enriched = pd.merge_asof(
        frame.sort_values("date"),
        macro_features,
        on="date",
        direction="backward",
    )
    return enriched.sort_values(["ticker", "date"]).reset_index(drop=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan)
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (valid - valid.mean()) / std


def score_factors(factors: pd.DataFrame, factor_weights: dict[str, float]) -> pd.DataFrame:
    if not factor_weights:
        raise ValueError("factor_weights cannot be empty.")

    missing = [factor for factor in factor_weights if factor not in factors.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing factor columns: {missing_text}")

    scored = factors.copy()
    score_parts: list[pd.Series] = []
    weight_sum = sum(abs(weight) for weight in factor_weights.values())
    if weight_sum == 0:
        raise ValueError("At least one factor weight must be non-zero.")

    for factor_name, weight in factor_weights.items():
        z_name = f"{factor_name}_z"
        scored[z_name] = scored.groupby("date")[factor_name].transform(cross_sectional_zscore)
        score_parts.append(scored[z_name].fillna(0.0) * (weight / weight_sum))

    scored["score"] = sum(score_parts)
    scored.loc[~scored["passes_universe"], "score"] = np.nan
    return scored
