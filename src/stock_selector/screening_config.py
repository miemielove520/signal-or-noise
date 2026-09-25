from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ScreeningThresholds:
    data_quality_min: float = 70.0
    confidence_min: float = 65.0
    market_score_min: float = 55.0
    relative_strength_min: float = 45.0
    signal_score_min: float = 65.0
    backtest_sample_min: int = 10
    backtest_win_rate_min: float = 0.55
    backtest_average_return_min: float = 0.0
    max_backtest_stop_hit_rate: float = 0.50
    backtest_trust_score_min: float = 60.0
    recent_backtest_score_min: float = 45.0
    backtest_decay_score_min: float = 45.0
    min_backtest_avg_dollar_volume: float = 10_000_000.0
    max_backtest_slippage_pct: float = 0.006
    long_fundamental_score_min: float = 60.0
    medium_long_sector_score_min: float = 45.0
    high_probability_target_score: float = 65.0
    near_watchlist_max_missing: int = 2
    near_watchlist_min_score: float = 55.0
    early_watchlist_max_missing: int = 4
    early_watchlist_min_score: float = 45.0

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> "ScreeningThresholds":
        if not values:
            return cls()
        defaults = asdict(cls())
        unknown = sorted(set(values).difference(defaults))
        if unknown:
            raise ValueError(f"Unknown screening threshold keys: {', '.join(unknown)}")
        merged = {**defaults, **values}
        return cls(
            data_quality_min=float(merged["data_quality_min"]),
            confidence_min=float(merged["confidence_min"]),
            market_score_min=float(merged["market_score_min"]),
            relative_strength_min=float(merged["relative_strength_min"]),
            signal_score_min=float(merged["signal_score_min"]),
            backtest_sample_min=int(merged["backtest_sample_min"]),
            backtest_win_rate_min=float(merged["backtest_win_rate_min"]),
            backtest_average_return_min=float(merged["backtest_average_return_min"]),
            max_backtest_stop_hit_rate=float(merged["max_backtest_stop_hit_rate"]),
            backtest_trust_score_min=float(merged["backtest_trust_score_min"]),
            recent_backtest_score_min=float(merged["recent_backtest_score_min"]),
            backtest_decay_score_min=float(merged["backtest_decay_score_min"]),
            min_backtest_avg_dollar_volume=float(merged["min_backtest_avg_dollar_volume"]),
            max_backtest_slippage_pct=float(merged["max_backtest_slippage_pct"]),
            long_fundamental_score_min=float(merged["long_fundamental_score_min"]),
            medium_long_sector_score_min=float(merged["medium_long_sector_score_min"]),
            high_probability_target_score=float(merged["high_probability_target_score"]),
            near_watchlist_max_missing=int(merged["near_watchlist_max_missing"]),
            near_watchlist_min_score=float(merged["near_watchlist_min_score"]),
            early_watchlist_max_missing=int(merged["early_watchlist_max_missing"]),
            early_watchlist_min_score=float(merged["early_watchlist_min_score"]),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TradingRules:
    preferred_entry_style: str = "balanced"
    preferred_entry_style_zh: str = "均衡"
    short_atr_stop_multiple: float = 1.5
    medium_atr_stop_multiple: float = 2.0
    long_atr_stop_multiple: float = 3.0
    short_target_r_multiple: float = 2.0
    medium_target_r_multiple: float = 2.5
    long_target_r_multiple: float = 3.0
    short_max_chase_pct: float = 0.04
    medium_max_chase_pct: float = 0.08
    long_max_chase_pct: float = 0.15
    short_time_stop_days: int = 20
    medium_time_stop_days: int = 60
    long_time_stop_days: int = 120
    trailing_stop_trigger_r: float = 1.5
    trailing_stop_lock_r: float = 0.2
    sell_rule: str = "stop_target_or_trend_break"
    sell_rule_zh: str = "止损、目标价或趋势跌破时退出"

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> "TradingRules":
        if not values:
            return cls()
        defaults = asdict(cls())
        unknown = sorted(set(values).difference(defaults))
        if unknown:
            raise ValueError(f"Unknown trading rule keys: {', '.join(unknown)}")
        merged = {**defaults, **values}
        return cls(
            preferred_entry_style=str(merged["preferred_entry_style"]),
            preferred_entry_style_zh=str(merged["preferred_entry_style_zh"]),
            short_atr_stop_multiple=float(merged["short_atr_stop_multiple"]),
            medium_atr_stop_multiple=float(merged["medium_atr_stop_multiple"]),
            long_atr_stop_multiple=float(merged["long_atr_stop_multiple"]),
            short_target_r_multiple=float(merged["short_target_r_multiple"]),
            medium_target_r_multiple=float(merged["medium_target_r_multiple"]),
            long_target_r_multiple=float(merged["long_target_r_multiple"]),
            short_max_chase_pct=float(merged["short_max_chase_pct"]),
            medium_max_chase_pct=float(merged["medium_max_chase_pct"]),
            long_max_chase_pct=float(merged["long_max_chase_pct"]),
            short_time_stop_days=int(merged["short_time_stop_days"]),
            medium_time_stop_days=int(merged["medium_time_stop_days"]),
            long_time_stop_days=int(merged["long_time_stop_days"]),
            trailing_stop_trigger_r=float(merged["trailing_stop_trigger_r"]),
            trailing_stop_lock_r=float(merged["trailing_stop_lock_r"]),
            sell_rule=str(merged["sell_rule"]),
            sell_rule_zh=str(merged["sell_rule_zh"]),
        )

    def atr_stop_multiple_for(self, horizon: str) -> float:
        return float(getattr(self, f"{horizon}_atr_stop_multiple"))

    def target_r_multiple_for(self, horizon: str) -> float:
        return float(getattr(self, f"{horizon}_target_r_multiple"))

    def max_chase_pct_for(self, horizon: str) -> float:
        return float(getattr(self, f"{horizon}_max_chase_pct"))

    def time_stop_days_for(self, horizon: str) -> int:
        return int(getattr(self, f"{horizon}_time_stop_days"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScreeningProfile:
    name: str
    name_zh: str
    thresholds: ScreeningThresholds
    trading_rules: TradingRules = TradingRules()
    ticker_symbols: tuple[str, ...] = ()
    sector_keywords: tuple[str, ...] = ()
    industry_keywords: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        name: str,
        values: dict[str, object],
        base_thresholds: ScreeningThresholds,
    ) -> "ScreeningProfile":
        metadata_keys = {
            "name_zh",
            "ticker_symbols",
            "sector_keywords",
            "industry_keywords",
            "trading",
        }
        threshold_values = {
            key: value for key, value in values.items() if key not in metadata_keys
        }
        thresholds = ScreeningThresholds.from_mapping(
            {**base_thresholds.to_dict(), **threshold_values}
        )
        return cls(
            name=name,
            name_zh=str(values.get("name_zh", name)),
            thresholds=thresholds,
            trading_rules=TradingRules.from_mapping(
                values.get("trading") if isinstance(values.get("trading"), dict) else None
            ),
            ticker_symbols=_string_tuple(values.get("ticker_symbols"), upper=True),
            sector_keywords=_string_tuple(values.get("sector_keywords"), upper=False),
            industry_keywords=_string_tuple(values.get("industry_keywords"), upper=False),
        )

    def matches(self, ticker: str, sector: str | None, industry: str | None) -> bool:
        ticker_text = ticker.upper().strip()
        sector_text = str(sector or "").lower().strip()
        industry_text = str(industry or "").lower().strip()
        if ticker_text and ticker_text in self.ticker_symbols:
            return True
        if any(keyword in sector_text for keyword in self.sector_keywords):
            return True
        return any(keyword in industry_text for keyword in self.industry_keywords)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "name_zh": self.name_zh,
            "ticker_symbols": list(self.ticker_symbols),
            "sector_keywords": list(self.sector_keywords),
            "industry_keywords": list(self.industry_keywords),
            "thresholds": self.thresholds.to_dict(),
            "trading_rules": self.trading_rules.to_dict(),
        }


@dataclass(frozen=True)
class ScreeningConfig:
    default_thresholds: ScreeningThresholds
    profiles: tuple[ScreeningProfile, ...] = ()

    def resolve_profile(
        self,
        ticker: str,
        sector: str | None = None,
        industry: str | None = None,
    ) -> ScreeningProfile:
        for profile in self.profiles:
            if profile.matches(ticker=ticker, sector=sector, industry=industry):
                return profile
        return ScreeningProfile(
            name="default",
            name_zh="默认规则",
            thresholds=self.default_thresholds,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "default_thresholds": self.default_thresholds.to_dict(),
            "profiles": [profile.to_dict() for profile in self.profiles],
        }

    def with_profile_thresholds(
        self,
        profile_thresholds: dict[str, ScreeningThresholds],
    ) -> "ScreeningConfig":
        profiles = tuple(
            ScreeningProfile(
                name=profile.name,
                name_zh=profile.name_zh,
                thresholds=profile_thresholds.get(profile.name, profile.thresholds),
                trading_rules=profile.trading_rules,
                ticker_symbols=profile.ticker_symbols,
                sector_keywords=profile.sector_keywords,
                industry_keywords=profile.industry_keywords,
            )
            for profile in self.profiles
        )
        return ScreeningConfig(
            default_thresholds=profile_thresholds.get("default", self.default_thresholds),
            profiles=profiles,
        )


def default_screening_config() -> ScreeningConfig:
    default_thresholds = ScreeningThresholds()
    profile_values = {
        "ai_infrastructure": {
            "name_zh": "AI基建规则",
            "ticker_symbols": [
                "NVDA",
                "AVGO",
                "AMD",
                "ANET",
                "MRVL",
                "SMCI",
                "DELL",
                "ARM",
                "TSM",
                "ASML",
                "AMAT",
            ],
            "industry_keywords": [
                "semiconductor equipment",
                "networking",
                "communications equipment",
                "computer hardware",
            ],
            "signal_score_min": 69,
            "market_score_min": 58,
            "relative_strength_min": 50,
            "backtest_sample_min": 12,
            "backtest_win_rate_min": 0.56,
            "max_backtest_stop_hit_rate": 0.50,
            "backtest_trust_score_min": 62,
            "recent_backtest_score_min": 48,
            "backtest_decay_score_min": 48,
            "medium_long_sector_score_min": 52,
            "trading": {
                "preferred_entry_style": "breakout",
                "preferred_entry_style_zh": "突破优先",
                "short_atr_stop_multiple": 1.7,
                "medium_atr_stop_multiple": 2.2,
                "long_atr_stop_multiple": 3.2,
                "short_target_r_multiple": 2.2,
                "medium_target_r_multiple": 2.8,
                "long_target_r_multiple": 3.4,
                "short_max_chase_pct": 0.05,
                "medium_max_chase_pct": 0.10,
                "long_max_chase_pct": 0.18,
                "short_time_stop_days": 15,
                "medium_time_stop_days": 45,
                "long_time_stop_days": 120,
                "sell_rule": "volatility_trend_break",
                "sell_rule_zh": "波动放大且趋势跌破时退出",
            },
        },
        "cybersecurity": {
            "name_zh": "网络安全规则",
            "ticker_symbols": ["PANW", "CRWD", "ZS", "FTNT", "OKTA", "S", "QLYS", "NET"],
            "industry_keywords": ["security", "cybersecurity"],
            "signal_score_min": 68,
            "relative_strength_min": 52,
            "backtest_win_rate_min": 0.57,
            "max_backtest_stop_hit_rate": 0.48,
            "backtest_trust_score_min": 62,
            "recent_backtest_score_min": 48,
            "backtest_decay_score_min": 48,
            "medium_long_sector_score_min": 48,
            "near_watchlist_min_score": 57,
            "trading": {
                "preferred_entry_style": "breakout",
                "preferred_entry_style_zh": "突破优先",
                "short_atr_stop_multiple": 1.6,
                "medium_atr_stop_multiple": 2.1,
                "long_atr_stop_multiple": 2.8,
                "short_target_r_multiple": 2.1,
                "medium_target_r_multiple": 2.6,
                "long_target_r_multiple": 3.0,
                "short_max_chase_pct": 0.045,
                "medium_max_chase_pct": 0.085,
                "long_max_chase_pct": 0.14,
                "short_time_stop_days": 15,
                "medium_time_stop_days": 45,
                "long_time_stop_days": 100,
                "sell_rule": "momentum_break_or_event_risk",
                "sell_rule_zh": "动量跌破或事件风险升高时退出",
            },
        },
        "saas_software": {
            "name_zh": "SaaS软件规则",
            "ticker_symbols": ["NOW", "CRM", "SNOW", "DDOG", "MDB", "ADBE", "INTU", "PLTR"],
            "industry_keywords": [
                "software",
                "cloud",
                "application",
                "infrastructure",
                "internet content",
            ],
            "signal_score_min": 67,
            "relative_strength_min": 50,
            "backtest_win_rate_min": 0.56,
            "max_backtest_stop_hit_rate": 0.50,
            "backtest_trust_score_min": 61,
            "recent_backtest_score_min": 46,
            "backtest_decay_score_min": 46,
            "medium_long_sector_score_min": 47,
            "trading": {
                "preferred_entry_style": "balanced",
                "preferred_entry_style_zh": "突破与回调均衡",
                "short_atr_stop_multiple": 1.5,
                "medium_atr_stop_multiple": 2.0,
                "long_atr_stop_multiple": 2.8,
                "short_target_r_multiple": 2.0,
                "medium_target_r_multiple": 2.5,
                "long_target_r_multiple": 3.1,
                "short_max_chase_pct": 0.04,
                "medium_max_chase_pct": 0.08,
                "long_max_chase_pct": 0.14,
                "short_time_stop_days": 18,
                "medium_time_stop_days": 55,
                "long_time_stop_days": 110,
                "sell_rule": "growth_trend_break",
                "sell_rule_zh": "成长股趋势跌破或基本面恶化时退出",
            },
        },
        "fintech_high_beta": {
            "name_zh": "金融科技高波动规则",
            "ticker_symbols": ["PYPL", "XYZ", "COIN", "HOOD", "SOFI", "AFRM", "UPST"],
            "industry_keywords": ["fintech", "credit services", "capital markets", "payments"],
            "data_quality_min": 72,
            "confidence_min": 68,
            "signal_score_min": 70,
            "market_score_min": 58,
            "relative_strength_min": 52,
            "backtest_sample_min": 14,
            "backtest_win_rate_min": 0.58,
            "max_backtest_stop_hit_rate": 0.52,
            "backtest_trust_score_min": 65,
            "recent_backtest_score_min": 50,
            "backtest_decay_score_min": 50,
            "near_watchlist_min_score": 58,
            "trading": {
                "preferred_entry_style": "pullback",
                "preferred_entry_style_zh": "回调优先",
                "short_atr_stop_multiple": 1.8,
                "medium_atr_stop_multiple": 2.4,
                "long_atr_stop_multiple": 3.2,
                "short_target_r_multiple": 2.4,
                "medium_target_r_multiple": 3.0,
                "long_target_r_multiple": 3.5,
                "short_max_chase_pct": 0.035,
                "medium_max_chase_pct": 0.07,
                "long_max_chase_pct": 0.12,
                "short_time_stop_days": 12,
                "medium_time_stop_days": 35,
                "long_time_stop_days": 90,
                "sell_rule": "high_beta_risk_cut",
                "sell_rule_zh": "高波动品种触发止损、放量破位或风险升高时退出",
            },
        },
        "defensive_quality": {
            "name_zh": "防御质量股规则",
            "ticker_symbols": ["COST", "WMT", "PG", "KO", "PEP", "MCD", "MDLZ", "CL", "CLX"],
            "sector_keywords": ["consumer defensive"],
            "industry_keywords": ["household", "beverages", "discount stores", "packaged foods"],
            "signal_score_min": 62,
            "market_score_min": 50,
            "relative_strength_min": 42,
            "backtest_win_rate_min": 0.54,
            "max_backtest_stop_hit_rate": 0.48,
            "backtest_trust_score_min": 58,
            "recent_backtest_score_min": 42,
            "backtest_decay_score_min": 42,
            "long_fundamental_score_min": 65,
            "medium_long_sector_score_min": 42,
            "early_watchlist_min_score": 42,
            "trading": {
                "preferred_entry_style": "pullback",
                "preferred_entry_style_zh": "回调优先",
                "short_atr_stop_multiple": 1.2,
                "medium_atr_stop_multiple": 1.7,
                "long_atr_stop_multiple": 2.4,
                "short_target_r_multiple": 1.8,
                "medium_target_r_multiple": 2.2,
                "long_target_r_multiple": 2.6,
                "short_max_chase_pct": 0.025,
                "medium_max_chase_pct": 0.05,
                "long_max_chase_pct": 0.09,
                "short_time_stop_days": 25,
                "medium_time_stop_days": 75,
                "long_time_stop_days": 160,
                "sell_rule": "defensive_trend_or_fundamental_break",
                "sell_rule_zh": "防御股趋势转弱或基本面质量下降时退出",
            },
        },
        "industrial_quality": {
            "name_zh": "工业质量股规则",
            "ticker_symbols": ["AAON", "CARR", "LII", "TT", "JCI", "IR", "ETN", "HON", "PH"],
            "sector_keywords": ["industrials"],
            "industry_keywords": [
                "building products",
                "hvac",
                "specialty industrial machinery",
                "electrical equipment",
            ],
            "signal_score_min": 64,
            "market_score_min": 52,
            "relative_strength_min": 45,
            "backtest_win_rate_min": 0.55,
            "max_backtest_stop_hit_rate": 0.50,
            "backtest_trust_score_min": 59,
            "recent_backtest_score_min": 44,
            "backtest_decay_score_min": 44,
            "long_fundamental_score_min": 62,
            "medium_long_sector_score_min": 45,
            "near_watchlist_min_score": 54,
            "trading": {
                "preferred_entry_style": "balanced",
                "preferred_entry_style_zh": "突破与回调均衡",
                "short_atr_stop_multiple": 1.4,
                "medium_atr_stop_multiple": 1.9,
                "long_atr_stop_multiple": 2.6,
                "short_target_r_multiple": 2.0,
                "medium_target_r_multiple": 2.5,
                "long_target_r_multiple": 3.0,
                "short_max_chase_pct": 0.035,
                "medium_max_chase_pct": 0.07,
                "long_max_chase_pct": 0.12,
                "short_time_stop_days": 20,
                "medium_time_stop_days": 60,
                "long_time_stop_days": 130,
                "sell_rule": "cycle_trend_break",
                "sell_rule_zh": "工业周期趋势跌破或板块转弱时退出",
            },
        },
        "semiconductor": {
            "name_zh": "半导体规则",
            "ticker_symbols": ["INTC", "QCOM", "MU", "TXN", "ADI", "LRCX", "KLAC", "MCHP"],
            "industry_keywords": ["semiconductor", "chip", "integrated circuit", "hardware"],
            "signal_score_min": 68,
            "market_score_min": 57,
            "relative_strength_min": 48,
            "backtest_sample_min": 12,
            "max_backtest_stop_hit_rate": 0.52,
            "backtest_trust_score_min": 62,
            "recent_backtest_score_min": 48,
            "backtest_decay_score_min": 48,
            "medium_long_sector_score_min": 50,
            "trading": {
                "preferred_entry_style": "breakout",
                "preferred_entry_style_zh": "突破优先",
                "short_atr_stop_multiple": 1.7,
                "medium_atr_stop_multiple": 2.2,
                "long_atr_stop_multiple": 3.0,
                "short_target_r_multiple": 2.2,
                "medium_target_r_multiple": 2.8,
                "long_target_r_multiple": 3.3,
                "short_max_chase_pct": 0.05,
                "medium_max_chase_pct": 0.10,
                "long_max_chase_pct": 0.16,
                "short_time_stop_days": 15,
                "medium_time_stop_days": 45,
                "long_time_stop_days": 115,
                "sell_rule": "cycle_volatility_break",
                "sell_rule_zh": "半导体周期波动放大且趋势跌破时退出",
            },
        },
        "mega_cap_tech": {
            "name_zh": "大盘科技规则",
            "ticker_symbols": [
                "AAPL",
                "MSFT",
                "NVDA",
                "AMZN",
                "META",
                "GOOGL",
                "GOOG",
                "AVGO",
                "ORCL",
            ],
            "signal_score_min": 66,
            "confidence_min": 67,
            "data_quality_min": 72,
            "backtest_win_rate_min": 0.56,
            "max_backtest_stop_hit_rate": 0.50,
            "backtest_trust_score_min": 60,
            "recent_backtest_score_min": 45,
            "backtest_decay_score_min": 45,
            "long_fundamental_score_min": 62,
            "trading": {
                "preferred_entry_style": "balanced",
                "preferred_entry_style_zh": "突破与回调均衡",
                "short_atr_stop_multiple": 1.4,
                "medium_atr_stop_multiple": 1.9,
                "long_atr_stop_multiple": 2.8,
                "short_target_r_multiple": 2.0,
                "medium_target_r_multiple": 2.5,
                "long_target_r_multiple": 3.2,
                "short_max_chase_pct": 0.04,
                "medium_max_chase_pct": 0.08,
                "long_max_chase_pct": 0.14,
                "short_time_stop_days": 20,
                "medium_time_stop_days": 60,
                "long_time_stop_days": 130,
                "sell_rule": "mega_cap_trend_or_quality_break",
                "sell_rule_zh": "大盘科技股趋势跌破或质量分恶化时退出",
            },
        },
    }
    return ScreeningConfig(
        default_thresholds=default_thresholds,
        profiles=tuple(
            ScreeningProfile.from_mapping(
                name=name,
                values=values,
                base_thresholds=default_thresholds,
            )
            for name, values in profile_values.items()
        ),
    )


def load_screening_config(path: str | Path | None) -> ScreeningConfig:
    if path is None:
        return default_screening_config()
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    default_raw = raw.get("screening_thresholds", raw.get("default", {}))
    if not isinstance(default_raw, dict):
        raise ValueError("Screening config default thresholds must be a table.")
    default_thresholds = ScreeningThresholds.from_mapping(default_raw)
    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        raise ValueError("screening profiles must be a TOML table.")
    profiles = tuple(
        ScreeningProfile.from_mapping(
            name=str(name),
            values=values,
            base_thresholds=default_thresholds,
        )
        for name, values in profiles_raw.items()
        if isinstance(values, dict)
    )
    if not profiles and "profiles" not in raw:
        profiles = default_screening_config().profiles
    return ScreeningConfig(default_thresholds=default_thresholds, profiles=profiles)


def load_screening_thresholds(path: str | Path | None) -> ScreeningThresholds:
    return load_screening_config(path).default_thresholds


def render_screening_config_toml(
    config: ScreeningConfig,
    header_lines: list[str] | tuple[str, ...] | None = None,
) -> str:
    lines: list[str] = []
    for line in header_lines or ():
        lines.append(f"# {line}" if line else "#")
    if lines:
        lines.append("")
    lines.append("[screening_thresholds]")
    lines.extend(_threshold_toml_lines(config.default_thresholds))
    for profile in config.profiles:
        lines.extend(["", f"[profiles.{profile.name}]"])
        lines.append(f'name_zh = "{_escape_toml_string(profile.name_zh)}"')
        if profile.ticker_symbols:
            lines.append(f"ticker_symbols = {_toml_string_list(profile.ticker_symbols)}")
        if profile.sector_keywords:
            lines.append(f"sector_keywords = {_toml_string_list(profile.sector_keywords)}")
        if profile.industry_keywords:
            lines.append(f"industry_keywords = {_toml_string_list(profile.industry_keywords)}")
        lines.extend(_threshold_toml_lines(profile.thresholds, base=config.default_thresholds))
        trading_lines = _trading_rule_toml_lines(profile.trading_rules)
        if trading_lines:
            lines.extend(["", f"[profiles.{profile.name}.trading]"])
            lines.extend(trading_lines)
    return "\n".join(lines).rstrip() + "\n"


def _string_tuple(value: object, upper: bool) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.upper().strip() if upper else value.lower().strip()
        return (normalized,)
    if isinstance(value, list):
        return tuple(
            (str(item).upper().strip() if upper else str(item).lower().strip())
            for item in value
            if str(item).strip()
        )
    text = str(value).upper().strip() if upper else str(value).lower().strip()
    return tuple(text.split())


def _threshold_toml_lines(
    thresholds: ScreeningThresholds,
    base: ScreeningThresholds | None = None,
) -> list[str]:
    values = thresholds.to_dict()
    base_values = base.to_dict() if base is not None else {}
    lines: list[str] = []
    for key, value in values.items():
        if base is not None and base_values.get(key) == value:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return lines


def _trading_rule_toml_lines(rules: TradingRules) -> list[str]:
    values = rules.to_dict()
    lines: list[str] = []
    for key, value in values.items():
        lines.append(f"{key} = {_toml_value(value)}")
    return lines


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return f'"{_escape_toml_string(str(value))}"'


def _toml_string_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(f'"{_escape_toml_string(value)}"' for value in values) + "]"


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
