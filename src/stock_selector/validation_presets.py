from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationPreset:
    name: str
    universe: str | None
    all_universes: bool
    period: str
    step_days: int
    min_history_days: int
    output_dir: str
    description: str
    description_zh: str


VALIDATION_PRESETS = {
    "quick": ValidationPreset(
        name="quick",
        universe="research-core",
        all_universes=False,
        period="1y",
        step_days=120,
        min_history_days=80,
        output_dir="outputs/walk_forward/preset_quick",
        description="Fast smoke validation on the larger research-core universe.",
        description_zh="在较大的research-core股票池上快速验证。",
    ),
    "standard": ValidationPreset(
        name="standard",
        universe="research-core",
        all_universes=False,
        period="5y",
        step_days=20,
        min_history_days=170,
        output_dir="outputs/walk_forward/preset_standard",
        description="Research-grade validation with stronger sample coverage.",
        description_zh="样本覆盖更充分的研究级验证。",
    ),
    "deep": ValidationPreset(
        name="deep",
        universe=None,
        all_universes=True,
        period="10y",
        step_days=20,
        min_history_days=252,
        output_dir="outputs/walk_forward/preset_deep",
        description="Slow all-universe validation for stronger evidence.",
        description_zh="较慢的全部股票池深度验证，用于获得更强证据。",
    ),
    "extreme": ValidationPreset(
        name="extreme",
        universe=None,
        all_universes=True,
        period="10y",
        step_days=10,
        min_history_days=252,
        output_dir="outputs/walk_forward/preset_extreme",
        description="Very slow maximum-coverage validation across all built-in universes.",
        description_zh="非常慢的最大覆盖验证，覆盖全部内置股票池并提高取样密度。",
    ),
}


def validation_preset_choices() -> tuple[str, ...]:
    return tuple(VALIDATION_PRESETS)


def apply_validation_preset(args) -> object:
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        _apply_validation_defaults(args)
        return args

    preset = VALIDATION_PRESETS[preset_name]
    has_manual_universe = bool(
        getattr(args, "tickers", None)
        or getattr(args, "universe", None)
        or getattr(args, "universe_file", None)
    )
    if not has_manual_universe:
        args.universe = preset.universe
    if not getattr(args, "all_universes", False) and not has_manual_universe:
        args.all_universes = preset.all_universes
    if getattr(args, "period", None) is None:
        args.period = preset.period
    if getattr(args, "step_days", None) is None:
        args.step_days = preset.step_days
    if getattr(args, "min_history_days", None) is None:
        args.min_history_days = preset.min_history_days
    if getattr(args, "output_dir", None) is None:
        args.output_dir = preset.output_dir
    args.preset_description = preset.description
    args.preset_description_zh = preset.description_zh
    return args


def _apply_validation_defaults(args) -> None:
    if getattr(args, "period", None) is None:
        args.period = "5y"
    if getattr(args, "step_days", None) is None:
        args.step_days = 20
    if getattr(args, "min_history_days", None) is None:
        args.min_history_days = 170
    if getattr(args, "output_dir", None) is None:
        args.output_dir = "outputs/walk_forward/latest"
    args.preset_description = ""
    args.preset_description_zh = ""
