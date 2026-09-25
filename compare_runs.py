from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.run_comparison import compare_validation_runs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two validation run folders.")
    parser.add_argument("previous_run", help="Previous validation output folder.")
    parser.add_argument("current_run", help="Current validation output folder.")
    parser.add_argument(
        "--output-dir",
        help="Output folder for comparison files. Defaults to the current run folder.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_validation_runs(
        previous_dir=args.previous_run,
        current_dir=args.current_run,
        output_dir=args.output_dir,
    )
    print("Validation run comparison completed. / 验证运行对比完成。")
    print(f"Previous run / 上一次运行: {args.previous_run}")
    print(f"Current run / 当前运行: {args.current_run}")
    print(f"Output folder / 输出文件夹: {result.output_dir}")
    print()
    if not result.adoption_decision.empty:
        final = result.adoption_decision[
            result.adoption_decision["check_name"] == "final_decision"
        ]
        if not final.empty:
            row = final.iloc[0]
            print("Config adoption decision / 配置采用结论")
            print(f"- {row['status']} / {row['status_zh']}: {row['detail_zh']}")
            print()
    for row in result.summary.itertuples(index=False):
        print(
            f"- {row.metric} / {row.metric_zh}: "
            f"{row.previous_value} -> {row.current_value}, "
            f"change={row.change}, {row.assessment_zh}"
        )
    print()
    print(f"Open report: {result.output_files['markdown_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
