from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_selector.dashboard import build_dashboard  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML dashboard from analysis outputs. "
        "从分析结果生成一个可用浏览器打开的网页看板。",
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--open", action="store_true", help="Open the page in your browser when done.")
    args = parser.parse_args(argv)

    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M")
    index_path = build_dashboard(
        outputs_dir=args.outputs_dir, generated_at=stamp, today=now.date()
    )
    print(f"Dashboard / 看板: {index_path}")

    if args.open:
        import webbrowser

        webbrowser.open(index_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
