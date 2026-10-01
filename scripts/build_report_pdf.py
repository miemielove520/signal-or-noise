"""Typeset results/REPORT.md as a paper-style PDF (results/REPORT.pdf).

Markdown is converted to HTML with a print stylesheet and printed with a
headless Chrome/Chromium, so no LaTeX installation is needed.

    pip install -e ".[research]"
    python scripts/build_report_pdf.py [--chrome /path/to/chrome]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)

STYLE = """
@page { size: Letter; margin: 22mm 20mm 22mm 20mm; }
html { font-size: 10.5pt; }
body {
  font-family: "Charter", "Iowan Old Style", "Georgia", serif;
  color: #111; line-height: 1.45; max-width: 100%;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1, h2, h3 { font-family: "Helvetica Neue", "Arial", sans-serif; line-height: 1.2; }
h1 { font-size: 20pt; margin: 0 0 4pt; letter-spacing: -0.2pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
h2, h3 { break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
em:first-child { color: #444; }
a { color: #1c5cab; text-decoration: none; }
code { font-family: "Menlo", "Consolas", monospace; font-size: 8.8pt; background: #f2f2f0; padding: 0 2pt; }
pre { background: #f6f6f4; padding: 6pt 8pt; font-size: 8.5pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt auto; font-size: 9pt; break-inside: avoid; }
th, td { padding: 3pt 7pt; border-bottom: 0.5pt solid #ccc; text-align: left; vertical-align: top; }
th { border-bottom: 0.9pt solid #555; font-family: "Helvetica Neue", "Arial", sans-serif; }
img { display: block; max-width: 92%; margin: 10pt auto 2pt; break-inside: avoid; }
blockquote { margin: 8pt 16pt; color: #333; border-left: 2pt solid #ccc; padding-left: 8pt; }
hr { border: none; border-top: 0.6pt solid #bbb; margin: 14pt 0; }
"""


def find_chrome(explicit: str | None) -> str:
    for candidate in ([explicit] if explicit else []) + list(CHROME_CANDIDATES):
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate if Path(candidate).exists() else shutil.which(candidate)
    raise SystemExit("No Chrome/Chromium found; pass --chrome /path/to/browser.")


def _python_markdown_indents(text: str) -> str:
    """Double 2-space list indentation (GitHub style) to the 4 spaces Python-Markdown expects."""
    lines, in_code = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not in_code and indent and indent % 2 == 0 and stripped:
            line = " " * (indent * 2) + stripped
        lines.append(line)
    return "\n".join(lines) + "\n"


def build(chrome: str, source: Path, target: Path) -> None:
    body = markdown.markdown(
        _python_markdown_indents(source.read_text(encoding="utf-8")),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{source.stem}</title><style>{STYLE}</style></head><body>{body}</body></html>"
    )
    # Write next to the report so relative figure paths resolve.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", dir=source.parent, delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
        page = Path(tmp.name)
    try:
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={target}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        page.unlink(missing_ok=True)
    print(f"wrote {target.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chrome", help="Path to a Chrome or Chromium executable.")
    args = parser.parse_args()
    build(find_chrome(args.chrome), RESULTS / "REPORT.md", RESULTS / "REPORT.pdf")


if __name__ == "__main__":
    main()
