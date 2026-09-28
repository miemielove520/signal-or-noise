"""Every module and every public function or class in the library is documented."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stock_selector"


class DocstringCoverageTest(unittest.TestCase):
    def test_modules_and_public_definitions_have_docstrings(self) -> None:
        missing = []
        for path in sorted(PACKAGE.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            relative = path.relative_to(PACKAGE)
            if path.name != "__init__.py" or tree.body:
                if ast.get_docstring(tree) is None:
                    missing.append(f"{relative} (module)")
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                    if ast.get_docstring(node) is None:
                        missing.append(f"{relative}:{node.name}")
        self.assertEqual(missing, [], "Add a docstring to: " + ", ".join(missing))


if __name__ == "__main__":
    unittest.main()
