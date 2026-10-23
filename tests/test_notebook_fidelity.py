"""Proves the modular code is statement-for-statement identical to the original notebook.

Every top-level AST node of every notebook code cell (cells 1-8) must appear,
unchanged, in the package modules; cells 9-11 must appear in scripts/run_benchmark.py.
Import statements are excluded: they are redistributed across modules by design.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "original" / "CV_PACKAGE_FINAL.ipynb"


def _nodes(source):
    return [
        ast.dump(n)
        for n in ast.parse(source).body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
        and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)
                 and isinstance(n.value.value, str))  # module docstrings added by the refactor
    ]


def _notebook_cells(indices):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {
        i: "".join(nb["cells"][i]["source"])
        for i in indices
        if nb["cells"][i]["cell_type"] == "code"  # skips markdown cell 4
    }


def _all_nodes(paths):
    out = []
    for p in paths:
        out.extend(_nodes(p.read_text(encoding="utf-8")))
    return out


def test_package_modules_preserve_cells_1_to_8():
    modules = sorted((ROOT / "crackvision").glob("*.py"))
    modules = [m for m in modules if m.name != "__init__.py"]
    pool = _all_nodes(modules)
    expected = []
    for src in _notebook_cells(range(1, 9)).values():
        expected.extend(_nodes(src))
    assert sorted(pool) == sorted(expected)


def test_benchmark_script_preserves_cells_9_to_11_in_order():
    script_nodes = _nodes((ROOT / "scripts" / "run_benchmark.py").read_text(encoding="utf-8"))
    expected = []
    for i in (9, 10, 11):
        expected.extend(_nodes(_notebook_cells([i])[i]))
    assert script_nodes == expected
