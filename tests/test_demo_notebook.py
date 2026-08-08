"""Runs the tutorial notebook's code cells against the installed package, so
API drift between src/ and the notebook shows up as a test failure instead
of being discovered by a reader. Skipped unless the optional `plotly` extra
(the `demo` group) is installed, since the notebook renders figures with it.
"""

import json
import pathlib

import pytest

pytest.importorskip("plotly")

NOTEBOOK = pathlib.Path(__file__).parent.parent / "demo" / "explore_arbital.ipynb"


def test_notebook_runs_top_to_bottom():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace = {}
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        code = compile(source, filename=f"<{NOTEBOOK.name} cell {i}>", mode="exec")
        exec(code, namespace)  # noqa: S102 -- executing our own notebook's cells, not untrusted input
