import json
from pathlib import Path


def test_teaching_notebooks_are_valid_and_clear_outputs():
    root = Path(__file__).parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) >= 6
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
