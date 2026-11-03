"""Build the reviewed Kaggle notebook from the maintained percent-cell sources."""
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

DESCRIPTIONS = {
    "Environment": "Install only missing packages, then confirm the audited two-GPU runtime.",
    "Dataset": "Verify the saved SVRDD audit and prepare leakage-screened manifests for the attached data.",
    "Settings": "Set the shared model, resolution, seed, and bounded training configuration.",
    "Worker": "Define the resumable per-model training worker and checkpoint preservation callback.",
    "Training launcher": "Run each bounded training job, preserve its checkpoint, and record its duration.",
    "Pilot": "Run the one-epoch pilot and confirm its saved checkpoint reloads for inference.",
    "Train YOLOv8n": "Train the YOLOv8 nano baseline within the shared experiment budget.",
    "Train YOLO11n": "Train the YOLO11 nano candidate with the same settings and budget.",
    "Train YOLO26n": "Train the YOLO26 nano candidate with the same settings and budget.",
    "Validation": "Compare all completed models on validation data and select the winner before test evaluation.",
    "Export": "Reload and retain the model checkpoints and compact project outputs in Kaggle storage.",
    "Test manifest and matching": "Define the fixed test paths, ground-truth loader, matching rule, and training evidence checks.",
    "GPU prediction and timing": "Set one shared prediction and synchronized timing boundary for all candidates.",
    "Held-out evaluation": "Define independent test metrics, per-class matching, and model timing measurements.",
    "Evaluation worker": "Evaluate one checkpoint and record its metrics, losses, timing, and memory evidence.",
    "Evaluation orchestration": "Freeze the validation-selected winner and evaluate all three models once on held-out test data.",
    "Run held-out evaluation": "Run the final evaluation after all three training runs and validation selection are complete.",
}


def read_percent_cells(path: Path) -> list[tuple[str, str]]:
    text = path.read_bytes().decode("utf-8")
    lines = text.splitlines(keepends=True)
    marks = [i for i, line in enumerate(lines) if line.startswith("# %%")]
    if not marks or marks[0] != 0:
        raise ValueError(f"Expected a percent-cell marker at the start of {path.name}")
    result = []
    for n, start in enumerate(marks):
        marker = lines[start].strip()
        title = marker[4:].strip().lstrip("—- ").strip()
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        body = "".join(lines[start + 1:end]).strip()
        if not body:
            raise ValueError(f"Empty source cell in {path.name}: {title}")
        result.append((title, body))
    return result


def cell_key(title: str) -> str:
    low = title.lower()
    if low.startswith("1."): return "Test manifest and matching"
    if low.startswith("2."): return "GPU prediction and timing"
    if low.startswith("3."): return "Held-out evaluation"
    if "per-checkpoint" in low: return "Evaluation worker"
    if "orchestrate" in low: return "Evaluation orchestration"
    if low.startswith("run once"): return "Run held-out evaluation"
    return title.split("—")[0].strip().rstrip(".")


def main() -> None:
    core = read_percent_cells(HERE / "notebook_core.py")
    metrics = read_percent_cells(HERE / "notebook_metrics.py")
    dataset = (HERE / "notebook_dataset.py").read_bytes().decode("utf-8").strip()
    if len(core) != 10 or len(metrics) != 6:
        raise ValueError(f"Unexpected sources: core={len(core)} cells, metrics={len(metrics)} cells")

    # Environment first, dataset second, then settings/worker/pilot and the three runs.
    # Validation follows YOLO26n; metrics run next; the original export remains last.
    specs = [core[0], ("Dataset", dataset), *core[1:9], *metrics, core[9]]
    code_cells = []
    notebook_cells = []
    backup = []
    for index, (title, source) in enumerate(specs, 1):
        key = "Dataset" if title == "Dataset" else cell_key(title)
        sentence = DESCRIPTIONS.get(key)
        if sentence is None:
            raise ValueError(f"Missing one-sentence Markdown description for cell: {title}")
        ast.parse(source, filename=f"cell-{index}-{key}.py")
        code_cells.append((key, source))
        slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
        notebook_cells.extend([
            {"cell_type": "markdown", "id": f"md-{index}-{slug}",
             "metadata": {}, "source": [f"## {index}. {key}\n", f"{sentence}\n"]},
            {"cell_type": "code", "execution_count": None, "id": f"code-{index}-{slug}",
             "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)},
        ])
        backup.extend(["# %% [markdown]\n", f"# ## {index}. {key}\n", f"# {sentence}\n\n", "# %%\n", source, "\n\n"])

    # The fourth Python cell must define the worker, and its embedded source must parse.
    assert [key for key, _ in code_cells[:4]] == ["Environment", "Dataset", "Settings", "Worker"]
    worker_tree = ast.parse(code_cells[3][1])
    worker_value = next(ast.literal_eval(node.value) for node in ast.walk(worker_tree)
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "worker" for t in node.targets))
    ast.parse(worker_value, filename="embedded-training-worker.py")

    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ipynb = HERE / "Road_Damage_Concise.ipynb"
    backup_path = HERE / "Road_Damage_Concise.py"
    ipynb.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    backup_path.write_text("".join(backup), encoding="utf-8")
    print(json.dumps({"notebook": str(ipynb), "percent_backup": str(backup_path),
                      "code_cells": len(code_cells), "total_cells": len(notebook_cells),
                      "code_lengths": [len(source) for _, source in code_cells]}, indent=2))


if __name__ == "__main__":
    main()
