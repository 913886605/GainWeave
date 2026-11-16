"""Read only explicitly paired user files, then create an immutable metric report."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from gainweave.training.data import (create_file, file_sha256, json_bytes,
                                     read_linear_rgb, read_mask, relative_file)
from .metrics import DEFINITION, aggregate_cases, evaluate_case


def run_evaluation(config_path):
    """Evaluate a manifest of already generated predictions; never load a model.

    Config paths are relative to the config file. Paired image/mask paths are
    relative to the manifest. No reference is inferred from a filename and no
    extra source directory is scanned.
    """
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("schema") != "gainweave-eval-v1":
        raise ValueError("Expected gainweave-eval-v1 configuration")
    def resolve(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    manifest_path = resolve(config["manifest"])
    declaration = json.loads(manifest_path.read_text())
    if declaration.get("schema") != "gainweave-paired-evaluation-v1" or declaration.get("color_space") != "linear_rgb":
        raise ValueError("Paired manifest must explicitly declare its schema and linear RGB")
    rows = declaration.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Paired manifest requires a nonempty rows list")
    output = resolve(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output}")
    name = config.get("output_name", "metrics.json")
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("output_name must be a simple .json filename")
    cases, identities, seen = [], {}, set()
    for index, row in enumerate(rows):
        if not isinstance(row.get("source_id"), str) or not row["source_id"].strip():
            raise ValueError("Every pair needs a nonempty source_id")
        if type(row.get("EV")) is not int or row["EV"] not in (-2, -1, 0, 1, 2):
            raise ValueError("Every pair needs an explicit integer EV in [-2,-1,0,1,2]")
        prediction = relative_file(manifest_path.parent, row["prediction"])
        reference = relative_file(manifest_path.parent, row["reference"])
        method = row.get("method", "GainWeave")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a nonempty label")
        view = row.get("view_id", row["reference"])
        key = (row["source_id"], view, row["EV"], method)
        if key in seen:
            raise ValueError("Duplicate source/view/EV/method pair")
        seen.add(key)
        pred, ref = read_linear_rgb(prediction), read_linear_rgb(reference)
        valid = region = None
        files = {"prediction": prediction, "reference": reference}
        for field in ("valid_mask", "region_mask"):
            if row.get(field):
                path = relative_file(manifest_path.parent, row[field])
                files[field] = path
                value = read_mask(path, ref.shape[:2])
                if field == "valid_mask":
                    valid = value
                else:
                    region = value
        for path in files.values():
            identities[str(path.relative_to(manifest_path.parent))] = file_sha256(path)
        cases.append(dict(pair_index=index, source_id=row["source_id"], view_id=view, EV=row["EV"],
                          method=method, regions=evaluate_case(pred, ref, valid, region)))
    result = dict(schema="gainweave-paired-metrics-v1", definition=DEFINITION,
                  manifest_sha256=file_sha256(manifest_path), inputs=identities,
                  cases=cases, aggregates=aggregate_cases(cases),
                  scope="Reference-paired relative HDR errors; no model selection or pass/fail threshold")
    output.mkdir(parents=True, exist_ok=True)
    create_file(output / name, json_bytes(result))
    return dict(report=str(output / name), pairs=len(cases), aggregates=len(result["aggregates"]))
