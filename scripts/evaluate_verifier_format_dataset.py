from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    script_root = Path(__file__).resolve().parents[1]
    candidates = [script_root, start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if (candidate / "scripts").exists() and (candidate / "datasets").exists() and (candidate / "ontology").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    repo_root = find_repo_root(Path.cwd())
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from end_to_end_verifier_evaluation import compute_binary_metrics, predict_texts

    parser = argparse.ArgumentParser(description="Evaluate a trained verifier model on a format-specific JSONL file.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    predictions = predict_texts(
        args.model_dir,
        [row["text"] for row in rows],
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    metrics = compute_binary_metrics(
        [row["label"] for row in rows],
        [prediction["prediction"] for prediction in predictions],
    )
    errors = []
    enriched = []
    for row, prediction in zip(rows, predictions):
        item = {**row, **prediction}
        enriched.append(item)
        if row["label"] != prediction["prediction"]:
            errors.append(
                {
                    "sentence_id": row.get("sentence_id"),
                    "candidate_id": row.get("candidate_id"),
                    "gold": row["label"],
                    "prediction": prediction["prediction"],
                    "score_valid": prediction.get("score_valid"),
                    "subject": row.get("subject_short"),
                    "predicate": row.get("candidate_relation_short"),
                    "object": row.get("object_short"),
                    "sentence": row.get("sentence") or row.get("source_sentence"),
                }
            )

    payload = {
        "dataset": str(args.dataset),
        "model_dir": str(args.model_dir),
        **metrics,
        "error_count": len(errors),
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions_path = args.output_json.with_name(args.output_json.stem + "_predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in enriched:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({key: payload[key] for key in ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn", "support", "error_count"]}, indent=2))
    print(f"saved {args.output_json}")
    print(f"saved {predictions_path}")


if __name__ == "__main__":
    main()
