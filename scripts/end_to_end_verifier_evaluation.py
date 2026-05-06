from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

WEAK_PIPELINE_PATH = SCRIPT_DIR / "build_logistics_weak_dataset.py"
FORMATTER_PATH = SCRIPT_DIR / "generate_ontology_aware_verifier_dataset.py"

OWL_PATH = ROOT / "ontology" / "internal-logistics-v2.owl"
DEFAULT_INPUT_TEXT = ROOT / "datasets" / "evaluation_only_interview_grounded_dataset" / "natural_logistics_benchmark_100_sentences.txt"
DEFAULT_GOLD_JSONL = ROOT / "datasets" / "evaluation_only_interview_grounded_dataset" / "natural_logistics_benchmark_100_gold.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "end_to_end_eval"

PRIMARY_FORMATS = [
    "A_sentence_only",
    "B_relation_marker",
    "C_relation_entity_markers",
    "D_ontology_context",
    "F_full_ontology_context",
]
DIAGNOSTIC_FORMATS = [
    "E_validation_context",
    "G_symbolic_triple_text",
]
VALID_PROXY_FLAGS = {"STRUCTURALLY_VALID", "SCHEMA_DEFAULT_UNIT_COMPATIBLE", "UNIT_COMPATIBLE"}
INVALID_PROXY_FLAGS = {"STRUCTURALLY_INVALID", "UNIT_INCOMPATIBLE"}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_uri(value: str) -> str:
    if value == "LiteralValue":
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("#"):
        return f"{formatter.IL_NS}{value[1:]}"
    return value


weak = load_module(WEAK_PIPELINE_PATH, "logistics_weak_pipeline")
formatter = load_module(FORMATTER_PATH, "ontology_aware_formatter")
formatter.OWL_PATH = OWL_PATH


def candidate_to_row(sentence: str, marked_text: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return formatter.normalize_schema_unit_row(
        {
            "text": f"[REL] {candidate['short']} [/REL] {marked_text}",
            "label": "INVALID",
            "candidate_relation": full_uri(candidate["uri"]),
            "subject": full_uri(candidate["subject_uri"]),
            "object": full_uri(candidate["object_uri"]),
            "sentence": sentence,
        }
    )


def build_candidate_rows(input_text_path: Path = DEFAULT_INPUT_TEXT) -> list[dict[str, Any]]:
    ontology_for_candidates = weak.parse_ontology(OWL_PATH)
    text = weak.repair_mojibake(input_text_path.read_text(encoding="utf-8", errors="replace"))
    return build_candidate_rows_from_text(text)


def build_candidate_rows_from_text(text: str) -> list[dict[str, Any]]:
    ontology_for_candidates = weak.parse_ontology(OWL_PATH)
    text = weak.repair_mojibake(text)
    rows: list[dict[str, Any]] = []
    for sentence_id, sentence in enumerate(weak.sentence_split(text), start=1):
        mentions = weak.detect_mentions(sentence, ontology_for_candidates)
        candidates, support_pairs = weak.infer_candidate_relations(mentions, ontology_for_candidates)

        support_by_relation: dict[str, list[dict[str, Any]]] = {}
        for pair in support_pairs:
            support_by_relation.setdefault(pair["relation_uri"], []).append(pair)

        for candidate_index, candidate in enumerate(candidates, start=1):
            supports = support_by_relation.get(candidate["uri"], [])
            tagged_mentions = []
            if supports:
                tagged_mentions.append(supports[0]["trigger"])
                if supports[0].get("unit") is not None:
                    tagged_mentions.append(supports[0]["unit"])
            elif mentions:
                tagged_mentions = mentions[:2]

            row = candidate_to_row(sentence, weak.mark_text(sentence, tagged_mentions), candidate)
            rows.append(
                {
                    **row,
                    "sentence_id": sentence_id,
                    "candidate_id": f"s{sentence_id:03d}_c{candidate_index:03d}",
                    "candidate_source": candidate.get("source", "unknown"),
                    "candidate_relation_short": formatter.short(row["candidate_relation"]),
                    "subject_short": formatter.short(row["subject"]),
                    "object_short": formatter.short(row["object"]),
                    "mentions": mentions,
                }
            )
    return rows


def candidate_rows_to_sentence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        sentence_id = int(row["sentence_id"])
        if sentence_id not in grouped:
            grouped[sentence_id] = {
                "sentence_id": sentence_id,
                "sentence": row["sentence"],
                "candidate_count": 0,
                "candidate_relations": [],
            }
        grouped[sentence_id]["candidate_count"] += 1
        grouped[sentence_id]["candidate_relations"].append(row["candidate_relation_short"])

    summaries = []
    for item in grouped.values():
        rels = item["candidate_relations"]
        summaries.append(
            {
                **item,
                "candidate_relations": rels,
                "candidate_relations_text": ", ".join(rels),
            }
        )
    return sorted(summaries, key=lambda x: x["sentence_id"])


def enrich_rows_with_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, **pred} for row, pred in zip(rows, predictions)]


def accepted_ratio(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    total = len(rows)
    if total == 0:
        return 0.0
    accepted = sum(1 for pred in predictions if pred["prediction"] == "VALID")
    return accepted / total


def format_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ontology = formatter.parse_ontology(formatter.OWL_PATH)
    formatted = []
    for row in rows:
        enriched = formatter.enrich_row(row, ontology, split="end_to_end")
        formatted.append(enriched)
    return formatted


def prepare_inputs(format_name: str, rows: list[dict[str, Any]]) -> list[str]:
    return [row["input_formats"][format_name] for row in rows]


def load_transformer_model(model_dir: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_texts(model_dir: Path, texts: list[str], batch_size: int = 32, max_len: int = 192) -> list[dict[str, Any]]:
    import torch

    tokenizer, model, device = load_transformer_model(model_dir)
    predictions: list[dict[str, Any]] = []
    id2label = getattr(model.config, "id2label", None) or {0: "INVALID", 1: "VALID"}
    id2label = {int(k): v for k, v in id2label.items()}

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).detach().cpu()
            pred_ids = probs.argmax(dim=1).tolist()
        for pred_id, prob in zip(pred_ids, probs.tolist()):
            predictions.append(
                {
                    "prediction": id2label.get(pred_id, str(pred_id)),
                    "score_valid": float(prob[1]) if len(prob) > 1 else None,
                    "score_invalid": float(prob[0]) if prob else None,
                }
            )
    return predictions


def triples_from_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = []
    for row, pred in zip(rows, predictions):
        if pred["prediction"] != "VALID":
            continue
        accepted.append(
            {
                "sentence_id": row["sentence_id"],
                "candidate_id": row["candidate_id"],
                "subject": row["subject_short"],
                "predicate": row["candidate_relation_short"],
                "object": row["object_short"],
                "sentence": row["sentence"],
                "score_valid": pred["score_valid"],
                "domain_range_valid": row["domain_range_valid"],
                "unit_compatible": row["unit_compatible"],
                "ontology_validation_flag": row["ontology_validation_flag"],
            }
        )
    return accepted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ttl(path: Path, triples: list[dict[str, Any]]) -> None:
    lines = ["@prefix : <http://example.org/internal-logistics#> .", ""]
    for triple in triples:
        subj = triple["subject"]
        pred = triple["predicate"]
        obj = triple["object"]
        if obj == "LiteralValue":
            obj_token = '"LiteralValue"'
        else:
            obj_token = f":{obj}"
        lines.append(f":{subj} :{pred} {obj_token} .")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def canonical_triple(triple: Any) -> tuple[int | None, str, str, str]:
    if isinstance(triple, dict):
        sentence_id = triple.get("sentence_id")
        subject = triple.get("subject")
        predicate = triple.get("predicate") or triple.get("relation")
        obj = triple.get("object")
    else:
        sentence_id = None
        subject, predicate, obj = triple
    if sentence_id is not None:
        sentence_id = int(sentence_id)
    return sentence_id, str(subject), str(predicate), str(obj)


def load_gold_triples(path: Path = DEFAULT_GOLD_JSONL) -> set[tuple[int | None, str, str, str]]:
    if not path.exists():
        return set()
    gold: set[tuple[int | None, str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sentence_id = int(row["sentence_id"])
            for triple in row.get("gold_triples", []):
                _sid, subject, predicate, obj = canonical_triple(triple)
                gold.add((sentence_id, subject, predicate, obj))
    return gold


def gold_end_to_end_metrics(accepted_triples: list[dict[str, Any]], gold_path: Path = DEFAULT_GOLD_JSONL) -> dict[str, Any]:
    gold = load_gold_triples(gold_path)
    predicted = {
        (int(triple["sentence_id"]), triple["subject"], triple["predicate"], triple["object"])
        for triple in accepted_triples
    }
    tp_items = predicted & gold
    fp_items = predicted - gold
    fn_items = gold - predicted
    precision = len(tp_items) / len(predicted) if predicted else 0.0
    recall = len(tp_items) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "gold_file": str(gold_path),
        "gold_available": bool(gold),
        "gold_triples": len(gold),
        "predicted_triples": len(predicted),
        "gold_tp": len(tp_items),
        "gold_fp": len(fp_items),
        "gold_fn": len(fn_items),
        "gold_precision": precision,
        "gold_recall": recall,
        "gold_f1": f1,
        "gold_true_positives": [list(item) for item in sorted(tp_items)],
        "gold_false_positives": [list(item) for item in sorted(fp_items)],
        "gold_false_negatives": [list(item) for item in sorted(fn_items)],
        "metric_reference": "manual gold triples" if gold else "gold file empty or unavailable",
    }


def proxy_gold_label(row: dict[str, Any]) -> str:
    flag = row.get("ontology_validation_flag")
    if flag in VALID_PROXY_FLAGS:
        return "VALID"
    if flag in INVALID_PROXY_FLAGS:
        return "INVALID"
    return "INVALID"


def compute_binary_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    tp = sum(1 for g, p in zip(gold, pred) if g == "VALID" and p == "VALID")
    tn = sum(1 for g, p in zip(gold, pred) if g == "INVALID" and p == "INVALID")
    fp = sum(1 for g, p in zip(gold, pred) if g == "INVALID" and p == "VALID")
    fn = sum(1 for g, p in zip(gold, pred) if g == "VALID" and p == "INVALID")
    total = len(gold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": total,
    }


def proxy_metrics(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [proxy_gold_label(row) for row in rows]
    pred = [prediction["prediction"] for prediction in predictions]
    metrics = compute_binary_metrics(gold, pred)
    return {f"proxy_{key}": value for key, value in metrics.items()}


def chunk_triples_by_sentence(
    triples: list[dict[str, Any]],
    chunk_size: int = 25,
    max_sentence_id: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    chunks: dict[str, list[dict[str, Any]]] = {}
    for start in range(1, max_sentence_id + 1, chunk_size):
        end = min(start + chunk_size - 1, max_sentence_id)
        key = f"sentences_{start:03d}_{end:03d}"
        chunks[key] = [
            triple for triple in triples if start <= int(triple["sentence_id"]) <= end
        ]
    return chunks


def write_graph_html(path: Path, triples: list[dict[str, Any]], title: str) -> None:
    nodes = []
    seen = set()
    for triple in triples:
        for node in (triple["subject"], triple["object"]):
            if node not in seen:
                seen.add(node)
                nodes.append(node)
    edges = [
        (triple["subject"], triple["predicate"], triple["object"], triple.get("score_valid"))
        for triple in triples
    ]
    rows = "\n".join(
        "<tr>"
        f"<td>{triple['sentence_id']}</td>"
        f"<td>{triple['subject']}</td>"
        f"<td>{triple['predicate']}</td>"
        f"<td>{triple['object']}</td>"
        f"<td>{triple.get('score_valid', '')}</td>"
        f"<td>{triple['sentence']}</td>"
        "</tr>"
        for triple in triples
    )
    node_items = "\n".join(f"<li>{node}</li>" for node in nodes)
    edge_items = "\n".join(
        f"<li>{subject} -- <strong>{predicate}</strong> --&gt; {obj}"
        + (f" (score={score:.3f})" if isinstance(score, float) else "")
        + "</li>"
        for subject, predicate, obj, score in edges
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.35; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f2f2f2; }}
    .grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 24px; }}
    code {{ background: #f6f6f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Accepted triples: <strong>{len(triples)}</strong>. Nodes: <strong>{len(nodes)}</strong>.</p>
  <div class="grid">
    <section>
      <h2>Nodes</h2>
      <ul>{node_items}</ul>
    </section>
    <section>
      <h2>Edges</h2>
      <ul>{edge_items}</ul>
    </section>
  </div>
  <h2>Triples</h2>
  <table>
    <thead>
      <tr><th>Sentence</th><th>Subject</th><th>Predicate</th><th>Object</th><th>Score VALID</th><th>Source sentence</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_chunked_kg_exports(format_dir: Path, triples: list[dict[str, Any]], max_sentence_id: int = 100) -> dict[str, Any]:
    chunks = chunk_triples_by_sentence(triples, chunk_size=25, max_sentence_id=max_sentence_id)
    chunk_summary = {}
    for chunk_name, chunk_triples in chunks.items():
        chunk_dir = format_dir / "kg_chunks" / chunk_name
        write_jsonl(chunk_dir / "accepted_triples.jsonl", chunk_triples)
        write_csv(chunk_dir / "accepted_triples.csv", chunk_triples)
        write_ttl(chunk_dir / "accepted_triples.ttl", chunk_triples)
        write_graph_html(chunk_dir / "graph.html", chunk_triples, f"{format_dir.name} - {chunk_name}")
        chunk_summary[chunk_name] = {
            "accepted_triples": len(chunk_triples),
            "ttl": str(chunk_dir / "accepted_triples.ttl"),
            "html": str(chunk_dir / "graph.html"),
        }
    (format_dir / "kg_chunks" / "chunk_summary.json").write_text(
        json.dumps(chunk_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return chunk_summary


def summarize_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = triples_from_predictions(rows, predictions)
    structurally_invalid_accepted = [
        triple for triple in accepted if triple["ontology_validation_flag"] in {"STRUCTURALLY_INVALID", "UNIT_INCOMPATIBLE"}
    ]
    return {
        "sentences": len({row["sentence_id"] for row in rows}),
        "candidate_count": len(rows),
        "accepted_triples": len(accepted),
        "rejected_candidates": len(rows) - len(accepted),
        "structurally_invalid_accepted": len(structurally_invalid_accepted),
        "avg_valid_score": (
            sum(pred["score_valid"] for pred in predictions if pred["score_valid"] is not None) / len(predictions)
            if predictions
            else None
        ),
        **proxy_metrics(rows, predictions),
        "metric_reference": "proxy ontology validation labels, not manual gold",
    }


def evaluate_format(
    format_name: str,
    model_dir: Path,
    rows: list[dict[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    max_len: int = 192,
    gold_path: Path = DEFAULT_GOLD_JSONL,
) -> dict[str, Any]:
    texts = prepare_inputs(format_name, rows)
    predictions = predict_texts(model_dir, texts, batch_size=batch_size, max_len=max_len)
    enriched_predictions = [{**row, **pred, "model_format": format_name} for row, pred in zip(rows, predictions)]
    accepted = triples_from_predictions(rows, predictions)
    summary = {
        "format": format_name,
        "model_dir": str(model_dir),
        **summarize_predictions(rows, predictions),
    }
    gold_metrics = gold_end_to_end_metrics(accepted, gold_path)
    summary.update({f"end_to_end_{key}": value for key, value in gold_metrics.items() if not isinstance(value, list)})

    format_dir = output_dir / format_name
    write_jsonl(format_dir / "candidate_predictions.jsonl", enriched_predictions)
    write_csv(format_dir / "accepted_triples.csv", accepted)
    write_jsonl(format_dir / "accepted_triples.jsonl", accepted)
    write_ttl(format_dir / "accepted_triples.ttl", accepted)
    (format_dir / "gold_comparison.json").write_text(
        json.dumps(gold_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chunk_summary = write_chunked_kg_exports(format_dir, accepted)
    summary["kg_chunks"] = chunk_summary
    (format_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_many(
    model_dirs: dict[str, str | Path],
    input_text_path: Path = DEFAULT_INPUT_TEXT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    max_len: int = 192,
    gold_path: Path = DEFAULT_GOLD_JSONL,
) -> list[dict[str, Any]]:
    rows = format_candidate_rows(build_candidate_rows(input_text_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_generated_candidates.jsonl", rows)

    summaries = []
    for format_name, model_dir in model_dirs.items():
        model_path = Path(model_dir)
        if not model_path.exists():
            summaries.append({"format": format_name, "model_dir": str(model_path), "status": "missing_model_dir"})
            continue
        summaries.append(evaluate_format(format_name, model_path, rows, output_dir, batch_size, max_len, gold_path))

    (output_dir / "all_format_summaries.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summaries


def default_model_dirs(model_root: Path = ROOT / "models") -> dict[str, Path]:
    return {
        "A_sentence_only": model_root / "A_sentence_only",
        "B_relation_marker": model_root / "B_relation_marker",
        "C_relation_entity_markers": model_root / "C_relation_entity_markers",
        "D_ontology_context": model_root / "D_ontology_context",
        "F_full_ontology_context": model_root / "F_full_ontology_context",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the released end-to-end ontology-guided evaluation workflow with "
            "locally available trained verifier checkpoints."
        )
    )
    parser.add_argument("--input-text", type=Path, default=DEFAULT_INPUT_TEXT)
    parser.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--formats", nargs="*", default=PRIMARY_FORMATS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = default_model_dirs(args.model_root)
    selected_model_dirs = {name: model_dirs[name] for name in args.formats if name in model_dirs}
    summaries = evaluate_many(
        model_dirs=selected_model_dirs,
        input_text_path=args.input_text,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        gold_path=args.gold_jsonl,
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
