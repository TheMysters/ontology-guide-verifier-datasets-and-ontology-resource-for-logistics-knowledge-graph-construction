from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from create_entity_disambiguation_c_reward_benchmark import CASES, FORMATS, format_row


SPLITS = ("train", "val", "test")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_entity_rows() -> list[dict[str, Any]]:
    rows = []
    idx = 1
    variants = [
        ("", ""),
        ("In the mapping file, ", ""),
        ("During ontology validation, ", ""),
        ("For triplet reconstruction, ", ""),
        ("In the interview notes, ", ""),
    ]
    for case in CASES:
        for prefix, suffix in variants:
            variant = dict(case)
            variant["sentence"] = prefix + case["sentence"][0].lower() + case["sentence"][1:] + suffix
            variant["valid_marked"] = prefix + case["valid_marked"][0].lower() + case["valid_marked"][1:] + suffix
            variant["invalid_marked"] = prefix + case["invalid_marked"][0].lower() + case["invalid_marked"][1:] + suffix
            rows.append(format_row(variant, "VALID", variant["valid_marked"], idx))
            idx += 1
            rows.append(format_row(variant, "INVALID", variant["invalid_marked"], idx))
            idx += 1
    return rows


def split_entity_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_relation: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_relation.setdefault(row["candidate_relation_short"], []).append(row)

    splits = {split: [] for split in SPLITS}
    for relation, relation_rows in sorted(by_relation.items()):
        pairs = []
        for start in range(0, len(relation_rows), 2):
            pairs.append(relation_rows[start : start + 2])
        rng.shuffle(pairs)
        n = len(pairs)
        train_n = max(1, int(round(n * 0.70)))
        val_n = max(1, int(round(n * 0.15))) if n >= 3 else 0
        if train_n + val_n >= n and n >= 3:
            train_n = n - 2
            val_n = 1
        split_pairs = {
            "train": pairs[:train_n],
            "val": pairs[train_n : train_n + val_n],
            "test": pairs[train_n + val_n :],
        }
        for split, grouped in split_pairs.items():
            for pair in grouped:
                splits[split].extend(pair)
    for split in SPLITS:
        rng.shuffle(splits[split])
    return splits


def repeat_to_size(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label = {
        "VALID": [row for row in rows if row["label"] == "VALID"],
        "INVALID": [row for row in rows if row["label"] == "INVALID"],
    }
    selected = []
    for label, target_label in [("VALID", target // 2), ("INVALID", target - target // 2)]:
        pool = by_label[label]
        if not pool:
            continue
        for idx in range(target_label):
            row = dict(pool[idx % len(pool)])
            row["v4_entity_repeat_index"] = idx // len(pool)
            selected.append(row)
    rng.shuffle(selected)
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "unique_text_label_pairs": len({(row["text"], row["label"]) for row in rows}),
        "label_distribution": dict(Counter(row["label"] for row in rows)),
        "source_family_distribution": dict(Counter(row.get("source_family", "unknown") for row in rows)),
        "source_type_distribution": dict(Counter(row.get("source_type", "unknown") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create v4 dataset that rewards C entity markers.")
    parser.add_argument("--base-root", type=Path, default=Path("ontology-improvements/generated/balanced_mixed_verifier_v3"))
    parser.add_argument("--output-root", type=Path, default=Path("ontology-improvements/generated/balanced_mixed_verifier_v4_c_rewarded"))
    parser.add_argument("--entity-train-size", type=int, default=600)
    parser.add_argument("--entity-val-size", type=int, default=200)
    parser.add_argument("--entity-test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    entity_splits = split_entity_rows(make_entity_rows(), args.seed)
    target_sizes = {
        "train": args.entity_train_size,
        "val": args.entity_val_size,
        "test": args.entity_test_size,
    }
    stats: dict[str, Any] = {
        "dataset": "balanced_mixed_verifier_v4_c_rewarded",
        "base_root": str(args.base_root),
        "output_root": str(args.output_root),
        "entity_target_sizes": target_sizes,
        "method": [
            "Start from balanced_mixed_verifier_v3.",
            "Append entity-disambiguation pairs where relation-only inputs are ambiguous.",
            "For A/B, positive and negative pairs often share the same relation and sentence.",
            "For C/D/F, entity markers expose which spans are candidate subject/object evidence.",
            "The intent is to reward triplet reconstruction, not relation keyword detection only.",
        ],
        "format_stats": {},
    }

    for format_name in FORMATS:
        all_rows = []
        format_stats = {}
        for split in SPLITS:
            base_rows = load_jsonl(args.base_root / format_name / f"{split}.jsonl")
            entity_rows = []
            for row in repeat_to_size(entity_splits[split], target_sizes[split], args.seed + len(split)):
                entity_rows.append(
                    {
                        **row,
                        "format": format_name,
                        "text": row["input_formats"][format_name],
                        "source_family": "entity_disambiguation",
                        "source_dataset": "entity_disambiguation_c_reward",
                    }
                )
            merged = base_rows + entity_rows
            random.Random(args.seed).shuffle(merged)
            write_jsonl(args.output_root / format_name / f"{split}.jsonl", merged)
            all_rows.extend(merged)
            format_stats[split] = {
                "base": summarize(base_rows),
                "entity_added": summarize(entity_rows),
                "merged": summarize(merged),
            }
        write_jsonl(args.output_root / format_name / "all.jsonl", all_rows)
        write_jsonl(args.output_root / format_name / "eval.jsonl", all_rows)
        format_stats["all"] = summarize(all_rows)
        stats["format_stats"][format_name] = format_stats

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "README.md").write_text(
        "# Balanced Mixed Verifier v4 C-Rewarded\n\n"
        "This dataset extends v3 with entity-disambiguation hard cases. It is designed to reward C-style "
        "relation+entity-marker inputs for triplet reconstruction.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
