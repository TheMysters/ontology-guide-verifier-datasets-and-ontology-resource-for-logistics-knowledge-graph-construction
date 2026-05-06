from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


RELATION_SPECS: list[dict[str, Any]] = [
    {
        "relation": "hasProductReference",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "The distributor reference is the product reference used to identify the item in stock.",
            "The complete reference combines the supplier code and the manufacturer reference to avoid duplicates.",
            "Without the product reference, different air filters can share the same description.",
            "The reference is a unique identifier for the product.",
            "The article reference must be preserved even when the designation is available.",
        ],
        "hard_negatives": [
            "The product description alone is only a reminder and does not properly identify the product.",
            "The sentence mentions a reference example but does not define the product reference field.",
        ],
    },
    {
        "relation": "hasDesignation",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "The second column is the product description, which acts as the product designation.",
            "The designation says air filter, but it does not say which vehicle the product fits.",
            "The commercial description should be stored as the product designation.",
            "A limited product description can still be kept as a designation.",
        ],
        "hard_negatives": [
            "The air filter example shows why designation is not enough to identify the product.",
            "The family code is more precise than the wording in the designation.",
        ],
    },
    {
        "relation": "hasSupplierCode",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "The first three characters of the distributor reference are the supplier code.",
            "ABX, PUR, and DUN are examples of supplier codes used by distributors.",
            "The supplier code is combined with the manufacturer reference in a complete reference.",
            "Procurement keeps the supplier code next to the product reference.",
        ],
        "hard_negatives": [
            "The supplier name can appear in sourcing notes without giving a supplier code.",
            "The distributor is mentioned, but no supplier code value is provided.",
        ],
    },
    {
        "relation": "hasManufacturerReference",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "The remaining characters correspond to the manufacturer's national reference.",
            "A complete reference includes the manufacturer reference after the supplier code.",
            "The manufacturer reference prevents duplicates when suppliers reuse six-digit codes.",
            "The OEM part number should be handled as the manufacturer reference.",
        ],
        "hard_negatives": [
            "The manufacturer is mentioned as a company, but no manufacturer reference is stated.",
            "Manufacturer wording in a note should not be confused with a manufacturer reference value.",
        ],
    },
    {
        "relation": "belongsToFamily",
        "subject": "Product",
        "object": "ProductFamily",
        "positives": [
            "The product family groups products into broad distributor-specific categories.",
            "Family 50 is more engine-related.",
            "Family 54 covers external peripheral elements such as tires and fenders.",
            "The family represents the functional area where the part is located in the vehicle.",
            "The product should be connected to the product family when the family code is present.",
        ],
        "hard_negatives": [
            "The interview compares families, but this sentence does not assign a product to a family.",
            "The word family appears as an example of vocabulary only.",
        ],
    },
    {
        "relation": "belongsToSubfamily",
        "subject": "Product",
        "object": "ProductSubfamily",
        "positives": [
            "The subfamily gives more detail inside the product family.",
            "Subfamily 13 specifically refers to passenger car brake discs.",
            "The subfamily represents the specific type of part within a functional area.",
            "The product should belong to a product subfamily when the subfamily code is present.",
        ],
        "hard_negatives": [
            "The same subfamily number can appear under different families without meaning the same thing.",
            "The sentence discusses subfamily logic but does not assign a product to one subfamily.",
        ],
    },
    {
        "relation": "hasCurrentStockQuantity",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "The current stock quantity is based on the Colmar site.",
            "Stock quantity is a variable value that changes every day.",
            "The stock-on-hand column should be mapped to current stock quantity.",
            "Operations update the current stock quantity after warehouse movements.",
        ],
        "hard_negatives": [
            "The sentence discusses stock policy without giving a current stock quantity.",
            "Storage constraints are mentioned, but no current stock quantity is provided.",
        ],
    },
    {
        "relation": "annualDemand",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "Standard annual sales should be represented as annual demand.",
            "Annual sales are linked to ABC classification.",
            "The demand column stores annual demand for each product.",
            "Calendar-year sales can be used as annual demand when rolling sales are unavailable.",
        ],
        "hard_negatives": [
            "Sales are mentioned, but the sentence says the annual value is not accurate enough.",
            "The interview discusses ABC classification without giving annual demand.",
        ],
    },
    {
        "relation": "hasRollingAnnualDemand",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "Rolling annual sales give the most accurate view of consumption.",
            "The file should use rolling annual sales instead of only 2025 sales.",
            "Annual sales in this case are calculated on a rolling basis.",
            "The last twelve months of consumption should be mapped to rolling annual demand.",
        ],
        "hard_negatives": [
            "The report mentions sales but does not state a rolling annual demand value.",
            "Rolling logic is discussed as a method, not as a product value.",
        ],
    },
    {
        "relation": "hasPurchaseUnit",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "Purchase packaging describes whether the product is bought in packs of five or ten.",
            "Brake discs are bought in pairs, so the purchase packaging affects the weight.",
            "The purchase unit can differ from the sales unit.",
            "The procurement quantity should be stored as the purchase unit.",
        ],
        "hard_negatives": [
            "The supplier is mentioned near purchase packaging but should not create a supplied-by relation.",
            "The sentence asks about packaging level but does not give a purchase unit.",
        ],
    },
    {
        "relation": "hasSalesUnit",
        "subject": "Product",
        "object": "LiteralValue",
        "positives": [
            "Windshield wipers can be bought in packs but sold individually.",
            "Sales packaging can be different from purchase packaging.",
            "The sales data needs to be adjusted according to the sales packaging.",
            "The customer-facing packaging quantity should be represented as the sales unit.",
        ],
        "hard_negatives": [
            "The text mentions sales but not the sales packaging quantity.",
            "A sales discussion should not automatically create a sales unit.",
        ],
    },
    {
        "relation": "hasDefaultUnit",
        "subject": "DimensionMeasurement",
        "object": "Millimetre",
        "positives": [
            "Length, width, and height are given in millimeters.",
            "Brake disc package dimensions are recorded in millimeters.",
            "The tire thickness is 185 mm, so the dimension unit is millimetre.",
            "Dimension measurements should use millimetres when the client confirms that unit.",
        ],
        "hard_negatives": [
            "The sentence mentions dimensions but does not state the unit.",
            "Dimensions are discussed as a concept without assigning millimetres.",
        ],
    },
    {
        "relation": "hasDefaultUnit",
        "subject": "DimensionMeasurement",
        "object": "Centimetre",
        "positives": [
            "For dimensions, the client may provide values in centimeters.",
            "Some dimension files use centimeters instead of millimeters.",
            "If the client validates centimeters, the dimension default unit should be centimetre.",
        ],
        "hard_negatives": [
            "Centimeters are mentioned as a possible unit but not selected for the current file.",
        ],
    },
    {
        "relation": "hasDefaultUnit",
        "subject": "WeightMeasurement",
        "object": "Kilogram",
        "positives": [
            "The listed weight of 7.7 kg is based on the purchase packaging.",
            "Weight can be recorded in kilograms.",
            "The product weight should use kilogram when the source value is in kg.",
            "The default unit for weight measurement is kilogram in this file.",
        ],
        "hard_negatives": [
            "Weight is discussed, but the unit is not specified.",
            "The sentence mentions kilograms and grams only as alternatives.",
        ],
    },
    {
        "relation": "hasDefaultUnit",
        "subject": "WeightMeasurement",
        "object": "Gram",
        "positives": [
            "For small parts, weight can be provided in grams.",
            "If the source weight is in grams, the weight measurement unit should be gram.",
        ],
        "hard_negatives": [
            "The interview says grams are possible, but this product is not assigned a gram value.",
        ],
    },
    {
        "relation": "suppliedBy",
        "subject": "Product",
        "object": "Supplier",
        "positives": [
            "The same product can be bought from different suppliers.",
            "A Bosch brake pad can be purchased from a national platform or a local supplier.",
            "The supplier field identifies the company that supplies the product.",
            "Sourcing strategy determines which supplier provides the product.",
        ],
        "hard_negatives": [
            "Supplier code appears in the reference but does not identify the supplier company.",
            "Supplier wording in a product designation should not create supplied-by.",
        ],
    },
    {
        "relation": "manufacturedBy",
        "subject": "Product",
        "object": "Manufacturer",
        "positives": [
            "Golda bridges manufacturers and distributors.",
            "A Bosch brake pad should link the product to Bosch as manufacturer when Bosch produced it.",
            "The manufacturer field identifies the company that manufactured the product.",
        ],
        "hard_negatives": [
            "The manufacturer reference is a code, not a manufactured-by company relation.",
            "A distributor can supply a product without manufacturing it.",
        ],
    },
]

COMBO_SENTENCES = [
    ("The complete reference combines supplier code and manufacturer reference.", [
        {"subject": "Product", "predicate": "hasSupplierCode", "object": "LiteralValue"},
        {"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"},
        {"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"},
    ]),
    ("Family and subfamily must be interpreted together to identify brake discs.", [
        {"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"},
        {"subject": "Product", "predicate": "belongsToSubfamily", "object": "ProductSubfamily"},
    ]),
    ("Annual sales and rolling annual sales are two different demand views.", [
        {"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"},
        {"subject": "Product", "predicate": "hasRollingAnnualDemand", "object": "LiteralValue"},
    ]),
    ("Purchase packaging and sales packaging can be different for the same product.", [
        {"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"},
        {"subject": "Product", "predicate": "hasSalesUnit", "object": "LiteralValue"},
    ]),
    ("Dimensions may be in millimeters while weight is recorded in kilograms.", [
        {"subject": "DimensionMeasurement", "predicate": "hasDefaultUnit", "object": "Millimetre"},
        {"subject": "WeightMeasurement", "predicate": "hasDefaultUnit", "object": "Kilogram"},
    ]),
]


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "ontology-improvements" / "scripts").exists() and (candidate / "nlp-detection").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, sentences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def triple_key(sentence_id: int, triple: dict[str, str]) -> tuple[int, str, str, str]:
    return sentence_id, triple["subject"], triple["predicate"], triple["object"]


def candidate_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return int(row["sentence_id"]), row["subject_short"], row["candidate_relation_short"], row["object_short"]


def load_interview_sentences(evaluator, input_dir: Path) -> list[str]:
    sentences: list[str] = []
    for path in sorted(input_dir.glob("*.txt")):
        text = evaluator.weak.repair_mojibake(path.read_text(encoding="utf-8", errors="replace"))
        sentences.extend(evaluator.weak.sentence_split(text))
    return sentences


def sentence_specs_from_relations() -> list[tuple[str, list[dict[str, str]], str]]:
    specs: list[tuple[str, list[dict[str, str]], str]] = []
    for spec in RELATION_SPECS:
        gold = [{"subject": spec["subject"], "predicate": spec["relation"], "object": spec["object"]}]
        for sentence in spec["positives"]:
            specs.append((sentence, gold, "interview_paraphrase_positive"))
        for sentence in spec["hard_negatives"]:
            specs.append((sentence, [], "interview_style_hard_negative"))
    for sentence, triples in COMBO_SENTENCES:
        specs.append((sentence, triples, "interview_paraphrase_multi_relation"))
    return specs


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    sentence_ids = sorted({int(row["sentence_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(sentence_ids)
    n = len(sentence_ids)
    train_ids = set(sentence_ids[: int(n * 0.7)])
    val_ids = set(sentence_ids[int(n * 0.7) : int(n * 0.85)])
    test_ids = set(sentence_ids[int(n * 0.85) :])
    return {
        "train": [row for row in rows if int(row["sentence_id"]) in train_ids],
        "val": [row for row in rows if int(row["sentence_id"]) in val_ids],
        "test": [row for row in rows if int(row["sentence_id"]) in test_ids],
        "all": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an interview-augmented ontology verifier training dataset.")
    parser.add_argument("--output-root", type=Path, default=Path("ontology-improvements/generated/interview_augmented_verifier"))
    parser.add_argument("--interview-dir", type=Path, default=Path("nlp-detection/data-input"))
    parser.add_argument("--owl", type=Path, default=Path("ontology-improvements/ontology-improvements/internal-logistics-v2.owl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    scripts_dir = repo_root / "ontology-improvements" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import end_to_end_verifier_evaluation as evaluator

    evaluator.OWL_PATH = args.owl
    evaluator.formatter.OWL_PATH = args.owl

    real_interview_sentences = load_interview_sentences(evaluator, args.interview_dir)
    specs = sentence_specs_from_relations()
    # Add selected exact interview sentences as unlabeled hard-negative/background examples.
    for sentence in real_interview_sentences:
        if any(token in sentence.lower() for token in ["reference", "family", "subfamily", "stock", "sales", "weight", "dimension", "supplier", "manufacturer", "packaging"]):
            specs.append((sentence, [], "real_interview_background"))

    deduped_specs = []
    seen_sentences = set()
    for sentence, triples, source_type in specs:
        normalized = " ".join(sentence.lower().split())
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        deduped_specs.append((sentence, triples, source_type))

    sentences = [sentence for sentence, _, _ in deduped_specs]
    gold_rows = [
        {"sentence_id": idx, "sentence": sentence, "gold_triples": triples, "source_type": source_type}
        for idx, (sentence, triples, source_type) in enumerate(deduped_specs, start=1)
    ]

    text_path = args.output_root / "interview_augmented_sentences.txt"
    gold_path = args.output_root / "interview_augmented_gold.jsonl"
    write_text(text_path, sentences)
    write_jsonl(gold_path, gold_rows)

    raw_candidates = evaluator.build_candidate_rows(text_path)
    candidate_rows = evaluator.format_candidate_rows(raw_candidates)
    source_type_by_sentence = {row["sentence_id"]: row["source_type"] for row in gold_rows}
    gold = {
        triple_key(idx, triple)
        for idx, (_, triples, _) in enumerate(deduped_specs, start=1)
        for triple in triples
    }

    annotated_rows = []
    candidate_keys = {candidate_key(row) for row in candidate_rows}
    for row in candidate_rows:
        label = "VALID" if candidate_key(row) in gold else "INVALID"
        annotated_rows.append(
            {
                **row,
                "label": label,
                "manual_gold_label": label,
                "source_sentence": row["sentence"],
                "source_type": source_type_by_sentence.get(int(row["sentence_id"]), "unknown"),
            }
        )

    rows_by_split = split_rows(annotated_rows, args.seed)
    formats = [
        "A_sentence_only",
        "B_relation_marker",
        "C_relation_entity_markers",
        "D_ontology_context",
        "F_full_ontology_context",
    ]
    for format_name in formats:
        for split, rows in rows_by_split.items():
            format_rows = [
                {
                    **row,
                    "text": row["input_formats"][format_name],
                    "format": format_name,
                    "split": split,
                }
                for row in rows
            ]
            write_jsonl(args.output_root / format_name / f"{split}.jsonl", format_rows)
            if split == "all":
                write_jsonl(args.output_root / format_name / "eval.jsonl", format_rows)

    covered_gold = gold & candidate_keys
    missing_gold = gold - candidate_keys
    stats = {
        "dataset": "interview_augmented_verifier",
        "interview_dir": str(args.interview_dir),
        "output_root": str(args.output_root),
        "sentences": len(deduped_specs),
        "real_interview_sentences_loaded": len(real_interview_sentences),
        "candidate_examples": len(annotated_rows),
        "gold_triples": len(gold),
        "gold_triples_covered_by_candidates": len(covered_gold),
        "gold_candidate_coverage": len(covered_gold) / len(gold) if gold else 0.0,
        "gold_triples_missing_from_candidates": len(missing_gold),
        "label_distribution": dict(Counter(row["label"] for row in annotated_rows)),
        "source_type_distribution": dict(Counter(row["source_type"] for row in annotated_rows)),
        "candidate_relation_distribution": dict(Counter(row["candidate_relation_short"] for row in annotated_rows)),
        "split_distribution": {split: len(rows) for split, rows in rows_by_split.items()},
        "missing_gold_triples": [list(item) for item in sorted(missing_gold)],
        "formats": formats,
        "strengthening_method": [
            "Exact interview sentences from nlp-detection/data-input are included as real-domain background and hard negatives.",
            "Positive paraphrases are derived from interview concepts such as distributor reference, supplier code, manufacturer reference, family/subfamily, stock, sales, packaging, dimensions, and units.",
            "Hard negatives include nearby trigger words where the ontology relation should not be asserted.",
            "Multi-relation examples reproduce interview statements where several columns must be interpreted together.",
            "All rows are converted to the same A/B/C/D/F input formats used by the existing BERT-like verifier models.",
            "Splits are sentence-grouped to reduce leakage between train, validation, and test.",
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_root / "README.md").write_text(
        "# Interview-Augmented Ontology Verifier Dataset\n\n"
        "This dataset is designed for retraining the A/B/C/D/F verifier models before evaluating on the natural 100-sentence benchmark.\n\n"
        "It combines real interview vocabulary from `nlp-detection/data-input` with controlled ontology-aware paraphrases and hard negatives.\n\n"
        "Use the format directories directly with `train_ontology_format_verifier.py`, for example:\n\n"
        "```powershell\n"
        "python ontology-improvements/scripts/train_ontology_format_verifier.py --format C_relation_entity_markers --data-root ontology-improvements/generated/interview_augmented_verifier --epochs 5 --batch-size 16\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
