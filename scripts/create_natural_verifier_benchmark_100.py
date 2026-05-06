from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SENTENCE_SPECS: list[tuple[str, list[dict[str, str]]]] = [
    ("The reference column stores the internal product reference used by warehouse operators.", [{"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"}]),
    ("The article code is the product reference that links purchase records to stock records.", [{"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"}]),
    ("When two labels differ, the product reference remains the stable product identifier.", [{"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"}]),
    ("Legacy files may call the product reference an internal article number.", [{"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"}]),
    ("The barcode is mentioned only as packaging text, not as a product reference.", []),
    ("The designation field contains the product designation shown to purchasing teams.", [{"subject": "Product", "predicate": "hasDesignation", "object": "LiteralValue"}]),
    ("A designation can include supplier wording without changing the product reference.", [{"subject": "Product", "predicate": "hasDesignation", "object": "LiteralValue"}]),
    ("The commercial label should be kept as the product designation.", [{"subject": "Product", "predicate": "hasDesignation", "object": "LiteralValue"}]),
    ("The designation describes the item but does not identify the supplier.", [{"subject": "Product", "predicate": "hasDesignation", "object": "LiteralValue"}]),
    ("The file mentions descriptive text, but no explicit product designation is provided.", []),
    ("The family code groups each product into a broader product family.", [{"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"}]),
    ("Brake pads belong to a product family used for reporting.", [{"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"}]),
    ("The family label is stored separately from the product designation.", [{"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"}]),
    ("A product family should be attached when the file gives the family code.", [{"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"}]),
    ("The word family appears in a supplier comment but no product family is assigned.", []),
    ("The subfamily code refines the product family with a product subfamily.", [{"subject": "Product", "predicate": "belongsToSubfamily", "object": "ProductSubfamily"}]),
    ("A product can belong to a product subfamily even when the family name is abbreviated.", [{"subject": "Product", "predicate": "belongsToSubfamily", "object": "ProductSubfamily"}]),
    ("The subfamily label should create a belongs-to-subfamily relation.", [{"subject": "Product", "predicate": "belongsToSubfamily", "object": "ProductSubfamily"}]),
    ("Subfamily 13 is the specific category below the family code.", [{"subject": "Product", "predicate": "belongsToSubfamily", "object": "ProductSubfamily"}]),
    ("The text compares two subfamilies but does not assign the product to either one.", []),
    ("The purchase unit tells how many pieces are bought from the supplier at once.", [{"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"}]),
    ("A purchase unit of ten means procurement orders the product in packs of ten.", [{"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"}]),
    ("The buying quantity is represented as the purchase unit.", [{"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"}]),
    ("Purchase unit and sales unit are separate commercial quantities.", [{"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"}, {"subject": "Product", "predicate": "hasSalesUnit", "object": "LiteralValue"}]),
    ("The supplier name appears near the purchase unit but should not create a supplied-by relation.", [{"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"}]),
    ("The sales unit tells how many pieces are sold to customers.", [{"subject": "Product", "predicate": "hasSalesUnit", "object": "LiteralValue"}]),
    ("A sales unit of one means the product is sold individually.", [{"subject": "Product", "predicate": "hasSalesUnit", "object": "LiteralValue"}]),
    ("The customer-facing packaging quantity is the sales unit.", [{"subject": "Product", "predicate": "hasSalesUnit", "object": "LiteralValue"}]),
    ("The word unit refers to the sales unit, not to kilograms or millimetres.", [{"subject": "Product", "predicate": "hasSalesUnit", "object": "LiteralValue"}]),
    ("The document says units were reviewed, but no sales unit value is stated.", []),
    ("Current stock quantity records how many pieces are available in the warehouse.", [{"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}]),
    ("The stock-on-hand column corresponds to current stock quantity.", [{"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}]),
    ("Operations update current stock quantity after each movement.", [{"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}]),
    ("A zero in stock means the current stock quantity is zero.", [{"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}]),
    ("The sentence discusses stock policy but gives no current stock quantity.", []),
    ("Annual demand is estimated from the units consumed during the last year.", [{"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"}]),
    ("The demand column stores annual demand for each product.", [{"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"}]),
    ("Planning uses annual demand to size replenishment rules.", [{"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"}]),
    ("The calendar-year demand should be mapped to annual demand.", [{"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"}]),
    ("Demand is mentioned as a concept, but no annual demand value is present.", []),
    ("Rolling annual demand uses the last twelve months instead of the calendar year.", [{"subject": "Product", "predicate": "hasRollingAnnualDemand", "object": "LiteralValue"}]),
    ("The rolling annual demand column smooths seasonal peaks.", [{"subject": "Product", "predicate": "hasRollingAnnualDemand", "object": "LiteralValue"}]),
    ("A product may have both annual demand and rolling annual demand.", [{"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"}, {"subject": "Product", "predicate": "hasRollingAnnualDemand", "object": "LiteralValue"}]),
    ("The last twelve months consumption is the rolling annual demand.", [{"subject": "Product", "predicate": "hasRollingAnnualDemand", "object": "LiteralValue"}]),
    ("The report rolls demand forward, but it does not state a rolling annual demand field.", []),
    ("The supplier code is the identifier assigned by the supplier.", [{"subject": "Product", "predicate": "hasSupplierCode", "object": "LiteralValue"}]),
    ("Procurement keeps the supplier code next to the product reference.", [{"subject": "Product", "predicate": "hasSupplierCode", "object": "LiteralValue"}]),
    ("A supplier code is not the same as a manufacturer reference.", [{"subject": "Product", "predicate": "hasSupplierCode", "object": "LiteralValue"}]),
    ("The external vendor identifier should be stored as supplier code.", [{"subject": "Product", "predicate": "hasSupplierCode", "object": "LiteralValue"}]),
    ("The word supplier appears in a note, but no supplier code is given.", []),
    ("The manufacturer reference is the code used by the manufacturer.", [{"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"}]),
    ("A manufacturer reference can differ from the supplier code.", [{"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"}]),
    ("Technical drawings list the manufacturer reference for each product.", [{"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"}]),
    ("The OEM part number should be kept as manufacturer reference.", [{"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"}]),
    ("The manufacturer is named, but no manufacturer reference value is shown.", []),
    ("The supplier field identifies the company that supplies the product.", [{"subject": "Product", "predicate": "suppliedBy", "object": "Supplier"}]),
    ("A distributor may supply a product without manufacturing it.", [{"subject": "Product", "predicate": "suppliedBy", "object": "Supplier"}]),
    ("The product is supplied by supplier S04.", [{"subject": "Product", "predicate": "suppliedBy", "object": "Supplier"}]),
    ("Supplier information should create a supplied-by relation.", [{"subject": "Product", "predicate": "suppliedBy", "object": "Supplier"}]),
    ("Supplier wording in the designation should not create a supplied-by relation.", [{"subject": "Product", "predicate": "hasDesignation", "object": "LiteralValue"}]),
    ("The manufacturer field identifies who manufactured the product.", [{"subject": "Product", "predicate": "manufacturedBy", "object": "Manufacturer"}]),
    ("A product manufactured by Bosch should link to a manufacturer.", [{"subject": "Product", "predicate": "manufacturedBy", "object": "Manufacturer"}]),
    ("Manufacturer information should not be confused with the supplier.", [{"subject": "Product", "predicate": "manufacturedBy", "object": "Manufacturer"}]),
    ("The item is manufactured by a brand but sold by a distributor.", [{"subject": "Product", "predicate": "manufacturedBy", "object": "Manufacturer"}, {"subject": "Product", "predicate": "suppliedBy", "object": "Supplier"}]),
    ("The sentence mentions manufacturing constraints but no manufacturer entity.", []),
    ("Dimensions are expressed in millimetres for every product box.", [{"subject": "DimensionMeasurement", "predicate": "hasDefaultUnit", "object": "Millimetre"}]),
    ("Length, width, and height use millimetres by default.", [{"subject": "DimensionMeasurement", "predicate": "hasDefaultUnit", "object": "Millimetre"}]),
    ("Historical dimensions were maintained in centimetres.", [{"subject": "DimensionMeasurement", "predicate": "hasDefaultUnit", "object": "Centimetre"}]),
    ("Dimension measurements should keep the unit separate from the numeric value.", []),
    ("The package dimension note mentions millimetres but not a product size value.", [{"subject": "DimensionMeasurement", "predicate": "hasDefaultUnit", "object": "Millimetre"}]),
    ("Weight values are recorded in kilograms in the product master.", [{"subject": "WeightMeasurement", "predicate": "hasDefaultUnit", "object": "Kilogram"}]),
    ("Small parts may have their weight expressed in grams.", [{"subject": "WeightMeasurement", "predicate": "hasDefaultUnit", "object": "Gram"}]),
    ("The default unit for weight measurement is kilogram.", [{"subject": "WeightMeasurement", "predicate": "hasDefaultUnit", "object": "Kilogram"}]),
    ("The text compares kilograms and grams without assigning a default weight unit.", []),
    ("Weight measurement keeps the unit separate from the numeric mass.", []),
    ("Product length should be represented through a length measurement node.", [{"subject": "Product", "predicate": "hasLengthMeasurement", "object": "LengthMeasurement"}]),
    ("The recorded width should create a width measurement for the product.", [{"subject": "Product", "predicate": "hasWidthMeasurement", "object": "WidthMeasurement"}]),
    ("The height column should create a height measurement for the product.", [{"subject": "Product", "predicate": "hasHeightMeasurement", "object": "HeightMeasurement"}]),
    ("Length, width, and height are all explicit dimension measurements.", [{"subject": "Product", "predicate": "hasLengthMeasurement", "object": "LengthMeasurement"}, {"subject": "Product", "predicate": "hasWidthMeasurement", "object": "WidthMeasurement"}, {"subject": "Product", "predicate": "hasHeightMeasurement", "object": "HeightMeasurement"}]),
    ("The dimensions describe the packed box, not the bare product.", [{"subject": "Product", "predicate": "hasLengthMeasurement", "object": "LengthMeasurement"}, {"subject": "Product", "predicate": "hasWidthMeasurement", "object": "WidthMeasurement"}, {"subject": "Product", "predicate": "hasHeightMeasurement", "object": "HeightMeasurement"}]),
    ("A width measurement is a kind of dimension measurement.", [{"subject": "WidthMeasurement", "predicate": "rdfs:subClassOf", "object": "DimensionMeasurement"}]),
    ("A height measurement is a kind of dimension measurement.", [{"subject": "HeightMeasurement", "predicate": "rdfs:subClassOf", "object": "DimensionMeasurement"}]),
    ("A length measurement is a kind of dimension measurement.", [{"subject": "LengthMeasurement", "predicate": "rdfs:subClassOf", "object": "DimensionMeasurement"}]),
    ("A weight measurement is a kind of measurement.", [{"subject": "WeightMeasurement", "predicate": "rdfs:subClassOf", "object": "Measurement"}]),
    ("The text says measurement several times but makes no subclass statement.", []),
    ("The product is stored in bin B17 in the warehouse.", []),
    ("Location notes should not be interpreted as manufacturer information.", []),
    ("The extraction should ignore candidate triples that are not expressed in the sentence.", []),
    ("A supplier can appear in a sentence about purchase unit without being the object of supplied by.", [{"subject": "Product", "predicate": "hasPurchaseUnit", "object": "LiteralValue"}]),
    ("A manufacturer can appear in a sentence about manufacturer reference without being a manufactured-by assertion.", [{"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"}]),
    ("The family and subfamily values may both be present in the same row.", [{"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"}, {"subject": "Product", "predicate": "belongsToSubfamily", "object": "ProductSubfamily"}]),
    ("The product reference and designation can both appear in the same record.", [{"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"}, {"subject": "Product", "predicate": "hasDesignation", "object": "LiteralValue"}]),
    ("Current stock and annual demand are different quantities.", [{"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}, {"subject": "Product", "predicate": "annualDemand", "object": "LiteralValue"}]),
    ("Rolling annual demand should not overwrite the current stock quantity.", [{"subject": "Product", "predicate": "hasRollingAnnualDemand", "object": "LiteralValue"}, {"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}]),
    ("Supplier code, supplier company, and product reference are three different fields.", [{"subject": "Product", "predicate": "hasSupplierCode", "object": "LiteralValue"}, {"subject": "Product", "predicate": "suppliedBy", "object": "Supplier"}, {"subject": "Product", "predicate": "hasProductReference", "object": "LiteralValue"}]),
    ("Manufacturer reference and manufacturer company should be stored separately.", [{"subject": "Product", "predicate": "hasManufacturerReference", "object": "LiteralValue"}, {"subject": "Product", "predicate": "manufacturedBy", "object": "Manufacturer"}]),
    ("Dimension units in millimetres should not be mixed with weight units in kilograms.", [{"subject": "DimensionMeasurement", "predicate": "hasDefaultUnit", "object": "Millimetre"}, {"subject": "WeightMeasurement", "predicate": "hasDefaultUnit", "object": "Kilogram"}]),
    ("A product family comment can mention suppliers without changing the family assignment.", [{"subject": "Product", "predicate": "belongsToFamily", "object": "ProductFamily"}]),
    ("A stock quantity comment can mention annual demand without giving an annual demand value.", [{"subject": "Product", "predicate": "hasCurrentStockQuantity", "object": "LiteralValue"}]),
    ("The benchmark sentence mentions product, unit, supplier, and family only as vocabulary examples.", []),
]


def find_repo_root(start: Path) -> Path:
    script_root = Path(__file__).resolve().parents[1]
    candidates = [script_root, start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if (candidate / "scripts").exists() and (candidate / "datasets").exists() and (candidate / "ontology").exists():
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
    return (
        int(row["sentence_id"]),
        row["subject_short"],
        row["candidate_relation_short"],
        row["object_short"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a natural-language 100-sentence verifier benchmark.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/evaluation_only_interview_grounded_dataset"),
    )
    parser.add_argument(
        "--owl",
        type=Path,
        default=Path("ontology/internal-logistics-v2.owl"),
    )
    args = parser.parse_args()

    if len(SENTENCE_SPECS) != 100:
        raise RuntimeError(f"Expected exactly 100 sentence specs, got {len(SENTENCE_SPECS)}")

    repo_root = find_repo_root(Path.cwd())
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    if not args.output_root.is_absolute():
        args.output_root = repo_root / args.output_root
    if not args.owl.is_absolute():
        args.owl = repo_root / args.owl

    import end_to_end_verifier_evaluation as evaluator

    evaluator.OWL_PATH = args.owl
    evaluator.formatter.OWL_PATH = args.owl

    sentences = [sentence for sentence, _ in SENTENCE_SPECS]
    gold_rows = [
        {"sentence_id": idx, "sentence": sentence, "gold_triples": triples}
        for idx, (sentence, triples) in enumerate(SENTENCE_SPECS, start=1)
    ]

    text_path = args.output_root / "natural_logistics_benchmark_100_sentences.txt"
    gold_path = args.output_root / "natural_logistics_benchmark_100_gold.jsonl"
    write_text(text_path, sentences)
    write_jsonl(gold_path, gold_rows)

    raw_candidates = evaluator.build_candidate_rows(text_path)
    candidate_rows = evaluator.format_candidate_rows(raw_candidates)
    gold = {
        triple_key(idx, triple)
        for idx, (_, triples) in enumerate(SENTENCE_SPECS, start=1)
        for triple in triples
    }
    candidate_keys = {candidate_key(row) for row in candidate_rows}
    annotated_rows = []
    for row in candidate_rows:
        label = "VALID" if candidate_key(row) in gold else "INVALID"
        annotated_rows.append(
            {
                **row,
                "label": label,
                "manual_gold_label": label,
                "source_sentence": row["sentence"],
                "split": "natural_benchmark_100",
            }
        )

    formats = [
        "A_sentence_only",
        "B_relation_marker",
        "C_relation_entity_markers",
        "D_ontology_context",
        "F_full_ontology_context",
    ]
    write_jsonl(args.output_root / "all_candidates.jsonl", annotated_rows)
    for format_name in formats:
        format_rows = [
            {
                **row,
                "text": row["input_formats"][format_name],
                "format": format_name,
            }
            for row in annotated_rows
        ]
        write_jsonl(args.output_root / format_name / "eval.jsonl", format_rows)

    covered_gold = gold & candidate_keys
    missing_gold = gold - candidate_keys
    stats = {
        "benchmark": "natural_verifier_benchmark_100",
        "sentences": len(SENTENCE_SPECS),
        "text_path": str(text_path),
        "gold_path": str(gold_path),
        "candidate_examples": len(annotated_rows),
        "gold_triples": len(gold),
        "gold_triples_covered_by_candidates": len(covered_gold),
        "gold_candidate_coverage": len(covered_gold) / len(gold) if gold else 0.0,
        "gold_triples_missing_from_candidates": len(missing_gold),
        "label_distribution": {
            "VALID": sum(1 for row in annotated_rows if row["label"] == "VALID"),
            "INVALID": sum(1 for row in annotated_rows if row["label"] == "INVALID"),
        },
        "missing_gold_triples": [list(item) for item in sorted(missing_gold)],
        "formats": formats,
        "note": (
            "Natural-language logistics benchmark authored from ontology relation themes. "
            "Candidate labels are aligned to generated candidates; missing gold triples measure candidate-generation recall failures."
        ),
    }
    (args.output_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
