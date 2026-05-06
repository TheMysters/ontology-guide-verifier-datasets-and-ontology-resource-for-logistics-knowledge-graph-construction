from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NLP_GENERATED = ROOT / "datasets" / "original_verifier_dataset"
OWL_PATH = ROOT / "ontology" / "internal-logistics-v2.owl"
OUTPUT_ROOT = ROOT / "datasets" / "original_verifier_dataset"

NS = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

IL_NS = "http://example.org/internal-logistics#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"

LENGTH_UNITS = {"Millimetre", "Centimetre", "Metre"}
WEIGHT_UNITS = {"Gram", "Kilogram"}
DIMENSION_MEASUREMENTS = {"DimensionMeasurement", "LengthMeasurement", "WidthMeasurement", "HeightMeasurement"}
SCHEMA_UNIT_RELATIONS = {"hasDefaultUnit", "hasUnit"}


def short(uri: str) -> str:
    if uri == "LiteralValue":
        return uri
    if uri.startswith(RDFS_NS):
        return "rdfs:" + uri.split("#")[-1]
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def camel_to_words(name: str) -> str:
    name = short(name)
    if name == "LiteralValue":
        return "literal value"
    if ":" in name:
        name = name.split(":", 1)[1]
    return " ".join(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").split()).lower()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resource(elem: ET.Element) -> str | None:
    return elem.get(f"{{{NS['rdf']}}}resource")


def extract_class_refs(elem: ET.Element) -> list[str]:
    refs: list[str] = []
    direct = resource(elem)
    if direct:
        refs.append(short(direct))
    for desc in elem.findall(".//rdf:Description", NS):
        about = desc.get(f"{{{NS['rdf']}}}about")
        if about:
            refs.append(short(about))
    return list(dict.fromkeys(refs))


def parse_ontology(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    classes: set[str] = set()
    individuals: dict[str, list[str]] = defaultdict(list)
    property_meta: dict[str, dict] = {}
    subclass_parents: dict[str, list[str]] = defaultdict(list)
    disjoint_groups: list[list[str]] = []

    for node in root.findall("owl:Class", NS):
        uri = node.get(f"{{{NS['rdf']}}}about")
        if not uri:
            continue
        cls = short(uri)
        classes.add(cls)
        for parent in node.findall("rdfs:subClassOf", NS):
            parent_uri = resource(parent)
            if parent_uri:
                subclass_parents[cls].append(short(parent_uri))

    for node in root.findall("owl:NamedIndividual", NS):
        uri = node.get(f"{{{NS['rdf']}}}about")
        if not uri:
            continue
        ind = short(uri)
        for rdf_type in node.findall("rdf:type", NS):
            type_uri = resource(rdf_type)
            if type_uri:
                individuals[ind].append(short(type_uri))

    for xpath, kind in (("owl:ObjectProperty", "object_property"), ("owl:DatatypeProperty", "datatype_property")):
        for node in root.findall(xpath, NS):
            uri = node.get(f"{{{NS['rdf']}}}about")
            if not uri:
                continue
            prop = short(uri)
            domains = []
            for elem in node.findall("rdfs:domain", NS):
                domains.extend(extract_class_refs(elem))
            ranges = []
            for elem in node.findall("rdfs:range", NS):
                ranges.extend(extract_class_refs(elem))
            domains = list(dict.fromkeys(domains))
            ranges = list(dict.fromkeys(ranges))
            property_types = []
            for rdf_type in node.findall("rdf:type", NS):
                type_uri = resource(rdf_type)
                if type_uri and type_uri.startswith(OWL_NS):
                    property_types.append(short(type_uri))
            inverse_of = []
            for inverse in node.findall("owl:inverseOf", NS):
                inverse_uri = resource(inverse)
                if inverse_uri:
                    inverse_of.append(short(inverse_uri))
            subproperties = []
            for parent in node.findall("rdfs:subPropertyOf", NS):
                parent_uri = resource(parent)
                if parent_uri:
                    subproperties.append(short(parent_uri))
            deprecated = any(
                (dep.text or "").strip().lower() == "true"
                for dep in node.findall("owl:deprecated", NS)
            )
            property_meta[prop] = {
                "uri": uri,
                "kind": kind,
                "domains": domains,
                "ranges": ranges,
                "has_complex_domain": bool(node.findall("rdfs:domain", NS) and not domains),
                "has_complex_range": bool(node.findall("rdfs:range", NS) and not ranges),
                "property_types": sorted(dict.fromkeys(property_types)),
                "inverse_of": sorted(dict.fromkeys(inverse_of)),
                "subproperty_of": sorted(dict.fromkeys(subproperties)),
                "deprecated": deprecated,
            }

    for node in root.findall("owl:AllDisjointClasses", NS):
        members = []
        for desc in node.findall(".//rdf:Description", NS):
            about = desc.get(f"{{{NS['rdf']}}}about")
            if about:
                members.append(short(about))
        if len(members) > 1:
            disjoint_groups.append(sorted(dict.fromkeys(members)))

    return {
        "classes": sorted(classes),
        "individual_types": dict(individuals),
        "property_meta": property_meta,
        "subclass_parents": dict(subclass_parents),
        "disjoint_groups": disjoint_groups,
    }


def is_subclass_or_same(child: str, parent: str, subclass_parents: dict[str, list[str]]) -> bool:
    if child == parent:
        return True
    seen = set()
    stack = list(subclass_parents.get(child, []))
    while stack:
        current = stack.pop()
        if current == parent:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(subclass_parents.get(current, []))
    return False


def entity_type(entity: str, ontology: dict) -> str:
    entity_short = short(entity)
    if entity_short == "LiteralValue":
        return "LiteralValue"
    if entity_short in ontology["classes"]:
        return entity_short
    types = ontology["individual_types"].get(entity_short, [])
    if types:
        return types[0]
    return "Unknown"


def ancestors(entity_cls: str, ontology: dict) -> list[str]:
    seen = set()
    ordered = []
    stack = list(ontology["subclass_parents"].get(entity_cls, []))
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        stack.extend(ontology["subclass_parents"].get(current, []))
    return ordered


def disjoint_group_context(entity_cls: str, ontology: dict) -> list[str]:
    related = {entity_cls, *ancestors(entity_cls, ontology)}
    groups = []
    for group in ontology["disjoint_groups"]:
        if related.intersection(group):
            groups.append(",".join(group))
    return groups


def normalize_schema_unit_row(row: dict) -> dict:
    normalized = dict(row)
    relation = short(normalized["candidate_relation"])
    if relation != "hasUnit":
        return normalized

    normalized["original_candidate_relation"] = normalized["candidate_relation"]
    normalized["candidate_relation"] = IL_NS + "hasDefaultUnit"

    subject = short(normalized["subject"])
    if subject == "Dimension":
        normalized["subject"] = IL_NS + "DimensionMeasurement"
    elif subject == "Weight":
        normalized["subject"] = IL_NS + "WeightMeasurement"
    return normalized


def domain_range_valid(row: dict, ontology: dict) -> bool | None:
    relation = short(row["candidate_relation"])
    if relation == "rdfs:subClassOf":
        return True
    meta = ontology["property_meta"].get(relation)
    if not meta:
        return None

    subject_type = entity_type(row["subject"], ontology)
    object_type = entity_type(row["object"], ontology)
    subclass_parents = ontology["subclass_parents"]

    domain_ok = True
    if meta["domains"]:
        domain_ok = any(is_subclass_or_same(subject_type, domain, subclass_parents) for domain in meta["domains"])
    elif meta["has_complex_domain"]:
        domain_ok = True

    range_ok = True
    if meta["kind"] == "datatype_property":
        range_ok = object_type == "LiteralValue" or short(row["object"]) == "LiteralValue"
    elif meta["ranges"]:
        range_ok = any(is_subclass_or_same(object_type, rng, subclass_parents) for rng in meta["ranges"])
    elif meta["has_complex_range"]:
        range_ok = True

    return bool(domain_ok and range_ok)


def unit_compatible(row: dict) -> bool | None:
    relation = short(row["candidate_relation"])
    subject = short(row["subject"])
    obj = short(row["object"])
    if relation not in SCHEMA_UNIT_RELATIONS | {"hasMeasurementUnit"}:
        return None
    if subject in DIMENSION_MEASUREMENTS:
        return obj in LENGTH_UNITS
    if subject in {"Weight", "WeightMeasurement"}:
        return obj in WEIGHT_UNITS
    return None


def validation_flag(row: dict, ontology: dict) -> str:
    dr = domain_range_valid(row, ontology)
    unit = unit_compatible(row)
    if dr is False:
        return "STRUCTURALLY_INVALID"
    if unit is False:
        return "UNIT_INCOMPATIBLE"
    if dr is True and unit is True and short(row["candidate_relation"]) == "hasDefaultUnit":
        return "SCHEMA_DEFAULT_UNIT_COMPATIBLE"
    if dr is True and unit is True:
        return "UNIT_COMPATIBLE"
    if dr is True:
        return "STRUCTURALLY_VALID"
    return "NOT_APPLICABLE"


def strip_existing_markers(text: str) -> str:
    return re.sub(r"\[/?(?:REL|E1|E2)\]\s*", "", text).strip()


def ontology_context(row: dict, ontology: dict) -> str:
    relation = short(row["candidate_relation"])
    subject = short(row["subject"])
    obj = short(row["object"])
    meta = ontology["property_meta"].get(relation, {})
    domains = ",".join(meta.get("domains", [])) or "Unknown"
    ranges = ",".join(meta.get("ranges", [])) or ("LiteralValue" if meta.get("kind") == "datatype_property" else "Unknown")
    subject_type = entity_type(row["subject"], ontology)
    object_type = entity_type(row["object"], ontology)
    return (
        f"[SUBJ] {subject} [/SUBJ] [SUBJ_TYPE] {subject_type} [/SUBJ_TYPE] "
        f"[REL] {relation} [/REL] [OBJ] {obj} [/OBJ] [OBJ_TYPE] {object_type} [/OBJ_TYPE] "
        f"[DOMAIN] {domains} [/DOMAIN] [RANGE] {ranges} [/RANGE]"
    )


def full_ontology_context(row: dict, ontology: dict) -> str:
    relation = short(row["candidate_relation"])
    meta = ontology["property_meta"].get(relation, {})
    subject_type = entity_type(row["subject"], ontology)
    object_type = entity_type(row["object"], ontology)

    subject_ancestors = ",".join(ancestors(subject_type, ontology)) or "None"
    object_ancestors = ",".join(ancestors(object_type, ontology)) or "None"
    property_types = ",".join(meta.get("property_types", [])) or "None"
    subject_inferred_types = ",".join([subject_type, *ancestors(subject_type, ontology)]) or subject_type
    object_inferred_types = ",".join([object_type, *ancestors(object_type, ontology)]) or object_type
    inverse_of = ",".join(meta.get("inverse_of", [])) or "None"
    subproperty_of = ",".join(meta.get("subproperty_of", [])) or "None"
    deprecated = "true" if meta.get("deprecated") else "false"
    cardinality_hint = "max_1" if "FunctionalProperty" in meta.get("property_types", []) else "not_declared"
    subject_disjoint = " | ".join(disjoint_group_context(subject_type, ontology)) or "None"
    object_disjoint = " | ".join(disjoint_group_context(object_type, ontology)) or "None"
    unit_family = unit_compatible(row)
    unit_context = "not_applicable" if unit_family is None else "unit_pair"

    return (
        f"{ontology_context(row, ontology)} "
        f"[REL_TYPE] {meta.get('kind', 'Unknown')} [/REL_TYPE] "
        f"[REL_CHARACTERISTICS] {property_types} [/REL_CHARACTERISTICS] "
        f"[CARDINALITY_HINT] {cardinality_hint} [/CARDINALITY_HINT] "
        f"[INVERSE_OF] {inverse_of} [/INVERSE_OF] "
        f"[SUBPROPERTY_OF] {subproperty_of} [/SUBPROPERTY_OF] "
        f"[DEPRECATED_RELATION] {deprecated} [/DEPRECATED_RELATION] "
        f"[INFERRED_SUBJ_TYPES] {subject_inferred_types} [/INFERRED_SUBJ_TYPES] "
        f"[INFERRED_OBJ_TYPES] {object_inferred_types} [/INFERRED_OBJ_TYPES] "
        f"[SUBJ_SUPERCLASSES] {subject_ancestors} [/SUBJ_SUPERCLASSES] "
        f"[OBJ_SUPERCLASSES] {object_ancestors} [/OBJ_SUPERCLASSES] "
        f"[SUBJ_DISJOINT_GROUPS] {subject_disjoint} [/SUBJ_DISJOINT_GROUPS] "
        f"[OBJ_DISJOINT_GROUPS] {object_disjoint} [/OBJ_DISJOINT_GROUPS] "
        f"[UNIT_CONTEXT] {unit_context} [/UNIT_CONTEXT]"
    )


def build_formats(row: dict, ontology: dict) -> dict[str, str]:
    sentence = row["sentence"]
    relation = short(row["candidate_relation"])
    subject = short(row["subject"])
    obj = short(row["object"])
    subject_words = camel_to_words(subject)
    object_words = camel_to_words(obj)
    marked_sentence = row["text"].split("[/REL]", 1)[-1].strip() if "[/REL]" in row["text"] else row["text"]
    validation = validation_flag(row, ontology)

    return {
        "A_sentence_only": sentence,
        "B_relation_marker": f"[REL] {relation} [/REL] {sentence}",
        "C_relation_entity_markers": f"[REL] {relation} [/REL] {marked_sentence}",
        "D_ontology_context": f"{ontology_context(row, ontology)} {marked_sentence}",
        "E_validation_context": f"[VALIDATION] {validation} [/VALIDATION] {ontology_context(row, ontology)} {marked_sentence}",
        "F_full_ontology_context": f"{full_ontology_context(row, ontology)} {marked_sentence}",
        "G_symbolic_triple_text": (
            f"candidate subject {subject_words}; candidate relation {camel_to_words(relation)}; "
            f"candidate object {object_words}; sentence: {sentence}"
        ),
    }


def enrich_row(row: dict, ontology: dict, split: str) -> dict:
    row = normalize_schema_unit_row(row)
    relation = short(row["candidate_relation"])
    subject = short(row["subject"])
    obj = short(row["object"])
    dr = domain_range_valid(row, ontology)
    unit = unit_compatible(row)
    validation = validation_flag(row, ontology)
    formats = build_formats(row, ontology)
    return {
        **row,
        "text": formats["C_relation_entity_markers"],
        "split": split,
        "candidate_relation_short": relation,
        "subject_short": subject,
        "object_short": obj,
        "subject_type": entity_type(row["subject"], ontology),
        "object_type": entity_type(row["object"], ontology),
        "domain_range_valid": dr,
        "unit_compatible": unit,
        "ontology_validation_flag": validation,
        "ontology_context_leakage_risk": validation in {"STRUCTURALLY_INVALID", "UNIT_INCOMPATIBLE"},
        "input_formats": formats,
    }


def write_format_splits(rows_by_split: dict[str, list[dict]], format_name: str) -> None:
    out_dir = OUTPUT_ROOT / format_name
    for split, rows in rows_by_split.items():
        formatted_rows = [
            {
                "text": row["input_formats"][format_name],
                "label": row["label"],
                "candidate_relation": row["candidate_relation"],
                "subject": row["subject"],
                "object": row["object"],
                "source_sentence": row["sentence"],
                "domain_range_valid": row["domain_range_valid"],
                "unit_compatible": row["unit_compatible"],
                "ontology_validation_flag": row["ontology_validation_flag"],
            }
            for row in rows
        ]
        write_jsonl(out_dir / f"{split}.jsonl", formatted_rows)


def build_stats(rows_by_split: dict[str, list[dict]]) -> dict:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    return {
        "total_examples": len(all_rows),
        "split_distribution": {split: len(rows) for split, rows in rows_by_split.items()},
        "label_distribution": dict(Counter(row["label"] for row in all_rows)),
        "domain_range_valid_distribution": dict(Counter(str(row["domain_range_valid"]) for row in all_rows)),
        "unit_compatible_distribution": dict(Counter(str(row["unit_compatible"]) for row in all_rows)),
        "ontology_validation_flag_distribution": dict(Counter(row["ontology_validation_flag"] for row in all_rows)),
        "candidate_relation_distribution": dict(Counter(row["candidate_relation_short"] for row in all_rows)),
        "leakage_risk_examples": sum(1 for row in all_rows if row["ontology_context_leakage_risk"]),
        "input_formats": [
            "A_sentence_only",
            "B_relation_marker",
            "C_relation_entity_markers",
            "D_ontology_context",
            "E_validation_context",
            "F_full_ontology_context",
            "G_symbolic_triple_text",
        ],
    }


def write_summary(stats: dict) -> None:
    lines = [
        "# Ontology-Aware Verifier Dataset",
        "",
        "This dataset keeps the same verifier examples as the existing logistics verifier dataset, but adds ontology metadata and multiple input formats.",
        "",
        "## Important Warning",
        "",
        "`E_validation_context` can leak symbolic information into the classifier because validation flags may correlate strongly with the label. Use it as an ablation to measure leakage/upper-bound behavior, not as the default final model input.",
        "",
        "Recommended fair comparison order:",
        "",
        "1. `A_sentence_only`",
        "2. `B_relation_marker`",
        "3. `C_relation_entity_markers`",
        "4. `D_ontology_context`",
        "5. `F_full_ontology_context`",
        "6. `E_validation_context` only as a diagnostic leakage-risk setting",
        "7. `G_symbolic_triple_text`",
        "",
        "## Stats",
        "",
        f"- Total examples: `{stats['total_examples']}`",
        f"- Leakage-risk examples: `{stats['leakage_risk_examples']}`",
        "",
        "### Label Distribution",
        "",
    ]
    for key, value in stats["label_distribution"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "### Ontology Validation Flags", ""])
    for key, value in stats["ontology_validation_flag_distribution"].items():
        lines.append(f"- `{key}`: {value}")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ontology = parse_ontology(OWL_PATH)
    rows_by_split: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        raw_rows = load_jsonl(NLP_GENERATED / f"logistics_verifier_{split}.jsonl")
        rows_by_split[split] = [enrich_row(row, ontology, split) for row in raw_rows]
        write_jsonl(OUTPUT_ROOT / f"ontology_aware_{split}.jsonl", rows_by_split[split])

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    write_jsonl(OUTPUT_ROOT / "ontology_aware_all.jsonl", all_rows)

    for format_name in build_formats(rows_by_split["train"][0], ontology):
        write_format_splits(rows_by_split, format_name)

    stats = build_stats(rows_by_split)
    (OUTPUT_ROOT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(stats)
    print("Wrote ontology-aware verifier dataset to", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
