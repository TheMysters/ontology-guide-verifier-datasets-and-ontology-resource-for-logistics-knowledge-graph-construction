from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BASE_DIR / "data-input"
DEFAULT_OWL_PATH = BASE_DIR / "ontology" / "internal-logistics-v2.owl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "generated"

NS = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}

ENTITY_TAGS = (
    ("owl:Class", "class"),
    ("owl:NamedIndividual", "individual"),
    ("owl:ObjectProperty", "object_property"),
    ("owl:DatatypeProperty", "datatype_property"),
)

IMPLICIT_RANGE_OBJECT_PROPERTIES = {
    "suppliedBy",
    "manufacturedBy",
    "belongsToFamily",
    "belongsToSubfamily",
}

MOJIBAKE_REPLACEMENTS = {
    "Ã¢â‚¬â„¢": "'",
    "Ã¢â‚¬Ëœ": "'",
    "Ã¢â‚¬Å“": '"',
    "Ã¢â‚¬Â": '"',
    "Ã¢â‚¬â€œ": "-",
    "Ã¢â‚¬â€": "-",
    "Ã¢â‚¬Â¦": "...",
    "Ã‚ ": " ",
    "Ã‚": "",
}


def short(uri: str) -> str:
    return uri.split("#")[-1] if "#" in uri else uri.rstrip("/").split("/")[-1]


def repair_mojibake(text: str) -> str:
    cleaned = text
    if any(token in cleaned for token in ("Ã¢â‚¬â„¢", "Ã¢â‚¬Å“", "Ã¢â‚¬", "Ãƒ")):
        try:
            candidate = cleaned.encode("latin-1").decode("utf-8")
            if candidate.count("\ufffd") <= cleaned.count("\ufffd"):
                cleaned = candidate
        except UnicodeError:
            pass
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned.replace("\u00a0", " ")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalize_for_matching(text: str) -> str:
    text = strip_accents(repair_mojibake(text).lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def unit_family(unit_short: str) -> str | None:
    if unit_short in {"Millimetre", "Centimetre", "Metre"}:
        return "dimension"
    if unit_short in {"Kilogram", "Gram"}:
        return "weight"
    return None


def schema_measurement_class(concept_short: str) -> tuple[str, str] | None:
    if concept_short == "Dimension":
        return "#DimensionMeasurement", "DimensionMeasurement"
    if concept_short == "Weight":
        return "#WeightMeasurement", "WeightMeasurement"
    return None


def singular_variants(token: str) -> list[str]:
    variants = [token]
    if token.endswith("ies") and len(token) > 3:
        variants.append(token[:-3] + "y")
    if token.endswith("es") and len(token) > 2:
        variants.append(token[:-2])
    if token.endswith("s") and len(token) > 1:
        variants.append(token[:-1])
    return list(dict.fromkeys(v for v in variants if v))


def spelling_variants(token: str) -> list[str]:
    variants = [token]
    if token.endswith("metre"):
        variants.append(token[:-2] + "er")
    return list(dict.fromkeys(variants))


def build_alias_pattern(alias: str) -> re.Pattern[str]:
    alias = normalize_space(alias)
    words = alias.split()
    if words and all(re.fullmatch(r"[A-Za-z]+", word) for word in words):
        tail = words[-1]
        tail_forms: list[str] = []
        for variant in spelling_variants(tail):
            tail_forms.extend(singular_variants(variant))
        tail_forms = list(dict.fromkeys(tail_forms))
        if tail not in tail_forms:
            tail_forms.insert(0, tail)
        variants = []
        for form in tail_forms:
            variants.append(form)
            if not form.endswith("s"):
                if form.endswith("y") and len(form) > 1 and form[-2] not in "aeiou":
                    variants.append(form[:-1] + "ies")
                variants.append(form + "s")
                if form.endswith(("s", "x", "z", "ch", "sh")):
                    variants.append(form + "es")
        tail_pattern = "(?:" + "|".join(re.escape(v) for v in dict.fromkeys(variants)) + ")"
        body = [re.escape(word) for word in words[:-1]] + [tail_pattern]
        return re.compile(r"(?<!\w)" + r"\s+".join(body) + r"(?!\w)", flags=re.IGNORECASE)
    return re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", flags=re.IGNORECASE)


def sentence_split(text: str) -> list[str]:
    text = repair_mojibake(text)
    chunks = re.split(r"\n+", text)
    sentences: list[str] = []
    for chunk in chunks:
        piece = normalize_space(chunk)
        if not piece:
            continue
        parts = re.split(r"(?<=[.!?])\s+", piece)
        for part in parts:
            sentence = normalize_space(part)
            if sentence:
                sentences.append(sentence)
    return sentences


def read_labels(node: ET.Element) -> list[str]:
    labels: list[str] = []
    for tag in ("rdfs:label", "skos:altLabel"):
        for elem in node.findall(tag, NS):
            if elem.text:
                value = normalize_space(repair_mojibake(elem.text))
                if value and value not in labels:
                    labels.append(value)
    return labels


def extract_resource_targets(node: ET.Element, tag: str) -> list[str]:
    targets: list[str] = []
    for elem in node.findall(tag, NS):
        resource = elem.get(f"{{{NS['rdf']}}}resource")
        if resource:
            targets.append(resource)
        for desc in elem.findall(".//rdf:Description", NS):
            resource = desc.get(f"{{{NS['rdf']}}}about")
            if resource:
                targets.append(resource)
    return list(dict.fromkeys(targets))


def parse_ontology(owl_path: Path) -> dict:
    tree = ET.parse(owl_path)
    root = tree.getroot()

    entities: dict[str, dict] = {}
    aliases: list[dict] = []
    relation_uris: list[str] = []
    subproperty_children: dict[str, list[str]] = defaultdict(list)
    property_domains: dict[str, list[str]] = {}
    property_ranges: dict[str, list[str]] = {}

    for xpath, kind in ENTITY_TAGS:
        for node in root.findall(xpath, NS):
            uri = node.get(f"{{{NS['rdf']}}}about")
            if not uri:
                continue
            labels = read_labels(node)
            fragment = short(uri)
            if fragment not in labels:
                labels.append(fragment)
            entities[uri] = {
                "uri": uri,
                "short": fragment,
                "kind": kind,
                "labels": labels,
                "canonical_label": labels[0],
            }
            if kind in {"object_property", "datatype_property"}:
                relation_uris.append(uri)
                property_domains[uri] = extract_resource_targets(node, "rdfs:domain")
                property_ranges[uri] = extract_resource_targets(node, "rdfs:range")
            for label in labels:
                aliases.append(
                    {
                        "alias": label,
                        "normalized_alias": normalize_for_matching(label),
                        "pattern": build_alias_pattern(label),
                        "uri": uri,
                        "kind": kind,
                        "canonical_label": labels[0],
                        "short": fragment,
                    }
                )

    for xpath in ("owl:ObjectProperty", "owl:DatatypeProperty"):
        for node in root.findall(xpath, NS):
            child_uri = node.get(f"{{{NS['rdf']}}}about")
            if not child_uri:
                continue
            for parent in node.findall("rdfs:subPropertyOf", NS):
                parent_uri = parent.get(f"{{{NS['rdf']}}}resource")
                if parent_uri:
                    subproperty_children[parent_uri].append(child_uri)

    semantic_groups: dict[str, list[str]] = {}
    semantic_group_index: dict[str, list[str]] = defaultdict(list)

    for parent_uri, children in sorted(subproperty_children.items()):
        deduped_children = sorted(dict.fromkeys(children), key=short)
        parent_short = short(parent_uri)
        parent_entity = entities.get(parent_uri, {"labels": [parent_short]})
        semantic_groups[parent_short] = [short(uri) for uri in deduped_children]

        keys = {parent_short}
        keys.update(parent_entity.get("labels", []))

        if parent_short.startswith("has") and len(parent_short) > 3:
            class_short = parent_short[3:]
            keys.add(class_short)
            for uri, meta in entities.items():
                if meta["kind"] == "class" and meta["short"].lower() == class_short.lower():
                    keys.update(meta["labels"])
                    semantic_groups[meta["short"]] = [short(child) for child in deduped_children]
                    break

        for key in keys:
            normalized_key = normalize_for_matching(key)
            if normalized_key:
                semantic_group_index[normalized_key].extend(deduped_children)

    alias_index = sorted(aliases, key=lambda item: len(item["alias"]), reverse=True)

    entity_dictionary: dict[str, list[dict]] = defaultdict(list)
    for alias in alias_index:
        entity_dictionary[alias["normalized_alias"]].append(
            {
                "uri": alias["uri"],
                "short": alias["short"],
                "kind": alias["kind"],
                "canonical_label": alias["canonical_label"],
            }
        )

    relation_vocabulary = [
        {
            "uri": uri,
            "short": short(uri),
            "label": entities[uri]["canonical_label"],
            "kind": entities[uri]["kind"],
        }
        for uri in sorted(set(relation_uris), key=short)
    ]

    return {
        "entities": entities,
        "alias_index": alias_index,
        "entity_dictionary": dict(entity_dictionary),
        "relation_vocabulary": relation_vocabulary,
        "semantic_groups": semantic_groups,
        "semantic_group_index": {
            k: sorted(dict.fromkeys(v), key=short) for k, v in semantic_group_index.items()
        },
        "property_domains": property_domains,
        "property_ranges": property_ranges,
    }


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def detect_mentions(text: str, ontology: dict) -> list[dict]:
    mentions: list[dict] = []
    occupied: list[tuple[int, int]] = []

    for entry in ontology["alias_index"]:
        for match in entry["pattern"].finditer(text):
            span = (match.start(), match.end())
            if any(overlaps(span, existing) for existing in occupied):
                continue
            occupied.append(span)
            mentions.append(
                {
                    "uri": entry["uri"],
                    "short": entry["short"],
                    "kind": entry["kind"],
                    "canonical_label": entry["canonical_label"],
                    "matched_text": text[span[0] : span[1]],
                    "start": span[0],
                    "end": span[1],
                }
            )

    mentions.sort(key=lambda item: (item["start"], item["end"]))
    return mentions


def infer_candidate_relations(mentions: list[dict], ontology: dict) -> tuple[list[dict], list[dict]]:
    candidates: dict[str, dict] = {}
    support_pairs: list[dict] = []

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for mention in mentions:
        by_kind[mention["kind"]].append(mention)

    class_mentions = by_kind.get("class", [])
    class_mentions_by_short = {mention["short"]: mention for mention in class_mentions}

    if "ProductFamily" in class_mentions_by_short:
        candidates["asserted::#Product::#belongsToFamily::#ProductFamily"] = {
            "uri": "#belongsToFamily",
            "short": "belongsToFamily",
            "source": "classification_concept_mention",
            "triple_mode": "ontology_assertion",
            "subject_uri": "#Product",
            "subject_short": "Product",
            "object_uri": "#ProductFamily",
            "object_short": "ProductFamily",
        }
    if "ProductSubfamily" in class_mentions_by_short:
        candidates["asserted::#Product::#belongsToSubfamily::#ProductSubfamily"] = {
            "uri": "#belongsToSubfamily",
            "short": "belongsToSubfamily",
            "source": "classification_concept_mention",
            "triple_mode": "ontology_assertion",
            "subject_uri": "#Product",
            "subject_short": "Product",
            "object_uri": "#ProductSubfamily",
            "object_short": "ProductSubfamily",
        }

    for mention in mentions:
        if mention["kind"] in {"object_property", "datatype_property"}:
            domain_uris = ontology["property_domains"].get(mention["uri"], [])
            grounded_domain = None
            for domain_uri in domain_uris:
                domain_short = short(domain_uri)
                if domain_short in class_mentions_by_short:
                    grounded_domain = (domain_uri, domain_short)
                    break
            if grounded_domain is None and len(domain_uris) == 1:
                grounded_domain = (domain_uris[0], short(domain_uris[0]))
            if grounded_domain is not None:
                if mention["kind"] == "datatype_property":
                    key = f"asserted::{grounded_domain[0]}::{mention['uri']}::LiteralValue"
                    candidates[key] = {
                        "uri": mention["uri"],
                        "short": mention["short"],
                        "source": "property_with_grounded_domain",
                        "triple_mode": "ontology_assertion",
                        "subject_uri": grounded_domain[0],
                        "subject_short": grounded_domain[1],
                        "object_uri": "LiteralValue",
                        "object_short": "LiteralValue",
                    }
                    continue

                range_uris = ontology["property_ranges"].get(mention["uri"], [])
                grounded_range = None
                for range_uri in range_uris:
                    range_short = short(range_uri)
                    if range_short in class_mentions_by_short:
                        grounded_range = (range_uri, range_short)
                        break
                if grounded_range is None and mention["short"] in IMPLICIT_RANGE_OBJECT_PROPERTIES and len(range_uris) == 1:
                    grounded_range = (range_uris[0], short(range_uris[0]))
                if grounded_range is not None:
                    key = f"asserted::{grounded_domain[0]}::{mention['uri']}::{grounded_range[0]}"
                    candidates[key] = {
                        "uri": mention["uri"],
                        "short": mention["short"],
                        "source": "property_with_grounded_domain_and_range",
                        "triple_mode": "ontology_assertion",
                        "subject_uri": grounded_domain[0],
                        "subject_short": grounded_domain[1],
                        "object_uri": grounded_range[0],
                        "object_short": grounded_range[1],
                    }
                    continue
            key = f"schema::{mention['uri']}"
            candidates[key] = {
                "uri": mention["uri"],
                "short": mention["short"],
                "source": "direct_property_mention",
                "triple_mode": "schema_property",
                "subject_uri": "#Product",
                "subject_short": "Product",
                "object_uri": "LiteralValue",
                "object_short": "LiteralValue",
            }

    units = by_kind.get("individual", [])
    schema_unit_uri = None
    schema_unit_short = "hasDefaultUnit"
    for relation in ontology["relation_vocabulary"]:
        if relation["short"] == "hasDefaultUnit":
            schema_unit_uri = relation["uri"]
            schema_unit_short = relation["short"]
            break
    if schema_unit_uri is None:
        for relation in ontology["relation_vocabulary"]:
            if relation["short"] == "hasUnit":
                schema_unit_uri = relation["uri"]
                schema_unit_short = relation["short"]
                break

    measurement_concepts = [
        mention for mention in by_kind.get("class", []) if mention["short"] in {"Dimension", "Weight"}
    ]
    if schema_unit_uri:
        for concept in measurement_concepts:
            schema_subject = schema_measurement_class(concept["short"])
            if schema_subject is None:
                continue
            subject_uri, subject_short = schema_subject
            for unit in units:
                concept_family = concept["short"].lower()
                compatible_family = unit_family(unit["short"])
                if compatible_family and compatible_family != concept_family:
                    continue
                key = f"asserted::{subject_uri}::{schema_unit_uri}::{unit['uri']}"
                candidates[key] = {
                    "uri": schema_unit_uri,
                    "short": schema_unit_short,
                    "source": "concept_unit_pair",
                    "triple_mode": "ontology_assertion",
                    "subject_uri": subject_uri,
                    "subject_short": subject_short,
                    "object_uri": unit["uri"],
                    "object_short": unit["short"],
                }
                support_pairs.append(
                    {
                        "trigger": concept,
                        "unit": unit,
                        "relation_uri": schema_unit_uri,
                        "relation_short": schema_unit_short,
                        "subject_uri": subject_uri,
                        "subject_short": subject_short,
                        "object_uri": unit["uri"],
                        "object_short": unit["short"],
                    }
                )

    for mention in mentions:
        keys = {
            normalize_for_matching(mention["short"]),
            normalize_for_matching(mention["canonical_label"]),
            normalize_for_matching(mention["matched_text"]),
        }
        group_children: list[str] = []
        for key in keys:
            group_children.extend(ontology["semantic_group_index"].get(key, []))
        for child_uri in sorted(dict.fromkeys(group_children), key=short):
            source = "semantic_expansion"
            if units:
                source = "semantic_expansion_with_unit"
            key = f"schema::{child_uri}"
            candidates[key] = {
                "uri": child_uri,
                "short": short(child_uri),
                "source": source,
                "triple_mode": "schema_property",
                "subject_uri": "#Product",
                "subject_short": "Product",
                "object_uri": "LiteralValue",
                "object_short": "LiteralValue",
            }
            support_pairs.append(
                {
                    "trigger": mention,
                    "unit": units[0] if units else None,
                    "relation_uri": child_uri,
                    "relation_short": short(child_uri),
                    "subject_uri": "#Product",
                    "subject_short": "Product",
                    "object_uri": "LiteralValue",
                    "object_short": "LiteralValue",
                }
            )

    ordered_candidates = sorted(
        candidates.values(),
        key=lambda item: (item["subject_short"], item["short"], item["object_short"]),
    )
    return ordered_candidates, support_pairs


def mark_text(text: str, mentions: list[dict]) -> str:
    selected = sorted(mentions, key=lambda item: (item["start"], item["end"]))[:2]
    if not selected:
        return text
    out = text
    tagged = []
    for idx, mention in enumerate(selected, start=1):
        tagged.append((mention["start"], mention["end"], f"E{idx}"))
    tagged.sort(key=lambda item: item[0], reverse=True)
    for start, end, tag in tagged:
        out = out[:start] + f"[{tag}]" + out[start:end] + f"[/{tag}]" + out[end:]
    return out


def assign_split(text: str) -> str:
    digest = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % 100
    if digest < 70:
        return "train"
    if digest < 85:
        return "val"
    return "test"


def phase_name(path: Path) -> str:
    stem = path.stem.lower()
    return stem.split("_")[0]


def preprocess_file(path: Path, ontology: dict) -> tuple[list[dict], list[dict]]:
    text = repair_mojibake(path.read_text(encoding="utf-8", errors="replace"))
    sentence_rows: list[dict] = []
    relation_rows: list[dict] = []

    for index, sentence in enumerate(sentence_split(text), start=1):
        mentions = detect_mentions(sentence, ontology)
        candidates, support_pairs = infer_candidate_relations(mentions, ontology)
        split = assign_split(sentence)
        sentence_id = f"{path.stem}_s{index:03d}"

        sentence_row = {
            "sentence_id": sentence_id,
            "source_file": str(path),
            "phase": phase_name(path),
            "split": split,
            "text": sentence,
            "mentions": [
                {
                    "uri": mention["uri"],
                    "short": mention["short"],
                    "kind": mention["kind"],
                    "canonical_label": mention["canonical_label"],
                    "matched_text": mention["matched_text"],
                    "start": mention["start"],
                    "end": mention["end"],
                }
                for mention in mentions
            ],
            "candidate_relations": candidates,
        }
        sentence_rows.append(sentence_row)

        support_by_relation: dict[str, list[dict]] = defaultdict(list)
        for pair in support_pairs:
            support_by_relation[pair["relation_uri"]].append(pair)

        for candidate in candidates:
            supports = support_by_relation.get(candidate["uri"], [])
            tagged_mentions: list[dict] = []
            if supports:
                tagged_mentions.append(supports[0]["trigger"])
                if supports[0]["unit"] is not None:
                    tagged_mentions.append(supports[0]["unit"])
            elif mentions:
                tagged_mentions = mentions[:2]

            relation_rows.append(
                {
                    "sentence_id": sentence_id,
                    "source_file": str(path),
                    "phase": phase_name(path),
                    "split": split,
                    "text": mark_text(sentence, tagged_mentions),
                    "sentence": sentence,
                    "candidate_relation": candidate["short"],
                    "candidate_relation_uri": candidate["uri"],
                    "weak_signal": candidate["source"],
                    "mentions": sentence_row["mentions"],
                }
            )

    return sentence_rows, relation_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stats(sentence_rows: list[dict], relation_rows: list[dict]) -> dict:
    mention_kinds = Counter()
    relation_counts = Counter()
    split_counts = Counter()
    phase_counts = Counter()

    for row in sentence_rows:
        split_counts[row["split"]] += 1
        phase_counts[row["phase"]] += 1
        for mention in row["mentions"]:
            mention_kinds[mention["kind"]] += 1

    for row in relation_rows:
        relation_counts[row["candidate_relation"]] += 1

    return {
        "total_sentences": len(sentence_rows),
        "sentences_with_mentions": sum(1 for row in sentence_rows if row["mentions"]),
        "sentences_with_candidate_relations": sum(1 for row in sentence_rows if row["candidate_relations"]),
        "total_relation_rows": len(relation_rows),
        "sentence_split_distribution": dict(split_counts),
        "phase_distribution": dict(phase_counts),
        "mention_kind_distribution": dict(mention_kinds),
        "candidate_relation_distribution": dict(relation_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ontology-aware weakly supervised logistics NLP dataset from phase text files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--owl-path", type=Path, default=DEFAULT_OWL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glob", type=str, default="phase*_en.txt")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ontology = parse_ontology(args.owl_path)
    sentence_rows: list[dict] = []
    relation_rows: list[dict] = []

    for path in sorted(args.input_dir.glob(args.glob)):
        file_sentence_rows, file_relation_rows = preprocess_file(path, ontology)
        sentence_rows.extend(file_sentence_rows)
        relation_rows.extend(file_relation_rows)

    ontology_snapshot = {
        "ontology": str(args.owl_path),
        "entity_dictionary": ontology["entity_dictionary"],
        "relation_vocabulary": ontology["relation_vocabulary"],
        "semantic_groups": ontology["semantic_groups"],
    }
    (args.output_dir / "ontology_interface.json").write_text(
        json.dumps(ontology_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_jsonl(args.output_dir / "logistics_sentences_all.jsonl", sentence_rows)
    write_jsonl(args.output_dir / "logistics_relation_candidates_all.jsonl", relation_rows)

    for split in ("train", "val", "test"):
        write_jsonl(
            args.output_dir / f"logistics_relation_candidates_{split}.jsonl",
            [row for row in relation_rows if row["split"] == split],
        )

    stats = build_stats(sentence_rows, relation_rows)
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Wrote ontology interface:", args.output_dir / "ontology_interface.json")
    print("Wrote sentence dataset:", args.output_dir / "logistics_sentences_all.jsonl")
    print("Wrote relation dataset:", args.output_dir / "logistics_relation_candidates_all.jsonl")
    print("Wrote stats:", args.output_dir / "dataset_stats.json")


if __name__ == "__main__":
    main()
