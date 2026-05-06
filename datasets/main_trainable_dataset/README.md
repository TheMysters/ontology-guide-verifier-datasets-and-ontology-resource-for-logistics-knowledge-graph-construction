# Main Trainable Dataset

## Purpose

This dataset is the main training and evaluation resource used for the core comparison experiments reported in the paper. In this repository, it is the dataset used by the public retraining notebook for the `C_relation_entity_markers` verifier.

## Composition

The dataset combines interview-style examples and ontology-aware synthetic examples, with the goal of creating a stricter verifier setting than the original dataset. It includes harder entity-disambiguation cases in which positive and negative candidates may share the same relation and nearly the same sentence while differing in subject--object alignment.

## Organization

This dataset is organized by retained verifier input format:

- `A_sentence_only/`
- `B_relation_marker/`
- `C_relation_entity_markers/`
- `D_ontology_context/`
- `F_full_ontology_context/`

Each retained format directory contains the split files used in the experiments, typically including:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `all.jsonl`
- `eval.jsonl`

Shared metadata is provided in:

- `stats.json`

## How to read this folder

Each format directory contains the same underlying examples rendered into a different verifier input representation. For example, the `C_relation_entity_markers/` directory is the one used by the public training notebook and by the local model layout already included in `models/main_trainable_dataset/C/`.

## Role in the experiments

In the paper, this dataset is the main trainable dataset used for the core comparison across verifier input representations.
