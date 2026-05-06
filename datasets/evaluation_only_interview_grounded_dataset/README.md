# Evaluation-Only Interview-Grounded Dataset

## Purpose

This dataset is used only for additional evaluation outside training. It provides a robustness check under a distinct sentence distribution.

## Composition

The dataset contains logistics-style sentences generated independently from the main training workflow. It is designed to test whether the ranking observed across verifier input configurations remains stable beyond the train/validation/test splits of the main trainable dataset.

## Files

The dataset includes:

- `natural_logistics_benchmark_100_gold.jsonl`
- `natural_logistics_benchmark_100_sentences.txt`
- `stats.json`

Format-specific evaluation files are also provided for the retained configurations:

- `A_sentence_only/`
- `B_relation_marker/`
- `C_relation_entity_markers/`
- `D_ontology_context/`
- `F_full_ontology_context/`

These directories contain `eval.jsonl` files corresponding to the released evaluation setting.

## How this folder is used locally

The script-level end-to-end evaluation workflow uses this dataset as its default evaluation source. The public end-to-end notebook instead runs a smaller local demonstration on manually defined test sentences so that readers can inspect the pipeline behavior more directly.

## Role in the experiments

In the paper, this dataset is the evaluation-only interview-grounded dataset used to assess robustness under a distinct sentence distribution.
