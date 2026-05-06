# Dataset Collection

This directory contains the verifier datasets used in the reported ontology-guided logistics knowledge graph construction experiments. The files here are the working copies used by the local scripts and notebooks. The archived public release should be cited through the Zenodo dataset DOI listed in the repository root README.

## Directory structure

- `original_verifier_dataset/`
- `main_trainable_dataset/`
- `evaluation_only_interview_grounded_dataset/`

## Dataset roles

### `original_verifier_dataset`

This dataset corresponds to the earlier verifier setting used to evaluate ontology-grounded candidate verification under comparatively regular conditions. It serves as the initial feasibility-oriented evaluation setting.

### `main_trainable_dataset`

This dataset is the main training and evaluation resource used for the core comparison experiments. It includes harder entity-disambiguation cases in which positive and negative candidates may share the same relation and nearly the same sentence while differing in subject--object alignment.

### `evaluation_only_interview_grounded_dataset`

This dataset is used only for additional evaluation outside training. It provides a robustness check under a distinct sentence distribution.

## File format

Datasets are provided in JSONL format, with one example per line. This facilitates split management, direct integration with Python-based training pipelines, and transparent inspection of individual verification instances.

## How these datasets are used in this repository

- `main_trainable_dataset/` is the primary source used by the training notebook.
- `evaluation_only_interview_grounded_dataset/` is used by the optional benchmark evaluation and by the script-level end-to-end evaluation workflow.
- `original_verifier_dataset/` is included for completeness and for reproducing the earlier verifier setting described in the paper.

## Retained verifier input formats

Where format-specific files are included, the retained configurations are:

- `A_sentence_only`
- `B_relation_marker`
- `C_relation_entity_markers`
- `D_ontology_context`
- `F_full_ontology_context`

## Relation to the ontology

These datasets are designed to be used together with the ontology resource provided in the archive. Candidate relations and verifier inputs are ontology-grounded, and the reported experiments rely on this shared semantic structure.
