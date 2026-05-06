# End-to-End Notebook

This folder contains the end-to-end evaluation notebook used to illustrate the workflow from ontology-guided preprocessing to verifier-based candidate validation.

## Notebook

- `end_to_end_ontology_format_evaluation.ipynb`

## Purpose

The notebook is intended as a reproducibility-oriented walkthrough of the evaluation pipeline rather than as a source of precomputed trained models. It loads the released ontology from `ontology/`, selects a default sentence from the released main trainable dataset, and applies a locally available trained verifier checkpoint from `models/`. Outputs are written under `results/end_to_end_eval/`.
