# Scripts

This directory contains the main scripts used to train, evaluate, and document the released verifier workflow. The primary public entry points are the training script, the end-to-end evaluation script, and the helper script for format-level evaluation. Additional generation scripts are included for transparency about dataset construction, but some of them reflect the original development workflow and are not required to reproduce the released experiments from the published assets.

## Recommended entry points

If you only want to understand or rerun the public workflow, start with:

- `train_ontology_format_verifier.py`
- `end_to_end_verifier_evaluation.py`
- `evaluate_verifier_format_dataset.py`

The remaining scripts are mainly provenance scripts showing how the released datasets were constructed.

## Primary runnable scripts

- `train_ontology_format_verifier.py`: trains verifier models for retained input formats from the released datasets;
- `end_to_end_verifier_evaluation.py`: runs ontology-guided preprocessing, loads local trained models from `models/`, and exports validated triples and metrics;
- `evaluate_verifier_format_dataset.py`: evaluates one trained verifier model on a released JSONL dataset file.

## Dataset generation and provenance scripts

- `generate_ontology_aware_verifier_dataset.py`: generates the original ontology-aware verifier dataset;
- `create_interview_augmented_verifier_dataset.py`: generates interview-augmented verifier data;
- `create_balanced_mixed_verifier_v4_c_rewarded.py`: generates the main trainable dataset used in the core experiments;
- `create_natural_verifier_benchmark_100.py`: generates the evaluation-only interview-grounded benchmark.
- `build_logistics_weak_dataset.py`: candidate-generation helper used by the end-to-end evaluation workflow.
- `end_to_end_verifier_evaluation_colab.py`: compatibility wrapper for Colab-oriented imports based on the same released workflow as `end_to_end_verifier_evaluation.py`.
