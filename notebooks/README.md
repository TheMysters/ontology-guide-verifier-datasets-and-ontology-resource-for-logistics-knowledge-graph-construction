# Notebooks

This directory contains the two notebook entry points intended for reproducibility. Together, they cover the two main public usage modes of the repository: retraining one verifier configuration from the released datasets and running the ontology-guided end-to-end evaluation workflow with locally available trained models.

## Recommended reading order

1. start with the training notebook if you want to reproduce one retained verifier checkpoint from the released data;
2. then open the end-to-end notebook if you want to inspect how ontology-guided preprocessing and model validation interact on concrete example sentences.

## End-to-end notebook

- `end_to_end/end_to_end_ontology_format_evaluation.ipynb`

This notebook illustrates the evaluation workflow from ontology-guided preprocessing to verifier-based candidate validation on the released evaluation setting.
It expects trained local model directories under `models/` and writes outputs under `results/end_to_end_eval/`.

## Colab training notebook

- `colab/train_C_relation_entity_markers_main_trainable_dataset.ipynb`

This notebook provides a Colab-oriented example for retraining the relation-plus-entity-marked verifier on the released main trainable dataset.
It can also be run locally from the repository root and saves the trained model under `models/main_trainable_dataset/C/`.
