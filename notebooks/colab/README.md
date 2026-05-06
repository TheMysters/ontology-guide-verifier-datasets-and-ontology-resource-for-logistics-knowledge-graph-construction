# Colab Training Notebook

This folder contains the training notebook used to retrain one of the reported verifier configurations from the released datasets. It is compatible with both Google Colab and a local repository checkout.

## Notebook

- `train_C_relation_entity_markers_main_trainable_dataset.ipynb`

## Purpose

The notebook provides a minimal retraining entry point for the relation-plus-entity-marked verifier on the released main trainable dataset. It saves the resulting checkpoint under `models/main_trainable_dataset/C/models/` and the associated metrics under `models/main_trainable_dataset/C/results/training/`.
