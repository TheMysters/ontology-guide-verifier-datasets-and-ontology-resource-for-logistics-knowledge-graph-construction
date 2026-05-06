# Original Verifier Dataset

## Purpose

This dataset corresponds to the original verifier setting used in the study before the introduction of the stricter main trainable dataset. It provides an earlier evaluation setting in which ontology-grounded candidate verification remains comparatively regular.

## Composition

The dataset is derived from interview-based source material together with ontology-aware synthetic generation. It is intended to test whether ontology-grounded candidate verification can be learned effectively under comparatively regular conditions.

## Files

- `ontology_aware_train.jsonl`
- `ontology_aware_val.jsonl`
- `ontology_aware_test.jsonl`
- `ontology_aware_all.jsonl`
- `stats.json`

## Format

The dataset is provided in JSONL format, with one example per line. Each example contains a sentence, a candidate relation, a verifier-oriented textual representation, and a binary label indicating whether the candidate is considered valid in context.

## Role in the experiments

In the paper, this dataset is used as the original verifier dataset. It supports the initial feasibility-oriented evaluation reported before the harder main trainable dataset. It is not the default dataset used by the public notebook entry points, but it is kept here so the earlier evaluation setting remains inspectable and reproducible.
