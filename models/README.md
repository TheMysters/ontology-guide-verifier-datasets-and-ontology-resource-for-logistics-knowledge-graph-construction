# Local Models

This directory is intended to contain locally available trained verifier models.

The public repository does not include the reported trained checkpoints because they are too large for GitHub. To run the notebooks or scripts locally, place trained Hugging Face model directories here.

## Recommended layout

```text
models/
  A_sentence_only/
  B_relation_marker/
  C_relation_entity_markers/
  D_ontology_context/
  F_full_ontology_context/
```

Each model directory should contain the files expected by `transformers`, including at least:

- `config.json`
- tokenizer files
- model weight files

The Colab training notebook can be used to retrain at least the `C_relation_entity_markers` model from the released datasets.

## Alternative layout for archived local experiments

If existing local checkpoints are already organized by dataset and format, they can also be kept in a nested structure such as:

```text
models/
  main_trainable_dataset/
    C/
      models/
        C_relation_entity_markers_main_trainable_dataset/
      results/
        training/
```

The released end-to-end notebook is configured to work directly with such a local checkpoint path when selected explicitly.

## Current repository layout

This repository already uses the archived local layout for the released `C_relation_entity_markers` example:

```text
models/
  main_trainable_dataset/
    C/
      models/
        C_relation_entity_markers_main_trainable_dataset/
      results/
        training/
```

The public end-to-end notebook looks for this checkpoint first.
