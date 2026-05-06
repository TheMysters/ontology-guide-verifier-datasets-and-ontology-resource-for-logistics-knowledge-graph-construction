# Ontology-Guided Logistics Verifier

This repository accompanies the paper on ontology-guided logistics knowledge graph construction. It provides the released code, local notebook entry points, and a reproducibility workflow around the ontology-guided verifier.

## Zenodo archives

- Ontology DOI: `https://doi.org/10.5281/zenodo.20043611`
- Datasets DOI: `https://doi.org/10.5281/zenodo.20051042`

The GitHub repository is the code and workflow companion. The stable archived artifacts are the Zenodo dataset release and the Zenodo ontology release.

## Repository contents

- `datasets/`: released verifier datasets used in the paper
- `ontology/`: local copy of the ontology used by the scripts and notebooks
- `scripts/`: training, evaluation, and end-to-end pipeline scripts
- `models/`: local directory for trained verifier checkpoints
- `results/`: local directory for training and end-to-end outputs
- `notebooks/end_to_end/`: local end-to-end demonstration notebook
- `notebooks/colab/`: retraining notebook for one retained verifier configuration
- `requirements.txt`: Python dependencies for the released scripts and notebooks

## Installation

Create a virtual environment, activate it, and install the required dependencies.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## What this repository is for

This repository is intended to let a reader:

1. inspect the released datasets and ontology;
2. retrain at least one verifier model locally or in Colab;
3. run the ontology-guided end-to-end pipeline with a local trained checkpoint;
4. inspect generated candidates, model predictions, accepted triples, and exported results.

The reported trained checkpoints are not distributed here because they are too large for GitHub. The expected workflow is therefore to place an existing local checkpoint under `models/` or to retrain one from the released datasets.

## Main released resources

- original verifier dataset
- main trainable dataset
- evaluation-only interview-grounded dataset
- `internal-logistics-v2.owl`

## Suggested starting points

- end-to-end example:
  - `notebooks/end_to_end/end_to_end_ontology_format_evaluation.ipynb`
- training example:
  - `notebooks/colab/train_C_relation_entity_markers_main_trainable_dataset.ipynb`
- script entry point for end-to-end evaluation:
  - `scripts/end_to_end_verifier_evaluation.py`

## Minimal usage

After creating a virtual environment and installing the dependencies, a minimal workflow is:

1. inspect the released datasets in `datasets/`;
2. open `notebooks/colab/train_C_relation_entity_markers_main_trainable_dataset.ipynb` locally or in Colab to retrain one verifier model on the published main trainable dataset and save it under `models/main_trainable_dataset/C/`;
3. open `notebooks/end_to_end/end_to_end_ontology_format_evaluation.ipynb` to load a local trained model from `models/`, run ontology-guided preprocessing on a small set of input sentences, generate candidate triples, and inspect validated outputs.

The same end-to-end workflow can also be run as a script once at least one trained local model is available. For example:

```powershell
python scripts/end_to_end_verifier_evaluation.py
```

The released end-to-end notebook supports both a flat checkpoint layout and the archived local layout already used in this repository. Typical locations are:

```text
models/
  A_sentence_only/
  B_relation_marker/
  C_relation_entity_markers/
  D_ontology_context/
  F_full_ontology_context/

models/
  main_trainable_dataset/
    C/
      models/
        C_relation_entity_markers_main_trainable_dataset/
```

Any missing model directory is skipped automatically during evaluation.

## Interpreting the reported scores

The reported `~75%` F1-score in the paper refers to candidate-level evaluation on the released main trainable test split for the `C_relation_entity_markers` configuration. It is not the expected success rate of a short end-to-end notebook demo on a handful of manually chosen sentences.

The end-to-end notebook answers a different question: given an input sentence, what candidates are generated upstream, which of them are retained by the trained verifier, and what accepted triples are finally exported. Its behavior therefore depends on both candidate generation and model scoring.

## License

The code in this repository is released under the Apache-2.0 License. Dataset and ontology artifacts are distributed with their accompanying documentation and release notes.
