# Ontology Resource

This directory contains the local copy of the ontology used by the released scripts and notebooks.

## Files

- `internal-logistics-v2.owl`: ontology used for ontology-guided preprocessing, candidate construction, and semantic typing.
- `ontology_description.md`: textual description of the ontology resource.

## Relation to the Zenodo release

The stable citable ontology archive should be referenced through the ontology Zenodo DOI listed in the repository root README. The files in this directory are the working copies used by the local reproducibility workflow.

## How this folder is used

The training and end-to-end scripts load `internal-logistics-v2.owl` directly from this directory. If you replace the ontology file locally, downstream preprocessing and candidate generation behavior may change.
