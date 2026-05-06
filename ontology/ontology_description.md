# Ontology Description

This archive contains the ontology resource used in the experiments reported in the associated submission.

## Main file

- `internal-logistics-v2.owl`: main ontology used for ontology-guided candidate construction and knowledge graph structuring.

## Scope

The ontology models an internal logistics domain for knowledge graph construction from documentation and structured operational data. It provides the domain vocabulary, class structure, object properties, datatype properties, and semantic constraints used during candidate generation and instance-level knowledge graph construction.

The ontology includes concepts for products, measurements, units, product families, physical locations, logistics agents, and logistics processes, together with object and datatype properties used to represent product attributes, measurement values, storage relations, transport relations, and related business semantics.

## Design basis

The upper-level organization of the ontology is inspired by the Industrial Ontologies Foundry Core Ontology (IOF Core), which provides reusable conceptual distinctions for industrial domains:

- Industrial Ontologies Foundry: https://www.industrialontologies.org/
- IOF ontology repository: https://github.com/iofoundry/ontology/

In particular, the ontology relies on the high-level distinctions between `MaterialEntity`, `Agent`, and `Process`. These categories provide the conceptual basis for distinguishing logistics objects, actors, and activities.

The ontology also uses SKOS for classification-oriented structures such as product families and subfamilies:

- SKOS reference: https://www.w3.org/2004/02/skos/

## Main modeling choices

Measurements are represented through explicit nodes that separate the measurement type, the numerical value, and the unit. This supports heterogeneous operational sources in which dimensions, weights, and related quantities may be expressed with different units or naming conventions.

Classification-oriented concepts such as product families and subfamilies are represented through lighter taxonomy-oriented structures that can be aligned with documentation-derived semantics.

## Role in the framework

The ontology is used:

1. as a vocabulary source for ontology-aware mention detection;
2. as a constraint source for candidate construction through domain/range and compatibility restrictions;
3. as the target structure for schema-level interpretation and instance-level graph construction.

## Notes

The `internal-logistics-v2.owl` file is the canonical ontology version for this archive. Other ontology variants present in the development workspace were intermediate or legacy files and are not part of the release artifact.
