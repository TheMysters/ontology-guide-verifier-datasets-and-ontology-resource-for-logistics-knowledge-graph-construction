from __future__ import annotations

"""
Compatibility wrapper for Colab-oriented imports.

The public repository exposes the released end-to-end workflow through
`end_to_end_verifier_evaluation.py`. This file re-exports the same public
helpers so that notebooks or scripts that import the older Colab-oriented
module name continue to work with the released repository layout.
"""

from end_to_end_verifier_evaluation import (  # noqa: F401
    DEFAULT_GOLD_JSONL,
    DEFAULT_INPUT_TEXT,
    DEFAULT_OUTPUT_DIR,
    DIAGNOSTIC_FORMATS,
    OWL_PATH,
    PRIMARY_FORMATS,
    VALID_PROXY_FLAGS,
    INVALID_PROXY_FLAGS,
    build_candidate_rows,
    candidate_to_row,
    chunk_triples_by_sentence,
    compute_binary_metrics,
    evaluate_format,
    evaluate_many,
    format_candidate_rows,
    full_uri,
    gold_end_to_end_metrics,
    load_gold_triples,
    load_module,
    load_transformer_model,
    predict_texts,
    prepare_inputs,
    proxy_gold_label,
    proxy_metrics,
    summarize_predictions,
    triples_from_predictions,
    write_chunked_kg_exports,
    write_csv,
    write_graph_html,
    write_jsonl,
    write_ttl,
)
