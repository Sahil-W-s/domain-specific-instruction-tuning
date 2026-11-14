# Source Scripts

Source files recovered from project uploads; numbered filenames are preserved.

## Main workflow

- Chunking: `08b_create_semantic_chunks.py`
- Initial generation: `29_generate_production_batch.py`
- Auditing: `31_audit_generated_examples.py`, `32_audit_evidence_blocks.py`
- Improved generation and pilot: scripts 38, 41–43
- Production v2 and reconciliation: scripts 45–61

## Diagnostics

The `diagnostics/` subdirectory contains semantic-judge tests, Groq judge benchmark iterations, and OpenRouter endpoint/text-output checks. These document development/testing and are not all required for the principal production workflow.

## Execution warning

Scripts expect the original project directory structure, including `data/frozen/`, `data/interim/`, and `scripts/`. The repository includes code, not the private/source-document corpus. Review paths and dependencies before running. Never execute generation or reconciliation against the frozen historical data unless working from a separate copy.
