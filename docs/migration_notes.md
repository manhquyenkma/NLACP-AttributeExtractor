# Migration Notes — Repository Reorganization
*Date: 2026-03-19*

## Summary

The full NLACP-AttributeExtractor repository was reorganized from a flat research layout into a clean, layered structure with a proper Python package `nlacp`.

---

## What Changed

### New Package: `nlacp/`
All reusable library code moved into a proper Python package with 6 sub-packages:

| Sub-package | Contents |
|---|---|
| `nlacp/pipeline/` | `pipeline.py` (was `src/nlp_engine.py`) |
| `nlacp/extraction/` | `relation_candidate.py`, `env_extractor.py`, `short_name_suggester.py` |
| `nlacp/normalization/` | `category_identifier.py`, `namespace_assigner.py`, `data_type_infer.py` |
| `nlacp/mining/` | `attribute_cluster.py`, `namespace_hierarchy.py` |
| `nlacp/evaluation/` | `evaluator.py` |
| `nlacp/io/` | `dataset_builder.py` |

### Scripts: `scripts/`
| Old Path | New Path |
|---|---|
| `main.py` | `scripts/run_pipeline.py` (rewritten as clean entry point) |
| `src/run_eval.py` | `scripts/run_eval.py` |
| `src/run_eval_vact.py` | `scripts/run_eval_vact.py` |
| `annotate_helper.py` | `scripts/annotate.py` |
| `src/generate_vact_dataset.py` | `scripts/generate_vact_dataset.py` |
| `data/generate_annotation_sheet.py` | `scripts/generate_annotation_sheet.py` |
| `data/convert_litroacp.py` | `scripts/convert_litroacp.py` |
| `data/filter_env.py` | `scripts/filter_env.py` |

### Tests: `tests/`
| Old Path | New Path |
|---|---|
| `test_pipeline.py` | `tests/test_pipeline.py` |
| `test_extractor_fix.py` | `tests/test_extractor_fix.py` |
| `verify_fixes.py` | `tests/verify_fixes.py` |

### Experiments: `experiments/`
| Old Path | New Path |
|---|---|
| `src/cnn_classifier.py` | `experiments/cnn/cnn_classifier.py` |
| `mining/attribute_cluster_glove.py` | `experiments/glove_clustering/attribute_cluster_glove.py` |
| `src/business_process_modeler.py` | `experiments/business_process/business_process_modeler.py` |

### Legacy: `legacy/`
| Old Path | New Path |
|---|---|
| `LitroACP/` | `legacy/litroacp/` |
| `nckh/` | `legacy/nckh/` |

### Outputs: `outputs/`
| Old Path | New Path |
|---|---|
| `dataset/policy_dataset.json` | `outputs/policies/policy_dataset.json` |
| `dataset/attribute_clusters.json` | `outputs/clusters/attribute_clusters.json` |
| `dataset/namespace_hierarchy.json` | `outputs/hierarchy/namespace_hierarchy.json` |
| `*.txt` (root) | `outputs/logs/*.txt` |

### Data: `data/` (unchanged)
- `data/raw/` — source JSONL corpora (ACRE, IBM, T2P, Cyber, Collected)
- `data/annotated/` — gold standard JSON annotation files
- `data/annotation_sheet.csv` — human annotation sheet
- `data/annotation_template.json` — moved from `dataset/`

---

## Import Changes

All `sys.path.insert(0, ...)` hacks replaced with proper package imports:

```python
# Old (broken after move):
sys.path.insert(0, 'src')
from relation_candidate import ...

# New (clean package import):
from nlacp.extraction.relation_candidate import ...
from nlacp.pipeline.pipeline import process_sentence
```

## Path Changes

| Component | Old hardcoded path | New path |
|---|---|---|
| `dataset_builder.py` | `dataset/policy_dataset.json` | `outputs/policies/policy_dataset.json` |
| `attribute_cluster.py` | `dataset/attribute_clusters.json` | `outputs/clusters/attribute_clusters.json` |
| `namespace_hierarchy.py` | `dataset/namespace_hierarchy.json` | `outputs/hierarchy/namespace_hierarchy.json` |
| `run_eval.py` output | root dir | `outputs/logs/evaluator_results.txt` |

---

## What Was Kept

- All `docs/` files kept as-is
- All `data/raw/` and `data/annotated/` datasets kept unchanged
- `nckh/` and `LitroACP/` preserved in `legacy/` (not deleted)
- Experiments kept in `experiments/` (not deleted)
- `dataset/` folder kept (still has `policy_dataset.json`)

---

## Entry Points After Migration

```bash
# Run full interactive pipeline
python scripts/run_pipeline.py

# Process single sentence
python scripts/run_pipeline.py --sentence "Nurses can read records during business hours."

# Run evaluation
python scripts/run_eval.py
python scripts/run_eval_vact.py

# Run tests
python tests/test_pipeline.py
python tests/test_extractor_fix.py

# Manual annotation
python scripts/annotate.py
```

---

## Validation Results

| Check | Result |
|---|---|
| All package imports | ✅ PASS |
| `tests/test_pipeline.py` assertions | ✅ PASS |
| `tests/test_extractor_fix.py` 8/8 | ✅ PASS |
