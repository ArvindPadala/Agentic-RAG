# Approved repository cleanup

Date: 2026-10-19

The user approved batches A (generated local clutter) and B (unused tracked
images), with one commit after each target's deletion. Batch A targets are
ignored/untracked, so their commits record the local removal here; they do not
represent tracked-file deletions. Batch B deletions are recorded directly in Git.

## Batch A — local removals

| Target removed | Reason |
|---|---|
| `__pycache__/` | Root Python bytecode cache; recreated on ordinary imports |
| `tests/__pycache__/` | Test bytecode cache; recreated on ordinary imports |
| `utils/__pycache__/` | Helper bytecode cache; recreated on ordinary imports |
| `eval/__pycache__/` | Evaluation bytecode cache; recreated on ordinary imports |
| `.pytest_cache/` | Previous test-run cache; tests can run without it |
| `.DS_Store` | macOS directory metadata; no application use |
| `ade_lambda.zip` | Generated Lambda bundle; recreate through notebook packaging before deployment |

## Batches C and D — subsequent approved cleanup

The user subsequently approved deleting all 16 Batch C files and the Batch D
screenshot, with one commit after each deletion. Batch C's database, ten older
PDFs and five historical evaluation outputs are recorded as individual Git
deletions. No external archive was created; tracked files remain in Git history.

| Local target removed | Reason |
|---|---|
| `images/gradio_ui.png` | Untracked, unreferenced screenshot; removal recorded here |

The legacy notebook now requires user-supplied PDFs in its local `documents/`
input directory. Evaluation scripts, requirements and `eval/golden_dataset.csv`
are retained. Regenerate answer/result files before running Ragas, and ensure
the evaluation dataset matches the indexed corpus. The retained historical
golden dataset is not a new IPCC benchmark. Existing published metric values
remain historical; this cleanup does not rerun or validate those measurements.
