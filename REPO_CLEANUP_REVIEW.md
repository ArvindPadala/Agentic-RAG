# Repository cleanup review — two-pass scan

Date: 2026-10-19

**Cleanup completed: all four batches were approved and executed.**
The original two-pass findings and exact target lists below are retained as the
audit record. There were 33 separate cleanup commits, ending at `8bfeb92`.
Hugging Face has not been updated. Existing review-document edits are preserved.

## Original approval batches — all completed

| Batch | Proposal | Confidence / consequence |
|---|---|---|
| A | Remove generated local caches, `.DS_Store`, and the Lambda build ZIP | High: disposable/generated; not tracked |
| B | Remove nine unused tracked image assets | High that current code/docs do not use them; illustrations may still have archival value |
| C | Retire 16 legacy data/result files from the final source repository after deciding whether to preserve an external copy | High that the IPCC demo does not need them; older ingestion/evaluation workflows do use some of them |
| D | Optionally remove local `images/gradio_ui.png` | Unreferenced screenshot; unique visual contents/value not assessed |

B + C cover **25 tracked files**, totaling **11,056,844 bytes (~10.54 MiB)**
at the pre-cleanup baseline `ddde2a1`. That baseline tracked tree had 102 files /
11,770,753 bytes (~11.23 MiB).
These are uncompressed working-tree blob sizes, not Git-history savings. Deletion
will not erase Git/HF history or clean the separate HF repository automatically.

The user approved A/B first and C/D subsequently, including retirement of the
bundled legacy data/results. No external archive was created. Tracked deleted
files remain available in Git history. Application/development source and current
local state were retained as described below.

## Scan 1 — inventory and workflow mapping

Inspected tracked files, root/ignored directories, implementation imports and
paths, README/showcase/setup/validation docs, evaluation scripts, notebook
references, Docker/Compose/deployment selection, CI, tests and developer tooling.

Inventory commands included:

```sh
git ls-files
git ls-tree -rl HEAD
git status --short --ignored
git ls-files -ci --exclude-standard
git ls-files --others --exclude-standard
```

| Tracked area | Files | Bytes at baseline | Original disposition |
|---|---:|---:|---|
| Root files + `utils/` | 40 | 447,449 | Retain; optional AWS retirement is a separate feature decision |
| `.github/` | 1 | 1,062 | Retain CI |
| `.opencode/` | 10 | 37,480 | Retain developer review tools/config/tests |
| `supabase/` | 2 | 9,763 | Retain migration and RLS regression checks |
| `tests/` | 16 | 120,999 | Retain application tests, browser journey and SQL fixtures |
| `eval/` | 13 | 534,571 | Five historical outputs proposed in C; keep scripts, requirements and golden input |
| `images/` | 9 | 536,836 | Batch B: unreferenced assets |
| `documents/` | 10 | 9,894,177 | Batch C: legacy input corpus |
| `document_chunks/` | 1 | 188,416 | Batch C: non-default generated database |

Ignored/local-only directories were classified separately. `.git/` internals and
third-party `.opencode/node_modules/` contents are not application cleanup targets.
Secrets, conversation memory, PDFs, SQLite contents and CSV document/answer
contents were not opened. This is an inventory/reference audit, not a line-by-line
security audit or a proof that arbitrary user-selected paths are never used.

## Scan 2 — cross-check and false-positive removal

Rechecked candidates against runtime/default paths, static imports, scripts,
notebook source references, tests/fixtures, documentation, CI and both deployment
manifests. Used targeted content searches and tracked-file `git grep` searches;
the notebook was searched separately because `.gitattributes:3` marks it binary.

Results:

- **No byte-identical tracked duplicates:** compared Git blob IDs for all 102
  files. This does not rule out overlapping logic or equivalent images.
- **Nine tracked images have no consumer references** in tracked Python,
  Markdown, workflow, TypeScript or JavaScript files; notebook image/HTML-source
  searches also found no references. Git ignore exceptions are retention rules,
  not runtime consumers. Neither deployment manifest includes these assets.
- **`document_chunks` is a collection name, not necessarily a directory.**
  `gemini_helpers.py:33-61`, `agent.py:69-74`, `app.py:1100-1103`, Compose line 13,
  and evaluation scripts use `./chroma_db` by default. No default path was found
  consuming `document_chunks/chroma.sqlite3`. Contents were not examined, so it
  cannot be declared empty, redundant or safely reproducible.
- **Old PDFs are still notebook inputs.** `pipeline_setup.ipynb:342-344` uploads
  `documents/`. The IPCC workflow downloads its own pinned sources instead
  (`showcase_corpus.py:25-71`; `SHOWCASE.md:13-41`). Removing old PDFs retires the
  bundled inputs for that notebook, not the current showcase.
- **Historical CSV outputs have real consumers.** The answer-generation scripts
  write the three result CSVs; `eval/evaluate_ragas.py:83-92` reads all three.
  `eval/calculate_retrieval_metrics.py:154-174` writes `eval/metrics.json`, and
  `eval/evaluate_ragas.py:163-175` writes `eval/ragas_metrics.txt`. They can be
  regenerated only with suitable legacy data/models/providers, not for free or
  necessarily bit-for-bit. Archive if historical evidence matters.
- **Golden input is not a generated-result throwaway.** All evaluation runners
  consume `eval/golden_dataset.csv`; preserve it with the evaluation tooling.
- **AWS helper source is not dead code.** The notebook imports `lambda_helpers`
  and packages/deploys `ade_s3_handler.py` (`pipeline_setup.ipynb:229-289`). The
  notebook also supplies the legacy ingestion setup still mentioned by the CLI.
- **Publicly disabled does not mean unimported.** `app.py:48` imports
  `upload_handler.py`, and `agent.py:25` imports `visual_grounding_helper.py`.
  Removing these would break imports even when public uploads are denied.
- Tests build temporary/mock data (for example,
  `tests/test_runtime_privacy.py:53-60`, `tests/test_showcase_corpus.py:27-35`,
  `tests/browser_demo_ux.py:203-216`); no current test dependency on the bundled
  old PDFs, nine tracked illustrations or `document_chunks/chroma.sqlite3` was
  found. This static result still needs post-cleanup test execution.

## Batch A — generated local clutter

Delete only after approval; these are already untracked/ignored.

| Exact target | Reason | Effect |
|---|---|---|
| `__pycache__/` | Compiled Python cache | Recreated on import |
| `tests/__pycache__/` | Test bytecode cache | Recreated on import |
| `utils/__pycache__/` | Helper bytecode cache | Recreated on import |
| `eval/__pycache__/` | Evaluation bytecode cache | Recreated on import |
| `.pytest_cache/` | Previous test-run metadata | Next run loses last-failed/cache metadata only |
| `.DS_Store` | macOS directory metadata | No application use |
| `ade_lambda.zip` | Generated AWS deployment bundle; `.gitignore:6-7` identifies it as rebuildable | Notebook packaging must recreate it before a future Lambda deploy |

Avoid a broad `git clean` or blanket ignored-file deletion: that would also target
credentials, local memory, active indexes, showcase data and OpenCode dependencies.

## Batch B — unused tracked images

All nine below are currently unreferenced and excluded from deployment. Removing
them does not remove generated citation images: the current showcase uses
`showcase_data/citations/` (`showcase_corpus.py:105-109`), and the legacy grounding
helper creates image bytes for S3. Static absence of references does not assess
whether you want to reuse these illustrations in future documentation.

```text
images/architecture_1.png
images/files.png
images/flow_ade_handler.png
images/folders_s3.png
images/step6.png
images/step7.png
images/steps3-5.png
images/steps7-12.png
images/tool.png
```

Six of these are tracked despite matching current ignore rules (`git ls-files
-ci --exclude-standard`); `.gitignore` does not untrack existing files.

## Batch C — legacy data/results to retire or archive

### C1. Non-default database — 1 file, 188,416 bytes

```text
document_chunks/chroma.sqlite3
```

Generated binary state does not belong in the final source package. Recommend
removing it from version control; preserve a local/external copy first if its
contents may be valuable. It is distinct from the default `chroma_db/` database.
No assumption is made that one database is a duplicate of the other.

### C2. Older input PDFs — 10 files, 9,894,177 bytes

```text
documents/CT_Study_of_the_Common_Cold.pdf
documents/Common_cold_clinincal_evidence.pdf
documents/Evaluation_of_echinacea_for_the_prevention_and_treatment_of_the_common_cold.pdf
documents/Prevention_and_treatment_of_the_common_cold.pdf
documents/The_common_cold_a_review_of_the_literature.pdf
documents/Understanding_the_symptoms_of_the_common_cold_and_influenza.pdf
documents/Viruses_and_Bacteria_in_the_Etiology_of_the_Common_Cold.pdf
documents/Vitamin_C_for_Preventing_and_Treating_the_Common_Cold.pdf
documents/rag.pdf
documents/transformer.pdf
```

Not required for the IPCC showcase, but they supply the notebook's old upload
example. Retiring these should include a short notebook/docs note explaining
that users supply their own local PDFs for the legacy workflow. Do not delete
the folder indiscriminately if new user-provided files appear before execution.

### C3. Historical generated evaluation outputs — 5 files, 437,415 bytes

```text
eval/baseline_results.csv
eval/hybrid_results.csv
eval/reranker_results.csv
eval/metrics.json
eval/ragas_metrics.txt
```

Not needed by the running app or CI. Removal requires rerunning the answer
generation stage before Ragas evaluation, or restoring archived outputs.
`EVALUATION_REPORT.md` and README contain historical metric claims, so keep
provenance in an external archive or clearly label those claims historical when
removing their underlying artifacts. These are optional archival cleanup, not
broken or proven meaningless files.

## Batch D — optional local screenshot

`images/gradio_ui.png` is ignored/untracked and has no found source references.
It is a cleanup candidate if you do not want to preserve this screenshot. It is
not an automatically regenerated runtime cache; do not include it in A silently.

## Retain in the final repository

### Application and operational source

Keep all current Python source, including `utils/logger.py`,
`manage_public_corpus.py`, and `deploy.py`. Keep `showcase_sources.json`: it is
the source/identity manifest, not downloaded data. Keep both requirements files,
`Dockerfile`, `docker-compose.yml`, `Makefile`, `.github/`, `.env.example`,
`.gitignore`, `.dockerignore`, and `.gitattributes`.

`pipeline_setup.ipynb`, `lambda_helpers.py`, and `ade_s3_handler.py` are optional
for the IPCC host but form an existing AWS ingestion feature. Retain by default.
Removing these three together is a separate feature-retirement decision requiring
documentation/setup changes, not a garbage-file deletion.

### Tests, evaluation tooling and engineering records

Keep all 16 files under `tests/`, both SQL files under `supabase/`, the six
evaluation Python scripts, `eval/requirements-dev.txt`, and
`eval/golden_dataset.csv`. Keep `README.md`, `SHOWCASE.md`, `SUPABASE_SETUP.md`,
`EVALUATION_REPORT.md`, `UX_ROADMAP.md`, `UX_VALIDATION.md`, `production_review.txt`,
and `AGENTS.md`. The latter records are useful engineering history, not runtime
requirements; they can be moved into a docs directory later without discarding
their content. No reorganization is proposed in this deletion batch.

Keep the ten tracked `.opencode/` files. They implement the project's existing
review commands and sandbox checks. Its ignored package manifests and
`node_modules/` support the active coding environment; do not delete them as app
clutter during this work. They are already excluded from deployment.

### Preserve local-only state

| Target | Why retain locally |
|---|---|
| `.env` | Operator configuration/secrets; never publish |
| `memory.json` | Potentially valuable personal state; never publish or delete as cache |
| `chroma_db/` | Default legacy/local/evaluation index; clearing it loses indexed data |
| `showcase_data/` | Current provisioned IPCC PDFs/index/citations; rebuildable but deleting forces download/re-embedding and interrupts local readiness |
| `.git/` | Repository metadata/history; not a cleanup target |

## Deployment boundary after cleanup

Final Git repository and HF runtime package have different needs: tests, CI,
evaluation source and engineering docs remain useful in Git even though HF does
not need them. Continue using an explicit runtime source selection rather than
manually uploading the whole cleaned repository. Never include `.env`, personal
memory, local indexes, old PDFs/results, caches or archives.

Before eventual HF work, validate the selected requirements and startup config
on the intended HF environment and enable `SHOWCASE_ENABLED=true`. Cleaning this
repository alone will not update or remove the old manually uploaded HF files.
No Git-to-HF automation or remote change is included in this proposal.

## Original execution plan

This is the pre-cleanup plan. The actual completed scope and checks follow it;
not every prospective follow-up listed here was performed.

1. Confirm exact batches and whether C needs an external archive or local-only
   retention. Recheck Git status and candidate paths for new user changes.
2. Remove only approved paths. If C is approved, add appropriate ignore rules
   for retired generated/input artifacts and document the legacy workflow impact.
   Preserve unrelated existing edits.
3. Run `PYTHONDONTWRITEBYTECODE=1 make test`, `make lint`, and `git diff --check`.
   Verify source packaging still excludes local data and run the showcase/browser
   and production-image checks appropriate to the final change. Run SQL checks
   if memory/schema behavior changes or as part of the later release gate.
4. Inspect staged/unstaged diffs and relevant untracked files; show the final
   removed/retained list. Commit/deploy only when requested.
5. Reconcile the separate HF artifact inventory and deploy the intended release
   after the user authorizes that remote work, then run its live checklist.

### Checks for the original proposal

Both inventory/reference passes were completed. No application or dependency
changes were made, so application tests were not rerun for this report. No
deletion behavior or post-cleanup runtime has been tested yet. Inspected the
existing tracked diffs and the new report; the staged diff is empty.
`git diff --check` and
`git diff --no-index --check /dev/null REPO_CLEANUP_REVIEW.md` passed.

## Completed execution and verification

- A/B: 16 individual commits, `3dc8c33` through `d08226c`.
- C/D: 17 individual commits, `b929f33` through `8bfeb92`.
- Seven Batch A targets and the Batch D screenshot were ignored/untracked;
  their individual commits record local removal in `CLEANUP_LOG.md`.
- The 25 tracked B/C files were each deleted in a separate commit.
- After each cleanup group, the full suite passed with **116 tests and 40
  dependency/fixture warnings**. Exact command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' make test`.
- `PYTHONDONTWRITEBYTECODE=1 make lint` reported **0 configured issues**, and
  `git diff --check` passed. These results were recorded during execution and
  were not rerun during this documentation reconciliation.
- Final path checks confirmed all approved targets absent and retained local
  configuration/memory, default index and showcase data present. Source, tests,
  requirements and golden input were unchanged by C/D cleanup.
- No new browser, SQL, Docker, hosted-provider or benchmark run was performed for
  these artifact deletions. No files were pushed or deployed to Hugging Face.
- Ignore-rule updates were proposed in the original plan but have not been
  applied; regenerated evaluation outputs would currently appear as untracked
  files. This is a follow-up, not completed work.

## Next evaluation prerequisite

Choose the evaluation corpus before generating fresh answers or metrics.
`eval/calculate_retrieval_metrics.py:39-40` still selects the historical golden
dataset and `./chroma_db` / `document_chunks`. `eval/evaluate_ragas.py:83-92`
expects the three deleted answer CSVs to be regenerated first. The current IPCC
showcase uses a separate content-addressed index under `showcase_data/`.

For an IPCC evaluation, use a matching reviewed question/evidence dataset and
explicitly select that showcase index; running the old scripts unchanged would
not measure the current IPCC demo. For a legacy-corpus evaluation, validate the
retained golden input against the existing local index first. The cleanup itself
does not establish either dataset's suitability or rerun pipeline evaluation.
