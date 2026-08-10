# EA-GPT

LLM-driven Enterprise Architecture generation using IBM's Business System Planning (BSP)
methodology. Given a list of business processes (and optionally supporting documentation),
EA-GPT extracts a CRUD matrix, clusters processes and data entities into candidate systems
via a BSP algorithm, and produces a full architecture analysis: system descriptions with
build/buy/hybrid recommendations, EA-principle compliance evaluation, and an optional
As-Is vs. To-Be gap analysis against your current application landscape.

This is a research prototype built for an MSc thesis (DEI, Instituto Superior Técnico,
Universidade de Lisboa) — not a production tool.

## How it works

1. **Extraction** — an LLM reads your process list (+ optional context document) and
   produces a CRUD matrix: which processes Create/Read/Update/Delete which data entities,
   plus a classification of each process as `atomic` (single-system ACID transaction) or
   `end_to_end` (legitimately spans departments/systems).
2. **Process-type review** — before clustering, you review and can override the LLM's
   atomic/end_to_end classification for each process. This single field has more influence
   over the final cluster boundaries than anything else in the pipeline, so it's a
   human-in-the-loop checkpoint rather than a silent LLM decision.
3. **BSP clustering** — a deterministic algorithm (`src/utils/bsp.py`) reorders the matrix
   for block-diagonal density and groups processes/entities into clusters, honoring the
   atomicity constraint (an atomic process's written entities are force-merged into one
   cluster) and end-to-end processes (allowed to span clusters).
4. **Iterative refinement** — you can add free-text architectural constraints; the LLM
   converts them into numeric affinity biases and re-runs BSP until you're satisfied.
5. **Systems analysis** — the LLM names each final cluster, describes its responsibility,
   and recommends Build/Buy/Hybrid with real market alternatives.
6. **EA compliance** — the final clustering is evaluated against a baseline set of EA
   principles.
7. **As-Is vs. To-Be (optional)** — upload your current landscape as its own CRUD matrix
   (own process/entity vocabulary, own system boundaries). ISA quality metrics (RSF, NAIEF,
   LCOISF, CPSMF, DIIEF) are computed independently for both architectures, and the LLM
   produces a migration plan by comparing the two **at the system level** — matching
   systems by what they actually do, not by trying to reconcile individual process/entity
   names across the two vocabularies.

## Project structure

```
src/
  app.py                  FastAPI + WebSocket app — the actual entrypoint, run via uvicorn
  main.py                 Obsolete CLI entrypoint, kept for reference — not used
  static/index.html       Web frontend (single file, no build step)
  utils/
    bsp.py                BSP algorithm, ISA metrics, Pydantic schemas
    generator.py           LLM calls (extraction, weights, systems analysis,
                           compliance, As-Is/To-Be comparison)
    matrix.py              CRUD matrix data structure + extraction schema
  resources/
    prompt.txt              Extraction prompt (process_type + CRUD rules)
    bsp_prompt.txt           EA-weight computation prompt
    description_prompt.txt   Systems analysis prompt
    compliance_prompt.txt    EA compliance prompt
    comparison_matrices_prompt.txt   As-Is vs To-Be gap analysis prompt
    ea_principles.txt        Baseline EA principles used for compliance checks
```

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Running it

```bash
cd src
python -m uvicorn app:app --reload --port 8080
```

Then open `http://localhost:8080`.

`main.py` is an obsolete terminal-driven copy of the same pipeline, kept in the repo for
reference, it is not the way this project is run.

## Testing

The BSP algorithm and ISA metrics (`src/utils/bsp.py`) are pure, deterministic Python —
no LLM calls — so they're covered by a `pytest` suite that needs no API key and runs in
seconds:

```bash
pip install -r requirements.txt   # includes pytest
pytest
```

`pytest.ini` (repo root) points pytest at `src/` so `from utils.bsp import ...` resolves
the same way it does at runtime. Tests live in `tests/`:

- `test_crud_weights.py` — CRUD operation weighting and entity-ownership priority (C > U > D > R)
- `test_atomicity_constraint.py` — the atomic-process co-location rule, including a
  regression test for mislabeling a cross-department process `atomic` collapsing the
  whole matrix into one cluster (see the Changelog)
- `test_pure_reader_absorption.py` — pure-reader relocation and the `end_to_end` exemption
- `test_isa_metrics.py` — hand-computed RSF/NAIEF/LCOISF/CPSMF/DIIEF on small known cases

There are no tests for the LLM-calling paths (`generator.py`, the pipeline in `app.py`) —
those would need mocked or recorded model responses, which isn't set up yet.

## Input formats

- **Process list**: comma-separated text, or a path to a `.csv` with one process name per row.
- **Context document** (optional): a `.pdf` or `.txt` file, or pasted text. If omitted, the
  LLM falls back on general knowledge of what each named process typically involves.
- **As-Is CRUD matrix** (optional, for the gap-analysis step): a CSV with columns
  `Process,Entity,Operation,System,ProcessType` — one row per CRUD entry in your current
  landscape's own matrix. `Operation` is one of `C/R/U/D`; `ProcessType` is one of
  `atomic/end_to_end/ambiguous`.

## Versioning

This project uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`), tracked
via git tags and the Changelog below:

- **MAJOR** — reserved for a stable/complete release (thesis submission). Everything before
  that is `0.x.y`.
- **MINOR** — a meaningful capability change: a new pipeline stage, a redesigned flow, a
  removed feature. This is the one to bump for the kind of "big change" worth writing a
  changelog entry for.
- **PATCH** — bug fixes, prompt tweaks, small UI fixes that don't change what the tool does.

Workflow for a big change:

```bash
git commit -m "..."                      # commit the change as usual
git tag -a v0.4.0 -m "short summary"      # tag the commit
git push && git push --tags               # if/when pushing to GitHub
```

Then add an entry to the Changelog below in the same commit or the next one. The tag lets
you check out or reference the exact code state for any version later (useful for pointing
a thesis chapter at "the version evaluated in Section 4.2"); the Changelog is the
human-readable summary of what changed and why.

## Changelog

### v0.4.0 — 2026-08-10
**Changed**
- **`src/utils/bsp.py` restructured into an explicit pipeline.** The old
  `_extract_blocks` did five different jobs in one ~155-line function
  (process grouping, entity grouping, atomic union-find, cluster pairing,
  leftover-entity assignment) with four nested closures capturing shared
  state. It's now a sequence of small, named, single-responsibility
  functions (`_weighted_jaccard`, `_average_linkage_group`,
  `_group_processes`/`_group_entities`, `_UnionFind`, `_enforce_atomicity`,
  `_pair_groups_to_clusters`, `_assign_unclaimed_entities`), orchestrated
  by `_extract_blocks` itself. Still one file, no package split. Comments
  for all new/rewritten logic are in simple European Portuguese, matching
  a style already present in a few places in the file before this change.
  **Behaviour is unchanged by default** — all 14 existing tests pass
  without modification, and `run_bsp`'s existing call shape and
  `BSPResult`'s existing fields are untouched; every addition below is
  additive and defaulted off.

**Added**
- **An explicit, ordered decision table (`decide()`)** for every
  merge/split call, replacing an implicit rule buried in code layout with
  a named priority order: an architect-confirmed override wins over
  everything, then the atomic force-merge rule (unchanged), then ordinary
  similarity-vs-threshold grouping. Grad, B., "Decision Tables in Systems
  Design," Session 19, Digest of Technical Papers, 1962 ACM National
  Conference, pp. 76-77.
- **`confirmed_overrides` parameter on `run_bsp`** (`{process_name:
  {entity_name, ...}}`): lets an architect-confirmed constraint exempt a
  specific entity from an atomic process's forced union, for that run
  only — `process_types` itself is never mutated, so the exemption is
  auditable as a deliberate override, not a silent reclassification.
- **`BSPResult.pending_conflicts`**: for every entity an atomic process is
  force-merging, checks whether a targeted, negative `ea_weights` value
  for that process/entity-owner pair would have flipped the ordinary
  grouping decision — using the *same* similarity-vs-threshold test as
  everything else, not a separate invented cutoff. Never applied
  automatically; only ever surfaced for confirmation. Caught and fixed a
  real bug during implementation testing: the first version could report
  a process "conflicting" with an entity it owns itself, and — since
  entity-to-entity comparisons never use `ea_weights` (an existing rule,
  unchanged) — was blind to whether any constraint actually targeted the
  pair at all. Fixed before landing; both cases are now covered by manual
  verification.
- **`BSPResult.hub_flags`**: flags any process/entity touching more than
  half of the opposite axis, before reordering — detection/reporting
  only in this pass, does not yet change clustering behaviour. Bureš,
  Cerny, Frajtak & Ahmed, "Testing the Consistency of Business Data
  Objects Using Extended Static Testing of CRUD Matrices," Cluster
  Computing 22(S4), S963-S976, 2019.
- **`derive_threshold()` and `run_bsp(adaptive_threshold=..., adaptive_k=...,
  adaptive_min_pairs=...)`**: an opt-in alternative to the fixed
  `density_threshold=0.5`, computing mean + k·stdev over the matrix's own
  pairwise-similarity distribution instead of a constant chosen without
  knowing what the data would look like. Off by default — `k=1.0` is an
  explicitly unvalidated starting point, not yet tested against real
  matrices. Akkasi, Seyyedi & Shams, "Presenting A Method for Benchmarking
  Application in the Enterprise Architecture Planning Process Based on
  Federal Enterprise Architecture Framework," IEEE Xplore.
- **`classify_changes()`**: diffs two iterations' clusters and labels each
  changed process/entity `"atomicity_override"` or `"ordinary"`. Exported
  but not called from `run_bsp` — not wired into `app.py`/the frontend
  yet, ready for a later pass.

**Not yet done (tracked as next steps, not silently deferred)**
- None of the new capabilities above are reachable from the web app —
  `app.py` and `src/static/index.html` are unchanged. `confirmed_overrides`,
  `pending_conflicts`, and `classify_changes` need UI/orchestration wiring
  before an architect can actually use them.
- `adaptive_threshold` needs empirical testing against real matrices to
  pick a sensible `k` (and decide whether it should become the default)
  before it's trustworthy.
- The extraction prompt's "every process must have 4-6 CRUD entries" rule
  (`src/resources/prompt.txt`, `src/utils/matrix.py`'s `MatrixResult`
  docstring) is under review — it may be forcing fabricated/truncated
  CRUD entries to hit an artificial density band, which the adaptive
  threshold may make unnecessary. Not changed in this version.

### v0.3.0 — 2026-08-08
**Added**
- `pytest` suite covering the BSP algorithm and ISA metrics (`tests/`), including a
  regression test for the atomic/end_to_end monolith failure mode found while
  investigating why one run stayed fragmented and another could collapse into one
  cluster. See "Testing" above.
- Live ISA metrics (RSF/NAIEF/LCOISF/CPSMF/DIIEF) shown after every BSP clustering
  iteration — initial pass and each refinement — with a colored delta vs. the previous
  iteration. Previously these were only computed once, after the fact, during the
  optional As-Is comparison step, so there was no quality signal available at the
  actual "are you satisfied with the clustering?" decision point.
- `requirements.txt` and this README (the project previously had neither).

### v0.2.0 — 2026-08-08
**Added**
- Process-type review step: before BSP runs, the LLM's `atomic`/`end_to_end` classification
  for every process is shown to the user for override, instead of being applied silently.
  Available in both the web app and the CLI.

**Changed**
- As-Is vs. To-Be gap analysis simplified: the LLM now compares As-Is and To-Be systems
  directly by name + process/entity footprint, instead of requiring a separate
  process/entity-level synonym-matching step with human confirmation.

**Removed**
- Synonym-matching subsystem (`match_synonyms`, `SynonymCandidate`/`SynonymMatchResult`
  schemas, `synonym_prompt.txt`, the synonym confirmation UI, and `unmatched_notes` from
  the gap-analysis result). ISA metrics computation for the As-Is matrix is unaffected —
  it still requires the full As-Is CRUD matrix, independent of this change.

### v0.1.0 — baseline
State as of commit `49b7766` ("as-is and to-be comparison") — first point this project
started being tracked with a changelog. Covers: LLM-driven CRUD matrix extraction, BSP
clustering with atomic/end-to-end handling, iterative architect-constraint refinement,
systems analysis with Build/Buy/Hybrid recommendations, EA principles compliance
evaluation, and an As-Is vs. To-Be comparison flow (via process/entity synonym matching,
later replaced in v0.2.0). See `git log` for the detailed commit history before this point.
