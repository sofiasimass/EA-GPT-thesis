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
reference — it is not the way this project is run.

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
git tag -a v0.3.0 -m "short summary"      # tag the commit
git push && git push --tags               # if/when pushing to GitHub
```

Then add an entry to the Changelog below in the same commit or the next one. The tag lets
you check out or reference the exact code state for any version later (useful for pointing
a thesis chapter at "the version evaluated in Section 4.2"); the Changelog is the
human-readable summary of what changed and why.

## Changelog

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
