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
   converts them into numeric affinity biases and re-runs BSP until you're satisfied. If a
   constraint conflicts with an atomic process's forced entity co-location, you're shown
   the conflict in plain language — no raw weight numbers — and asked to confirm before
   it's applied, one entity at a time. Atomicity is never silently overridden; a decline
   leaves the forced merge exactly as it was.
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

## Open questions for advisors

- **Should two entities ever be allowed to merge into one cluster purely from structural
  CRUD-matrix similarity (weighted Jaccard) when every process involved is `end_to_end`, or
  should a merge in that case always require an explicit architect `entity_weight`?**
  Context: in an all-`end_to_end` run, `Feedback Record` and `Customer Profile` merged into
  one cluster in the initial (no-constraint) BSP pass because they're touched by the exact
  same 5 processes everywhere (weighted Jaccard 0.636, threshold 0.5) — a strong, genuine
  data signal, not noise. This doesn't contradict the "end_to_end tends toward ~1 cluster
  per entity" rule of thumb (confirmed against a textbook matrix), since that's a tendency
  for *diffuse* end_to_end footprints, not a guarantee against a pair with perfect process
  overlap. Current behavior: kept as-is (data-driven default, architect can always override
  with an explicit `entity_weight`, which works correctly). Worth confirming with the
  thesis advisors which behavior is actually intended before treating this as settled.

## Changelog

### v0.4.24 — 2026-08-12
**Fixed**
- **`_greedy_reorder` (matrix row/column reordering) was non-deterministic
  across separate app runs.** Found live: re-testing the redesign's join
  constraint ("Menu and Inventory must be tightly integrated") sometimes
  worked and sometimes silently didn't, with identical matrix and weights.
  Root cause: `remaining = set(items) - {seed}`, and `max(remaining, ...)`
  breaks ties by iteration order — a Python `set` of strings iterates in
  an order that depends on per-process hash randomization, so the same
  matrix could reorder its columns differently every time the server
  restarted. Fixed by using a list instead of a set, so ties are always
  broken by original item order. Regression-tested
  (`tests/test_reordering.py`).
- **`_average_linkage_group` only ever compared a new item to the most
  recently opened group, never to earlier ones.** Even after the fix
  above made column order reproducible, the same real case (`Menu` /
  `Inventory Record`, joined by a `+0.5 entity_weight`) still failed to
  merge whenever an unrelated entity (`Supplier Profile`) landed between
  them in column order — `Inventory Record` was only ever compared to
  `Supplier Profile`'s group, never to `Menu`'s. Fixed by comparing each
  new item against every group formed so far and joining whichever has
  the best average (if it clears the threshold), instead of only the
  last group. The anti-chaining protection (comparing against a group's
  full average, not a single neighbour) is unchanged — only which groups
  get considered as candidates changed. Regression-tested
  (`test_entity_weights_still_merges_when_an_unrelated_entity_sits_between_them`).
  36/36 tests pass.

### v0.4.23 — 2026-08-12
**Changed**
- **Redesigned `bsp.py`'s cluster-assembly stage — a cluster is now defined
  directly by its entity group, and a process is a member of every cluster
  it genuinely writes to, with no competition involved.** Motivated by a
  known, twice-reproduced residual bug: `_pair_groups_to_clusters`'s
  CRUD-score tie-break (deciding which entity group a process group
  "wins") was blind to `process_weights`, and `_absorb_pure_readers`
  couldn't correct a bad placement when none of a reader's entities had a
  known creator. Removing the competition removes the tie-break itself,
  not just one axis's blind spot in it — confirmed against both known
  cases (`Financial Management`/`Supplier and Procurement Management`
  sharing a cluster; `Greeting/seating`/`Waitlist`/`Loyalty` gluing
  together), and against a fully end-to-end scenario shared by the thesis
  advisors, where the expected classical-BSP result is close to one
  cluster per entity.
  - Removed: `_pair_groups_to_clusters`, `_assign_unclaimed_entities`,
    `_effective_op_weight` (only existed to stop the competition from
    undoing an already-confirmed split — nothing left to undo), and
    `_group_processes` (only fed the now-removed competition).
  - Added: `_assign_process_membership` — builds one `Cluster` per entity
    group, then gives each process membership in every cluster it writes
    (C/U/D) to. Needs no atomic/end_to_end special-casing: `_enforce_atomicity`
    already guarantees an unexempted atomic process's writes land in a
    single entity group before this step even runs.
  - Renamed and rewritten: `_absorb_pure_readers` → `_place_pure_readers`.
    No longer *moves* a reader a competing step had already (possibly
    wrongly) seeded somewhere — there's no such step left. Instead it
    *places* a still-homeless reader into whichever cluster creates most
    of what it reads, and — new — leaves it unplaced on a genuine tie
    instead of guessing. `process_type` no longer matters here.
  - `_extract_blocks` renamed `_build_clusters`; no longer takes
    `process_weights` (had no remaining use once assembly stopped being a
    contest). `run_bsp`'s own signature and `BSPResult`'s fields are
    unchanged — `app.py` needed zero changes.
  - `process_weights`' role narrows to the visual matrix reordering and to
    `_detect_pending_conflicts`; `entity_weights` is now the axis with
    direct influence on cluster shape.
- **Bug found and fixed while verifying the redesign**: the owner lookup
  inside `_place_pure_readers` scanned every column of the matrix for each
  cluster's processes, instead of just that cluster's own entities. Once a
  single process could legitimately belong to several clusters (the whole
  point of this redesign), a process that creates entities in two
  different clusters could overwrite the correct owner of an entity in one
  cluster with the wrong cluster's index — found live via a reproduction
  script (a `CRUD` process spanning 3 entity clusters corrupted the owner
  map for 2 of them), not by inspection. Fixed by checking only `c.entities`
  per cluster. Regression-tested (`test_owner_lookup_is_not_corrupted_by_a_writer_in_several_clusters`).

**Tests**
- `tests/test_pure_reader_absorption.py` rewritten (only intentional
  exception to "all existing tests pass unmodified" for this change): its
  3 cases tested *moving* an already-(possibly-wrongly-)seeded reader, a
  scenario that can no longer occur. Rewritten to test *placement* instead,
  same 3 underlying concerns (clear creator → placed there; genuine
  ambiguity → left unplaced, replacing the old "end_to_end stays put"
  case since process_type no longer matters here; a writer is untouched),
  plus 1 new test for the owner-lookup bug above. 34/34 tests pass
  (33 pre-existing/adapted + 1 new).

### v0.4.22 — 2026-08-12
**Added**
- **16 new permanent regression tests, closing the top-priority gap from
  `TODO_next_session.md`.** Everything fixed live this session
  (`entity_weights`, the ownership fix, the `_assign_unclaimed_entities`
  veto, `process_span`-as-membership) was only ever verified via live app
  runs and one-off scratch scripts — real bugs, real fixes, but zero
  permanent protection against regressing them later. Now committed as
  `pytest`:
  - `tests/test_entity_weights.py` — `_group_entities` splitting/joining
    via `entity_weights`; `_detect_pending_conflicts` on a pure
    entity-weight signal (the self-owned-pair case process_weights alone
    can't represent), a pure process-weight signal, and both combined
    additively.
  - `tests/test_process_ownership.py` — ownership follows CREATE not
    UPDATE after a confirmed override; an orphaned entity survives as its
    own cluster instead of vanishing; `_assign_unclaimed_entities` vetoes
    a negatively-weighted reattachment (the `Financial Transaction`/
    `Supplier Profile` case).
  - `tests/test_process_span.py` — reads never count toward span, only
    writes; an `end_to_end` process becomes a genuine member of every
    cluster it writes to (the `[P1]->[A]`, `[P1,P2]->[B]` shape); an
    atomic process without a confirmed override never spans at all.
  - `tests/test_validate_extraction.py` — density (sparse/dense),
    entity-sharing, never-created, the dangling-reference-reported-once
    fix, and zero-operations.
  33/33 tests pass (17 pre-existing + 16 new).

### v0.4.21 — 2026-08-12
**Changed**
- **`Generator.extract()` now retries up to 3 times, not just once, and
  keeps the best attempt — not necessarily the last one.** Motivated by
  live confirmation, three separate times in one session (an atomic-only,
  an end_to_end-only, and a mixed-type run), that a single retry regularly
  left real `_validate_extraction` issues unresolved. Deliberately still
  bounded, not unbounded — nothing guarantees the model ever satisfies
  every check simultaneously, so it stops after `max_attempts` (default 3)
  and returns whichever attempt had the fewest remaining issues, tracked
  across all attempts rather than just trusting the final one (a later
  attempt can regress on something an earlier one got right).
- **Return shape changed**: `extract()` now returns `(result,
  remaining_issues)` instead of just `result`. `app.py`'s call site
  updated accordingly, and `remaining_issues` is folded into the existing
  `data_quality_warning` message/log key (`extraction_issues`) instead of
  a new, separate channel — closing the visibility gap flagged earlier:
  these issues used to only ever reach the server console, never the user.
  `last_extraction_validation_debug.json` (v0.4.18) now logs every
  attempt's issues, not just attempt 1 and the single retry.
  17/17 tests pass (generator.py has no pytest coverage; verified by
  compiling, importing `app.py` cleanly, and a full read-through of the
  loop logic — not yet run against a live extraction).

### v0.4.20 — 2026-08-12
**Changed**
- **`process_span` is no longer just metadata — it's now real cluster
  membership.** Closes the "known residual" flagged in v0.4.19. Until now,
  a process with a real write relationship outside its primary cluster was
  recorded in `process_span` but never actually added to that other
  cluster's `.processes` — so e.g. `Supplier Profile` ended up with
  `"processes": []` even though `Supplier and Procurement Management
  Process` genuinely creates it. `run_bsp` now walks `process_span` after
  computing it and adds the process to every cluster it spans, not just
  its primary one. This is the normal case for `end_to_end` processes
  (confirmed by the user against the classical BSP shape:
  `[P1] -> [Fornecedor]`, `[P1, P2] -> [Financeiro]`, same process P1 a
  real member of both) and only happens for an `atomic` process via an
  explicit `confirmed_override` — `_enforce_atomicity`'s hard rule is
  unchanged, an atomic process still can't leave its one cluster on its
  own. Verified both ways: the real end_to_end case from v0.4.19
  (`Supplier and Procurement Management Process` now a genuine member of
  both `Financial Transaction` and `Supplier Profile`), and a synthetic
  atomic-with-confirmed-override case producing exactly the
  `[P1]->[Fornecedor]` / `[P1,P2]->[Financeiro]` shape. 17/17 tests pass.

### v0.4.19 — 2026-08-12
**Fixed**
- **`_assign_unclaimed_entities` ignored `entity_weights` entirely, and
  could silently glue back together what a negative weight had just asked
  to be kept apart.** Found live via a three-way test (all-atomic,
  all-end_to_end, mixed process types): a user constraint separating
  `Financial Transaction` and `Supplier Profile` (both `process_weights`
  and `entity_weights` at `-0.5`, applied across two iterations) had no
  effect at all. Root cause: `Supplier and Procurement Management Process`
  creates both entities with identical raw priority, so
  `_pair_groups_to_clusters` ties between their entity groups and
  (arbitrarily, by iteration order) assigns the process to one — leaving
  the other group "unclaimed." `_assign_unclaimed_entities` then re-scores
  purely by raw CRUD weight to decide where the leftover group goes, with
  no awareness `entity_weights` exists — so the loser gets reattached
  right back to the winner's cluster, silently undoing the separation with
  no error and no PendingConflict (that mechanism only covers atomic
  processes via `_enforce_atomicity`; this path bypasses it entirely).
  Candidate clusters are now vetoed outright — never chosen regardless of
  CRUD score — if they'd introduce a negative `entity_weight` against an
  entity already inside. Verified against the real reproduction:
  `Financial Transaction` and `Supplier Profile` now land in separate
  clusters, and `process_span` correctly shows
  `Supplier and Procurement Management Process` accessing both — the same
  "owns one cluster, reaches into another" shape as a textbook BSP
  worked example the user provided. 17/17 tests pass.
- **Known residual, not yet fixed**: the winning side of the tie
  (`Supplier and Procurement Management Process` landing in the
  `Financial Transaction` cluster) is still decided by
  `_pair_groups_to_clusters`'s plain CRUD tie-break, which — like the bug
  above, but not yet fixed — doesn't check `process_weights` either. In
  this same real case, `Financial Management and Compliance Process` and
  `Supplier and Procurement Management Process` end up as cluster-mates
  despite a `-0.5` process_weight between them, and the new standalone
  `Supplier Profile` cluster ends up with `processes: []` — correct in
  that nothing is silently lost anymore, but not the cleaner shape the
  textbook example shows (every cluster owned by a real process). Left
  alone for now, scoped exactly to what was agreed; worth a follow-up
  decision on whether `_pair_groups_to_clusters`'s own tie-break should
  also become weight-aware.

### v0.4.18 — 2026-08-12
**Added**
- **`Generator.extract()` now writes `src/last_extraction_validation_debug.json`**
  — `_validate_extraction`'s findings across the extraction attempt(s)
  (`attempt_1_issues`, whether a retry happened, `remaining_issues` after
  the retry). Motivated directly by the live three-way test: the atomic
  run had three simultaneous `never_created`/dangling-reference violations
  that survived the retry, and the only trace of that was `print()`
  statements to the server console — invisible to anything that only reads
  files back afterward. Same directory and naming convention as the
  existing `last_extraction_debug.json`. Console prints are unchanged,
  kept for whoever's watching the terminal live. Verified the file writes
  correctly on both the clean-pass and retried-with-remaining-issues
  paths. 17/17 tests pass.

### v0.4.17 — 2026-08-12
**Fixed**
- **Two small `_validate_extraction` bugs found in the same review pass**
  as v0.4.16's prompt fixes:
  - `touched_entities` and `op_entity_names` were word-for-word identical
    set comprehensions over `operations`, computed twice under different
    names — consolidated into one, used by both the orphan check and the
    dangling-reference check.
  - The "never created" check wasn't scoped to properly-defined entities,
    so a hallucinated entity name referenced in an R/U/D operation (already
    caught, correctly, by the dangling-reference check) got a second,
    confusingly overlapping "needs a Create somewhere" issue for the same
    underlying problem. Now scoped to `entity_names ∩ touched_entities`, so
    a dangling reference is explained exactly once, by the check that
    names it correctly.
  - Verified both directions: a dangling/undefined entity reference is now
    reported once, not twice; a properly-defined entity with no Create
    (e.g. `Reservation`, the real case from v0.4.11) is still caught
    correctly. 17/17 tests pass.

### v0.4.16 — 2026-08-12
**Changed**
- **Three `prompt.txt` clarity/consistency fixes**, found during a review
  pass of the prompt and `_validate_extraction` together:
  - Merged the two "entity must connect to the matrix" rules, which used
    to live in separate, non-adjacent sections (ENTITIES said "must appear
    in an operation"; DENSITY REQUIREMENT separately said "must be
    created") — now stated together in ENTITIES as one rule and its
    refinement, instead of two disconnected callouts.
  - DENSITY REQUIREMENT now opens with an explicit, general statement that
    a single (process, entity) pair can carry more than one operation,
    using a clean example with no dependency on a name-pattern heuristic
    ("Manage Clients" → C, R, U, D all on "Client"). The existing
    "Purchase order creation and approvals" example stays as a more
    specific secondary case (the "two-step process name" signal).
  - Added a guard note after the worked example: entity/process names
    used in examples throughout the prompt (Purchase Order, Budget,
    Manage Clients, etc.) are patterns to follow, not literal names to
    copy — a hedge against the entity-naming-consistency question raised
    earlier, on the chance the concrete example names were themselves
    anchoring output.
  - Deliberately left alone: the RULE 1/RULE 2 process-classification
    keyword conflict found in the same review (RULE 1 lists "Management"
    as an `end_to_end` signal, but RULE 2's own example, "Waste logging
    and spoilage management," is atomic) — the user can always correct a
    process_type in the review step, so this was judged lower priority
    than the entity/density issues above.
  - Verified: prompt still formats cleanly with no stray template braces,
    17/17 tests pass (no code changes this round, prompt text only).

### v0.4.15 — 2026-08-11
**Fixed**
- **A combined-op cell like `"CU"` rendered completely empty in the
  matrix UI.** Root cause, confirmed from a screenshot: `index.html`'s
  `buildMatrixTable` checked `'CRUD'.includes(val)` to decide whether a
  cell holds a valid operation — that's a *substring* check, not "every
  character in val is C/R/U/D". `"CRUD".includes("CU")` is `false` (C is
  followed by R in the literal string "CRUD", not U), so the `if` guarding
  `td.textContent = val` never ran and the cell showed only its background
  colour, no letters. This bug existed the whole time but was invisible
  before v0.4.12, since `Matrix.add_entry` used to overwrite down to a
  single character — fixing that bug is what exposed this one. Same
  substring-check mistake also silently broke `isSpanAccess` detection and
  the `op-${val}` CSS class lookup (only `op-C`/`op-R`/`op-U`/`op-D` exist
  as real classes — `op-CU` matches none of them). Replaced all three with
  a proper `isCrudValue()` helper (`/^[CRUD]+$/.test(val)`), and added
  `dominantOpColor()` (same C > U > D > R priority `_OP_PRIORITY` already
  uses) so a combined cell's text colour reflects its highest-priority
  operation instead of silently falling back to white via
  `OP_COLORS[val]` (also keyed by single letters only).
- Also revised `prompt.txt`'s worked example per user feedback to
  demonstrate this exact case going forward (`C → Purchase Order` then
  `U → Purchase Order`, the approval step) — confirmed live: the fresh
  extraction now genuinely produces a `"CU"` cell, which is what surfaced
  this bug in the first place.

### v0.4.14 — 2026-08-11
**Fixed**
- **`prompt.txt`'s DENSITY REQUIREMENT example never showed a process
  doing more than one operation on its own entity.** Confirmed live: a
  fresh extraction had zero cells with combined operations anywhere —
  every process spread its 4 operations across 4 different entities,
  never revisiting one. The worked example (`"Purchase order creation and
  approvals"`) only ever showed `C → Purchase Order` then reads/updates on
  *other* entities, despite the process's own name implying two stages
  (create, then approve) on the *same* entity. Revised the example to show
  both `C → Purchase Order` and `U → Purchase Order` (approval = a status
  update to the same order), and added an explicit instruction: a process
  name with two steps joined by "and" is a signal the second step updates
  what the first step created, not a separate entity.
- **`process_span` counted reads as spans, which was noisy and not what
  it was for.** User feedback: reading a reference entity from another
  cluster is completely normal (every process does it) and isn't a
  meaningful "this process spans systems" signal — only a WRITE outside a
  process's own cluster represents real cross-system responsibility.
  `_compute_process_span` now only counts C/U/D (`_op_weight > R`, the
  same threshold `_enforce_atomicity` already uses to decide what counts
  as a "write"), not plain reads. Confirmed against the real scenario:
  `Purchase order creation and approvals`'s span dropped from `[1, 14,
  15]` to `[14, 15]` — the read-only `Inventory Record` cluster no longer
  counts, only the real `Budget` update does.
- **The blue border on `process_span` matrix cells didn't make sense on
  its own.** User feedback: colour it like the cluster it's reaching
  into, not an arbitrary accent colour. `index.html` now tints a span cell
  with that entity's own cluster colour (a lighter tint than a true
  member cell, since the process accesses it but doesn't belong there),
  and sets `dataset.cluster` on it so hovering that cluster's chip
  highlights the cell too, consistent with how member cells already work.
  17/17 tests pass.

### v0.4.13 — 2026-08-11
**Fixed**
- **`process_span` was computed correctly but never reached the UI.**
  v0.4.9 wired it into `log.json` only — the live `"clustering"` WebSocket
  messages that actually drive the matrix view never included it. Found
  live: `log.json` correctly showed `"Purchase order creation and
  approvals": [1, 14, 15]`, but the matrix in the browser only showed the
  process inside cluster 15 (what it owns/creates), with its update to
  `Budget` (cluster 14) rendered as an ordinary dimmed cross-cluster touch
  — visually indistinguishable from a stray/unintentional reference. Added
  `process_span` to both `"clustering"` sends in `app.py` (initial and
  per-iteration), and wired it through `index.html`
  (`appendClustering` → `buildClusteredMatrix` → `buildMatrixTable`): a
  cell outside a process's own cluster block that IS on its recorded
  `process_span` now gets a blue outline/tint and a tooltip explaining
  it's a legitimate cross-system access, instead of just being dimmed
  like ordinary noise. `process_span` cluster IDs are translated through
  an id→array-index map rather than assumed to line up, since the two use
  different numbering. 17/17 tests pass (backend-only changes are
  covered; the rendering change has no Python test surface).

### v0.4.12 — 2026-08-11
**Fixed**
- **`Matrix.add_entry` overwrote instead of combining operations on the
  same (process, entity) pair.** Flagged as a known, unfixed bug earlier
  this session (`TODO_next_session.md`); user re-raised it independently
  ("a process like 'Manage Clients' should get C,R,U,D on Client") without
  knowing it was already on the list. The LLM emits one entry per
  operation type, so a process with full CRUD on one entity arrives as
  separate C/R/U/D records — `add_entry` kept overwriting the cell with
  whichever came last (`self.matrix[p][e] = operation.upper()`), so
  `"Recipe development and testing"` → `"Recipe"` (C, then R, then U)
  ended up as the single value `"U"`, silently losing the `C`. This wasn't
  just cosmetic: `_compute_entity_owners` and `_absorb_pure_readers` both
  read these cells through `_op_weight`, which already expects and
  correctly handles combined strings like `"CRU"` — so the higher-priority
  operations were being lost before `_op_weight` ever got a chance to see
  them, silently corrupting entity-ownership resolution and pure-reader
  classification for any process with more than one operation type on the
  same entity. Now appends instead of overwriting (`"C"` + `"R"` + `"U"` →
  `"CRU"`, no duplicate letters). Added `tests/test_matrix.py` — no
  coverage existed for `Matrix` at all before this. 17/17 tests pass.

### v0.4.11 — 2026-08-11
**Added**
- **`_validate_extraction` now also requires every entity to be CREATED by
  at least one process.** User-proposed rule, grounded in the same CRUD-
  matrix-anomaly literature already cited for hub detection (Bureš, Cerny,
  Frajtak & Ahmed, 2019): an entity with only R/U/D operations and no C
  anywhere has no origin in the process landscape — either the entity is
  wrong, or the process that actually produces it is missing its Create.
  Confirmed against the same real degenerate extraction from v0.4.10:
  `Reservation` is read/updated by two processes but created by neither —
  now correctly flagged. Also added equivalent guidance to `prompt.txt`'s
  DENSITY REQUIREMENT section so the LLM self-checks this before
  finalising, not just on retry. All 14 tests pass unchanged.

### v0.4.10 — 2026-08-11
**Added**
- **`_validate_extraction` (`generator.py`) now catches degenerate CRUD
  matrices, not just structural ones.** Found live: an extraction produced
  a matrix where almost every process had exactly 3 CRUD entries on a
  single private entity (below the stated 4-6 density floor), with hardly
  any entity genuinely shared across processes — even though real context
  (`enunciado.pdf`) was attached, ruling out the "no context" fallback
  path as the cause. This shape is structurally valid (entity count in
  range, no orphans, no dangling references, no zero-op processes) so
  the existing checks passed it through silently, with no retry, even
  though it violates two things `prompt.txt` explicitly asks for. Two new
  checks reuse the same existing retry-with-reminder mechanism:
  - Per-process density: flags processes with <4 or >6 CRUD entries
    (`prompt.txt`'s stated 4-6 range).
  - Entity sharing: flags the whole extraction when more than half of the
    entities are touched by only one process — entities are supposed to
    be shared data objects, and BSP clustering has no signal to work with
    otherwise.
  Verified against the real degenerate extraction that prompted this: both
  checks fire (plus a pre-existing, previously-unnoticed orphan-entity
  issue on the same file), and would now trigger a retry instead of
  silently returning. All 14 tests pass unchanged (`generator.py` isn't
  covered by the existing suite, but nothing else imports or calls this
  function differently).
- **Still open**: whether this specific case was the LLM under-using a
  real document, or `fitz`'s PDF text extraction returning too little
  usable text for this particular PDF (e.g. a scanned/image-based page
  with no real text layer) — `app.py`'s existing "Extracting CRUD matrix —
  N processes, X chars of context…" status message is the place to check
  next time this happens, before assuming it's the LLM's fault.

### v0.4.9 — 2026-08-11
**Fixed**
- **A confirmed override moved a process's ownership to the wrong cluster.**
  Found live: after confirming `Purchase Order` should split away from
  `Budget` (both written by the atomic `Purchase order creation and
  approvals`), that process ended up owning the *Budget* cluster, not the
  *Purchase Order* cluster it actually creates. Root cause: `_pair_groups_
  to_clusters` used `_effective_op_weight`, which zeroes out a confirmed-
  exempt (process, entity) pair — correct for stopping a split-away entity
  from being silently reabsorbed (`_assign_unclaimed_entities`, unchanged),
  but wrong here: zeroing the process's strongest real relationship (what
  it CREATES) made it gravitate to whatever it merely updates instead. Now
  uses raw `_op_weight`, so ownership always follows the true CRUD
  relationship regardless of override bookkeeping.
- **Generalized `_compute_e2e_span` → `_compute_process_span`
  (`BSPResult.e2e_span` → `process_span`).** Previously only `end_to_end`
  processes were tracked as spanning multiple clusters. A process with a
  confirmed override is in the exact same situation — it owns the cluster
  holding what it creates, but still needs access to the cluster holding
  what it only updates — so it's now tracked the same way. Wired into
  `log.json` (`initial_bsp`, each `bsp_iterations` entry, and `final_bsp`)
  so this is visible without re-deriving it by hand.
- All 14 tests pass unchanged; verified against a reproduction of the
  live scenario (`Purchase order creation and approvals` / `Purchase
  Order` / `Budget`) — the process now owns the `Purchase Order` cluster
  and `process_span` correctly lists both clusters it touches.

### v0.4.8 — 2026-08-11
**Added**
- **`log.json` now keeps a full audit trail of atomicity-override decisions
  (`atomicity_overrides.resolution_history`).** Found live: once a
  `PendingConflict` is confirmed or declined, it used to collapse into a
  bare `{process: [entity, ...]}` set — `conflicting_process` and
  `reasoning` were only ever logged for conflicts still unresolved at the
  end (`still_pending`), so nothing explained *why* a resolved override
  happened after the fact. `app.py` now records `{iteration, process,
  entity, conflicting_process, reasoning, decision}` for every override at
  the moment it's confirmed/declined, and both `bsp_iterations` and
  `final_bsp` expose the cumulative list. Purely additive to the log
  shape — `confirmed`/`declined` are unchanged; all 14 tests still pass.

### v0.4.7 — 2026-08-11
**Added**
- **`entity_weights` is now wired end-to-end: the LLM can actually produce it.**
  v0.4.6 built the `bsp.py` plumbing (entity-axis grouping, conflict
  detection) but nothing produced real `entity_weights` yet — every real
  run still only used `process_weights`. This phase closes that gap:
  - `bsp.py`: `EAWeightResult` replaced by `ConstraintWeightsResult`
    (`process_biases: List[ProcessWeight]` + `entity_biases: List[EntityWeight]`,
    the schema flagged back in v0.4.0-v0.4.3 as "replaced in a later phase").
    New `EntityWeight` schema mirrors `ProcessWeight` but for entity pairs.
  - `bsp_prompt.txt`: added an `EXACT ENTITY NAMES` block (mirroring
    `EXACT PROCESS NAMES`), a new explanation of when to emit an
    `entity_bias` instead of/alongside a `process_bias` (same process
    creating entities on both sides of a separation; hub entities), and an
    "ENTITY AXIS" sub-step added to TYPE C and TYPE D (the separation/
    isolation types — where the real hub-entity bug lives) with worked
    examples. TYPE A/B and the overall structure are untouched, since
    they're already confirmed working from live testing.
  - `generator.py`: `_compute_ea_weights` (name kept, per earlier decision)
    now returns `(process_weights, process_reasoning, entity_weights,
    entity_reasoning)` — four values instead of two. Extracted the
    duplicated name-validation/resolution logic into `_resolve_biases`,
    shared by both axes.
  - `app.py`: new `acc_entity_weights`/`acc_entity_reasoning` accumulators,
    threaded into both `run_bsp(...)` calls and both `log.json` blocks
    (`bsp_iterations` and `final_bsp`) alongside `process_weights`. The
    weight-reasoning text handed to `describe_systems`/
    `evaluate_ea_compliance` now includes entity-axis lines too.
- Fixed a real display bug this made reachable: when a `PendingConflict`'s
  entity is self-owned (the same atomic process creates both the anchor
  and the conflicting entity — exactly the case `entity_weights` exists to
  catch), `conflicting_process` used to equal the process itself, so the
  UI read "Currently merged with P's system" inside P's own card.
  `_detect_pending_conflicts` now leaves `conflicting_process` empty for
  that case, and `index.html`'s override card shows "forced together with
  *entity* (created by this same process)" instead.
- Verified with a network-free hand-test script (schema shape, prompt
  placeholder formatting, `_resolve_biases` on both axes including a
  hallucinated-name skip) — no live LLM call was made. All 14 existing
  tests still pass unchanged. **Not yet tested against the live app with a
  real constraint** — that's the natural next step before trusting this
  in practice.

### v0.4.6 — 2026-08-11
**Added**
- **`entity_weights`: a second, independent axis of architect bias, on
  pairs of ENTITIES rather than processes.** Motivated by a real bug found
  through live testing (v0.4.4's "known open risk"): a hub entity (e.g.
  `Financial Transaction`) touched by several atomic processes could
  transitively glue unrelated clusters together, and `process_weights`
  had no way to express "these two entities specifically must not be
  together" — only "these two processes must not be together," which
  breaks down when the processes creating the entities aren't the ones
  a constraint is really about, or when a *single* atomic process creates
  both entities (no second process exists to hold a weight against).
  `_group_entities` now takes `entity_weights` directly (mirroring how
  `_group_processes` already took `process_weights`), and
  `_detect_pending_conflicts` sums a direct entity-pair bias together with
  the existing process-pair bias — either axis alone (or both together)
  can trigger a conflict. Replaced the old `_biased_similarity` one-off
  closure (which faked an entity-level check by proxying through
  `process_weights` and `entity_owners`) with a proper reusable function,
  `_entity_similarity_with_bias`.
- Atomicity itself still cannot be silently broken by either axis — same
  as before, a conflict only ever surfaces as `BSPResult.pending_conflicts`
  for the architect to confirm, never auto-applied.
- Pure plumbing: no prompt/schema/LLM changes yet, so no real caller
  produces `entity_weights` today — `app.py` still only passes
  `process_weights`. Hand-tested with synthetic `entity_weights` dicts
  (see the two cases above); all 14 existing tests pass unchanged, since
  every new parameter defaults to `None`/`{}`.

### v0.4.5 — 2026-08-11
**Changed**
- **Renamed `ea_weights` → `process_weights` (and `ea_reasoning` →
  `process_reasoning`) throughout `bsp.py` and `app.py`, first step of a
  broader two-axis redesign.** The architect's weights have always lived
  on pairs of *processes*, but the real gap this session's live testing
  kept exposing (see v0.4.4's "known open risk") is that a constraint
  often needs to separate *entities*, not just the processes that touch
  them. Before adding a new `entity_weights` axis on top, the existing
  one needed a name that says what it actually is — `process_weights` —
  so the two don't read as the same thing. Pure rename, no behaviour
  change: `EAWeight` → `ProcessWeight`, `app.py`'s `acc_weights`/
  `acc_reasoning` → `acc_process_weights`/`acc_process_reasoning`,
  `log.json`'s `"ea_weights"` key → `"process_weights"`.
  `_compute_ea_weights` (`generator.py`) keeps its name — "EA" is
  Enterprise Architecture, which stays valid regardless of axis.
  Verified against all 14 existing tests (none reference the old names
  directly) — 14/14 pass unchanged.

### v0.4.4 — 2026-08-11
**Fixed**
- **`bsp_prompt.txt` had no category for one-sided isolation constraints,
  so it produced zero weights for them.** TYPE C (added in v0.4.1) only
  handles two-sided separation ("X and Y should be apart"), which
  requires identifying two named domains. A constraint like "Waitlist,
  seating plan, pacing controls should be in a separate isolated system"
  names only one side and asks for separation from *everything else* —
  it doesn't fit TYPE A, B, or C's shape, so the LLM produced an empty
  `ea_weights` result rather than forcing it into the wrong pattern.
  Found live: the constraint had no effect at all, and `ea_weights` came
  back completely empty for it.
- Added **TYPE D — Isolation constraints** to Step 1, mirroring TYPE C's
  structure but for the one-sided case: identify the named process(es),
  then treat every *other* process in `EXACT PROCESS NAMES` as the
  implicit "everything else" side, assigning strong negative weights
  (-0.4 to -0.5) across all of those pairs — explicitly instructed not to
  skip processes that seem name-unrelated, since the whole point is
  catching structural connections a name-only reading would miss.

**Known open risk, not yet confirmed either way:** TYPE D fixes weight
*generation*, but `ea_weights` still only influence `_group_processes`
(v0.4.0) — the later cluster-assembly steps
(`_pair_groups_to_clusters`/`_assign_unclaimed_entities`, both fixed in
v0.4.2) only respect `confirmed_overrides`, not raw `ea_weights` directly.
So even with correct, strong negative weights now generated, an isolated
process could still end up pulled back into the same cluster at the final
assembly step if its own entities are only reachable through processes
the weights don't cover — the same underlying gap the entity/process
representation discussion (still unresolved) is about. Worth retesting
this exact constraint before assuming it's fully fixed.

### v0.4.3 — 2026-08-11
**Added**
- **The atomicity-override mechanism (`pending_conflicts`/`confirmed_overrides`,
  built into `bsp.py` in v0.4.0-v0.4.2) is now wired into `app.py` and the
  frontend.** After each constraint iteration, if any of the architect's
  weights conflict with an atomic process's forced write co-location, the
  UI now shows each conflict in plain language — which process, which
  entity, which other process it's currently stuck with, and why — with a
  per-entity choice between "split apart" and "keep merged." Nothing is
  ever applied automatically; a decline is remembered so it isn't asked
  again every subsequent iteration.
- New WebSocket message type `atomicity_override_confirm` (server → client),
  sent from `pipeline()` in `app.py` right after each iteration's
  `_compute_ea_weights` call whenever unresolved conflicts exist. Reuses
  the existing untyped `answer`/`value` protocol, same as
  `process_type_review`, with a response shape of
  `{process: {entity: true|false}}`.
- New frontend builder `appendAtomicityOverrideConfirm` in
  `src/static/index.html`, modeled directly on `appendProcessTypeReview` —
  one group per conflicting process, one row per conflicting entity within
  it, radio choice per row.
- `log.json`'s `bsp_iterations` and `final_bsp` both gained an
  `atomicity_overrides` key (`confirmed`/`declined`/`still_pending`) for
  auditing which overrides were applied, declined, or never resolved.
- Not yet tested end-to-end through an actual browser session — verified
  the underlying `bsp.py` mechanism extensively against real data (see
  v0.4.0-v0.4.2), verified `app.py` compiles and the decision-application
  logic is correct by careful re-reading, and confirmed all 14 existing
  tests still pass (this pass touches no `bsp.py` code). The live
  WebSocket round-trip — does the UI render correctly, does confirming
  actually change the next clustering iteration — still needs a real
  run to confirm.

### v0.4.2 — 2026-08-11
**Fixed**
- **`_pair_groups_to_clusters` and `_assign_unclaimed_entities` ignored
  `confirmed_overrides`, silently re-merging entities the union-find had
  just correctly kept apart.** Found via a real session: even after
  `_enforce_atomicity` correctly separated two entities that a single
  atomic process creates (both confirmed exempt), the final cluster-
  assembly step recomputed a raw CRUD-weight score from scratch — with no
  memory that a deliberate separation had just been made — and reattached
  the orphaned entity to the same cluster anyway. Root cause: a tie in
  `_pair_groups_to_clusters` (a process touching two now-separated
  entities equally) got broken by iteration order, and the leftover
  entity was then reattached by `_assign_unclaimed_entities` purely
  because the same process still structurally connected to it.
- Added `_effective_op_weight()`, used by both functions in place of
  `_op_weight()` directly: identical, except it returns 0 for any
  (process, entity) pair already listed in `confirmed_overrides`, so a
  confirmed separation can no longer be silently undone by either the
  initial pairing or the leftover-entity reattachment step.
- `_assign_unclaimed_entities` also gained a fallback: previously, an
  entity-group that scored 0 against every existing cluster was silently
  dropped from the output. Now it becomes its own new cluster instead —
  needed because the fix above can legitimately produce a 0-score
  situation (the only real connection was the one just confirmed exempt).
- Verified against the real failing case end-to-end: `Order` and
  `Service Log`, both created by the single atomic process `Line
  execution during service`, now correctly land in different clusters
  once confirmed — `Service Log` moves to the cluster that actually
  matches its architectural intent (`Post-service reconciliation`/`Waste
  logging and spoilage management`) instead of staying stuck with `Order`.

### v0.4.1 — 2026-08-11
**Fixed**
- **`bsp_prompt.txt` had no instructions for separation constraints, so it
  silently inverted their sign.** Step 1 (user constraints) only defined
  two types — TYPE A and TYPE B — and both are exclusively about
  *co-locating* processes (positive weights only). A constraint asking for
  the opposite ("Orders and Inventory Records should be handled by
  separate systems") had no matching category, so the LLM forced it
  through the closest available pattern and applied the *only* weight
  instruction it had: a strong **positive** weight — pulling the targeted
  processes together instead of apart, exactly contradicting both the
  user's intent and the LLM's own generated reasoning text (observed live:
  weight `+0.5` on a pair whose reasoning said the interaction "should be
  minimized"). Found via a real session — `Order taking, pairing guidance,
  course pacing` and `Stocking and storage allocation` ended up merged
  into one cluster despite an explicit separation constraint targeting
  exactly that pair.
- Added **TYPE C — Separation constraints** to Step 1, mirroring TYPE B's
  structure but for "separate systems"/"must not share"/"isolated from"
  language, with explicit negative-weight instructions (-0.4 to -0.5) and
  a worked example using the real failing case above.
- Added an explicit sign-convention rule to the RULES section as a safety
  net for constraint phrasings that don't cleanly match TYPE A/B/C:
  separation language always means negative, co-location language always
  means positive, and the numeric sign must always agree with the
  reasoning text generated alongside it.
- Note: this fix addresses the weight's *sign* being wrong. It does not
  by itself let a corrected negative weight override an atomic process's
  forced union — that's the separate, already-tracked gap in
  `TODO_next_session.md` (`confirmed_overrides`/`pending_conflicts` still
  need wiring into `app.py`/the frontend). Both bugs were masking each
  other in the same test session; this one is fixed, the other isn't yet.
- **`src/utils/initial_matrices/`-adjacent tooling**: `app.py` now saves
  every extracted matrix (`mat.matrix` + `mat.process_types`) to
  `src/resources/initial_matrices/` as its own timestamped file, so real
  matrices accumulate across sessions instead of only the single
  overwritten `last_extraction_debug.json`. Directly enabled diagnosing
  this bug — both the sign-inversion and a separate, still-open
  observation that two extractions of the same process list produced
  meaningfully different entity sets (only 11 of ~15-19 entities shared
  between two runs), which needs follow-up to determine whether it's
  expected LLM variability or something else.

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
