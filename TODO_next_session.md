# Next session — pick up here

Context: `src/utils/bsp.py` was restructured into an explicit pipeline (see
README v0.4.0-v0.4.3 for the full history — including two real bugs found
and fixed via live testing: an inverted weight sign in `bsp_prompt.txt`,
and `_pair_groups_to_clusters`/`_assign_unclaimed_entities` silently
re-merging entities the union-find had already correctly separated). All
14 existing tests pass throughout.

## -1. FIRST: test entity_weights against the live app (new, not yet done)
v0.4.6-v0.4.7 added a second, independent bias axis — `entity_weights`, on
pairs of ENTITIES rather than processes — motivated by a real bug: a hub
entity (or a single atomic process creating two entities that should be
apart) that `process_weights` alone can't represent. Built and verified
with network-free hand-tests (`bsp.py` plumbing, schema shape, prompt
formatting, name-resolution logic) but **never run against the live LLM or
a real constraint.**
- [ ] Run the app, write a separation/isolation constraint (TYPE C or D)
      where the "hub entity" or "same-process-creates-both" pattern is
      real (e.g. re-try the `Waitlist, seating plan, pacing controls`
      constraint from earlier this session, or the `Financial Transaction`
      hub scenario) — confirm the LLM actually emits `entity_biases` this
      time, not just `process_biases`.
- [ ] Check `log.json`'s new `"entity_weights"` key (in both
      `bsp_iterations` and `final_bsp`) has real content, not `{}`.
- [ ] Confirm a `PendingConflict` driven purely by `entity_weights` (no
      `process_weights` involved) surfaces correctly in the
      `atomicity_override_confirm` card, and that the new "forced together
      with X (created by this same process)" text (for self-owned pairs)
      reads sensibly rather than the old "merged with P's system" text.
- [ ] Sanity-check the LLM isn't over-emitting `entity_biases` for
      constraints that were already well-served by `process_biases` alone
      — the prompt says "when in doubt, emit both," which is safe but
      could be noisy; decide if that guidance needs tightening.

## 0. Test the atomicity-override UI end-to-end (not yet done)
The mechanism (`confirmed_overrides`/`pending_conflicts`) is now wired into
`app.py` and `src/static/index.html` (v0.4.3) but has **never been run
through an actual browser session** — everything so far was verified via
scripts, not the live WebSocket flow.
- [ ] Run the app, reach a constraint that should trigger a conflict (e.g.
      a separation constraint touching an atomic process that creates two
      entities directly, like the real `Line execution during service` /
      `Order` / `Service Log` case from this session).
- [ ] Confirm the `atomicity_override_confirm` card actually renders
      correctly, grouped by process with one row per conflicting entity.
- [ ] Confirm clicking "Split apart" vs "Keep merged" and hitting confirm
      actually changes the next clustering iteration the way it should.
- [ ] Confirm a decline is genuinely not asked again in a later iteration.
- [ ] Check `log.json`'s new `atomicity_overrides` key (`confirmed`/
      `declined`/`still_pending`) reflects what you actually did.

## 1. Test the adaptive threshold
- [ ] Run `run_bsp(matrix, adaptive_threshold=True, adaptive_k=1.0)` against
      a few real/realistic matrices (not just the tiny 4-process order
      example already checked) — ideally something closer to real size
      (10-20 entities, per the current density rule). Real matrices now
      accumulate in `src/resources/initial_matrices/` every time you run
      the app.
- [ ] Compare cluster output against the same matrix with
      `adaptive_threshold=False` (the fixed `0.5` default) — same clusters?
      More fragmented? Less?
- [ ] Try a few different `adaptive_k` values (0.5, 1.0, 1.5, 2.0) on the
      same matrix and see how sensitive the result is.
- [ ] Decide: is `k=1.0` reasonable, does it need to change, and is
      `adaptive_threshold` worth making the default eventually?
- [ ] Also sanity-check `adaptive_min_pairs` (default 5) — does it correctly
      fall back to the fixed threshold on small matrices?

## 2. Decide on the 4-6 operations-per-process density rule
- [ ] Re-read `src/resources/prompt.txt`'s "DENSITY REQUIREMENT" section
      and `src/utils/matrix.py`'s `MatrixResult` docstring — both currently
      force every process into 4-6 CRUD entries (padding sparse processes
      with fabricated reads, truncating dense ones).
- [ ] With the adaptive threshold in hand, check whether a matrix with more
      natural (unpadded) density variation still clusters sensibly — if so,
      the density rule may no longer be needed to make the algorithm work.
- [ ] Decide: relax the floor, keep it, or replace the ceiling with the
      hub-detection mechanism (`BSPResult.hub_flags`, already built,
      currently detection-only) instead of "drop the weakest reads."

## 3. Remaining follow-ups (not urgent)
- [ ] Wire `classify_changes()` into the iteration loop so the frontend can
      show "what changed and why" between iterations, with atomicity
      overrides visually distinguished from ordinary changes. Not done yet
      — the function exists and is exported but nothing calls it.
- [ ] Add regression tests for the new opt-in machinery
      (`derive_threshold`, `decide`, `_enforce_atomicity`/
      `_pair_groups_to_clusters`/`_assign_unclaimed_entities` with
      `confirmed_overrides`, `_detect_pending_conflicts`) — everything so
      far was verified with real data via ad-hoc scripts, not committed
      as `pytest` regression tests. Worth doing now that the mechanism is
      live in the app.
- [x] `Matrix.add_entry` overwrote instead of combining multiple operation
      records for the same (process, entity) pair — fixed in v0.4.12, with
      regression tests in `tests/test_matrix.py`.
- [ ] The matrix-variability question from earlier this session (two
      extractions of the same process list sharing only ~11 of ~15-19
      entities) is still unresolved — never confirmed whether the input
      was identical between those two runs.

## Reference
- Plan file from the restructure: `C:\Users\simas\.claude\plans\ok-then-i-allow-fizzy-lollipop.md`
- README changelog (v0.4.0 through v0.4.3) has the full history, with citations.
