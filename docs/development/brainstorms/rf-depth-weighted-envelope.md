# RF depth-weighted envelope

**Status:** parked 2026-08-22, to investigate 2026-08-23.
**Branch:** `feature/depth-weighted-iff-attribution`.

## The direction

The `/N` estimator landed on this branch is **not** meant to be the receptive
field map on its own. It is meant to be a **receptive-field envelope**: a
normalised spatial profile that is afterwards **combined with the max receptive
field**.

That reframes what the numbers mean and dissolves the objection recorded below.
An envelope does not have to be a firing rate in Hz, and its absolute level does
not have to be alpha-invariant — it has to describe *shape*. Normalisation is
what makes the level irrelevant, and the max map is what restores the amplitude.

## What is already implemented

The `/N` change is complete and verified in `src/`:

- `data/vertex_accumulator.py` — new `count` array (contact points per vertex).
- `data/vertex_estimate.py` — `weighted_mean_or_nan` → `attributed_mean_or_nan`,
  divisor is `count`; `count_zero_credit` added; a grazing-only vertex is now
  `0.0` on the map instead of an excluded `0/0`.
- `pipelines/rf_single_touch_pipeline.py` — estimator repointed, `n_eff` guarded
  (`0/0` is now reachable and is written as an explicit NaN), summary JSON gains
  a `zero_credit_vertices` block and loses `no_estimate_vertices.zero_total_weight`.
- `gui/touch_playback_explorer.py` — same estimator, new "Zero credit" counter.
- `data/rf_population_heatmap.py`, `gui/rf_feature_space_explorer.py`,
  `gui/touch_population_explorer.py` — repointed from `weight_sum` to `count`
  (numerically identical there; all weights are 1.0).
- `data/touch_playback_data.py` — the duplicate-row docstring's "the scalar IFF
  factors out of numerator and denominator" argument was **true only for
  `/Σw`** and has been rewritten.

Tests: `test_vertex_estimate.py` rewritten, 22 pass. `test_vertex_accumulator.py`
passes unchanged (41).

## What is NOT done

55 expectations still encode the old `/Σw` formula and fail:
`test_vertex_weights.py` (31) and `test_rf_depth_weighting_parity.py` (24).
Every one inspected so far is an old-formula expectation, not a defect. They were
deliberately left failing rather than rewritten, because how they *should* read
depends on the envelope framing below.

## The finding that prompted the reframe

On the branch's own worked-example fixture — seven vertices, three frames, IFF
50/100/40 Hz, ground-truth receptive field at **v3** — the peak moves to v7:

| alpha | peak | v1 | v2 | v3 | v4 | v5 | v6 | v7 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | **v2** | 50.00 | 75.00 | 63.33 | 70.00 | 40.00 | 63.33 | 63.33 |
| 0.5 | **v7** | 27.39 | 59.28 | 53.56 | 51.62 | 21.91 | 22.24 | 61.13 |
| 1.0 | **v7** | 15.00 | 48.50 | 47.00 | 40.00 | 12.00 | 7.83 | 59.00 |
| 2.0 | **v7** | 4.50 | 36.05 | 39.61 | 28.00 | 3.60 | 0.98 | 54.97 |

Under `/Σw` the alpha=1 peak was v3 at 73.44 Hz. Alpha=0 is unchanged either way
(`Σw == N` when every weight is 1.0), so the parity baseline is intact.

**Mechanism.** v7 carries weights 0.92/0.93/0.95 — always deep under the finger,
never the RF centre. v3 carries 0.42/1.00/0.50 — full depth exactly in the frame
where the neuron fires hardest. Dividing by `Σw` cancelled each vertex's absolute
depth level, so only frame-to-frame variation in its own depth could move it.
Dividing by `N` stops that cancellation and lets absolute depth enter the value,
so "always deep" beats "deep at the right moment". The map becomes
`depth × firing rate` rather than `firing rate`.

Two consequences of the same mechanism:

- A vertex touched in exactly one frame is no longer alpha-invariant (v1: 50.0 Hz
  at every alpha under `/Σw`; 15.0 at alpha=1, 4.5 at alpha=2 under `/N`).
- Overall map level falls as alpha rises, since every weight is ≤ 1.

## Questions for tomorrow

1. **Normalised how?** Per touch, per session, or per neuron? Divide by the map's
   own max, by its sum, or by the deepest vertex's value? Each answer changes
   whether two touches are comparable.
2. **Combined with the max map how?** Product, or envelope-as-mask on the max, or
   something else? The max map is per-vertex `max(IFF)` and is deliberately
   *unweighted* — see `vertex_accumulator`'s note that a weighted maximum has no
   meaning.
3. **Does the v7 peak survive normalisation?** Normalising rescales the whole map
   by one constant; it cannot re-rank vertices. So if the envelope's *shape* is
   what matters, v7 > v3 is still the shape. Worth deciding whether that is
   acceptable for an envelope or whether the ranking itself needs fixing.
4. **Does alpha still mean what the docstrings say?** `vertex_weights` documents
   `alpha -> inf` as "all credit to the deepest vertex, the old
   `contact_depth = max()`". Under `/N` that limit sends every non-deepest weight
   to 0 *and* shrinks the surviving values, which is a different limit.
5. **What happens to the confidence channel?** `weight_sum` and Kish `n_eff` were
   built to describe the evidence behind a weighted mean. Their interpretation
   under `/N` — and under an envelope — has not been revisited.
6. **Where does normalisation live?** Adding it inside `vertex_estimate` would
   make that module know about map-level context it currently does not; a
   separate stage may be the cleaner boundary.

## Do not lose

- The two divisors coincide exactly at alpha=0. Any future formula should keep
  that, or the parity test stops meaning anything.
- Duplicate `(frame_index, vertex_id)` rows must stay separate under `/N`.
  Merging them into one row of weight `w1 + w2` was equivalent under `/Σw` and is
  **not** equivalent now — it would credit one sample where two landed.
