"""Accumulator sums -> the estimate, and the single definition of "no estimate".

``vertex_accumulator`` produces ``value_sum`` and ``count``; this module is the
one place that turns that pair into a per-vertex number, and the one place that
decides where **no number exists**.  It completes the trio that keeps the saved
maps and the on-screen heatmap the same numbers:

* :mod:`analysis.receptive_field_mapping.data.touch_frame_weights` — one depth
  -> weight conversion,
* :mod:`analysis.receptive_field_mapping.data.vertex_accumulator` — one
  reduction,
* this module — one estimator, and one answer to "which vertices have no
  estimate at all".

Before it existed the third step was spelled twice: ``_compute_touch_rf``
*raised* on a contacted vertex whose weights all came out zero, while
``gui.touch_playback_explorer.mean_heatmap_scalars`` quietly painted the same
vertex grey.  The pipeline therefore refused sessions the viewer had already
drawn without complaint.  A viewer disagreeing with the file is the failure mode
the shared-reduction work existed to prevent, so the predicate lives here and
both callers import it.

The estimator
-------------
::

    estimate[v] = value_sum[v] / count[v]
                = sum_{i: v_i = v} w_i * x_i  /  |{i: v_i = v}|

The divisor is the **number of contact points**, not the sum of their weights.
That choice is the whole point of the depth weighting and it is worth being
explicit about, because the two divisors answer different questions:

``/ sum(w)`` — a weighted **mean**
    The weights cancel out of the answer.  A vertex touched only at ``w = 0.1``
    still reports a full-size firing rate, because the small numerator is
    divided by an equally small denominator.  Depth decides *whose sample counts
    more*; it cannot change the level of the result.

``/ count`` — an **attributed** rate, which is what this module computes
    The weights survive into the answer.  Each contact point contributes
    ``w_i * x_i`` but still counts as one whole sample, so a vertex that was only
    ever brushed lightly reports a correspondingly small number.  Depth decides
    *how much of the firing this vertex is credited with*.

Worked example — three contact points on one vertex, IFF 10/20/30 Hz at weights
1.0/0.5/0.1.  The numerator is ``10 + 10 + 3 = 23`` either way.  A weighted mean
divides by ``1.6`` and returns ``14.375 Hz``; this estimator divides by ``3`` and
returns ``7.667 Hz``.

At ``depth_weight_alpha = 0`` every weight is exactly ``1.0``, so ``sum(w) ==
count`` and the two forms are the *same arithmetic on the same operands*.  The
alpha-zero parity baseline is therefore untouched by this choice, which is what
makes the parity test still meaningful.

``count`` counts contact points, not frames.  Two contact points of one frame
landing on the same vertex are two contributions to the numerator, so they are
two contributions to the denominator; see
:meth:`analysis.receptive_field_mapping.data.touch_playback_data` for why such
duplicates exist and are passed through rather than collapsed.

What "no estimate" means
------------------------
``NaN`` in the returned array means **no estimate**.  It never means "zero
response", and it must never be read as one.  One fact lands on it for a
contacted vertex:

``value_sum`` is NaN
    Every contact point contributing to that vertex carried a NaN neuron value —
    e.g. the unit was not held during the touch window.  There is a press but no
    neural measurement.

``count == 0`` also yields NaN, but for a *contacted* vertex it cannot happen:
being contacted is precisely having at least one contact point.  It is the
answer for the un-pressed remainder of the mesh, which callers normally exclude
before asking.

What is **no longer** a "no estimate" case
------------------------------------------
A vertex that was contacted but only ever **grazed** — every contact point at or
above the skin surface, so every weight is ``0``.  Under a weighted mean its
estimate was ``0/0``: undefined, dropped from the map, counted as an exclusion.
Under this estimator the divisor is the contact count, which is positive, so the
vertex lands on the map at exactly ``0.0``.

That is not a degradation, it is the formula's own answer, and it is a
*measurement*: the vertex was touched, it received zero depth credit, so zero
firing is attributed to it.  Reporting it as missing data would now be the
dishonest option — nothing is missing.

Those vertices are still counted, by :func:`count_zero_credit`, because a ``0.0``
that arises from zero credit is a different statement from a ``0.0`` that arises
from a silent neuron, and a reader of the map cannot tell them apart by looking.
The count is a diagnostic about vertices that are **on** the map, not an
exclusion tally.

Contract
--------
This module knows about **per-vertex sums, counts and weights** and nothing
else.  It must never learn about penetration depth, ``alpha``, millimetres,
``TouchEvent``, Qt or files.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NoEstimateCounts",
    "attributed_mean_or_nan",
    "has_estimate",
    "count_no_estimate",
    "count_zero_credit",
]


@dataclass(frozen=True)
class NoEstimateCounts:
    """How many entries have no estimate, split by cause.

    ``nan_value`` says there was no neural measurement while the vertex was
    pressed.  ``no_contact`` says there were no contact points at all, which is
    zero for any caller that has already restricted to the contacted set — it is
    kept separate rather than merged so that a caller passing an unrestricted
    array sees the two apart instead of reading the whole un-pressed mesh as
    missing data.

    Note what is *not* here: a grazing-only vertex.  It is no longer an
    exclusion — it has an estimate of ``0.0`` — and is counted by
    :func:`count_zero_credit` instead.
    """

    nan_value: int
    no_contact: int

    @property
    def total(self) -> int:
        """Entries excluded from the map for either reason."""
        return self.nan_value + self.no_contact


def _as_pair(value_sum, count):
    """Validate and return the two 1-D float arrays this module operates on."""
    values = np.asarray(value_sum, dtype=np.float64)
    counts = np.asarray(count, dtype=np.float64)
    if values.ndim != 1 or counts.ndim != 1:
        raise ValueError(
            f"vertex_estimate: value_sum and count must be 1-D, one entry "
            f"per vertex; got shapes {values.shape} and {counts.shape}."
        )
    if values.shape != counts.shape:
        raise ValueError(
            f"vertex_estimate: value_sum and count must cover the same "
            f"vertices in the same order; got {values.shape} and "
            f"{counts.shape}. A mismatch means the numerator and the "
            f"denominator have been sliced differently, which would divide one "
            f"vertex's total by another vertex's contact count."
        )
    if np.any(counts < 0.0):
        bad = np.flatnonzero(counts < 0.0)
        raise ValueError(
            f"vertex_estimate: count holds negative value(s) at index/indices "
            f"{bad.tolist()[:10]} (values {counts[bad].tolist()[:10]}). A contact "
            f"count is a tally of contact points and cannot be negative; a "
            f"negative one means a weight array has been passed where the "
            f"accumulator's ``count`` was expected."
        )
    return values, counts


def attributed_mean_or_nan(value_sum, count) -> np.ndarray:
    """Return ``value_sum / count``, or ``NaN`` where there is no estimate.

    The divisor is the **contact count**, never the weight sum: dividing by the
    weight sum would cancel the depth weighting back out of the answer and turn
    the result into a plain weighted mean, in which a barely-grazed vertex is
    indistinguishable from a fully-pressed one.  See the module docstring for the
    worked example.

    ``NaN`` in the result means *no estimate* — see the module docstring.  It is
    never a measured zero.  A grazing-only vertex is a measured zero and comes
    back as ``0.0``.

    Notes
    -----
    Where ``count > 0`` the returned value is exactly ``value_sum / count`` — the
    same IEEE division, on the same operands, that a bare ``value_sum / count``
    would produce.  The guard only substitutes the divisor where the quotient
    would be ``0/0``, so it cannot perturb any vertex that has an estimate.
    """
    values, counts = _as_pair(value_sum, count)
    contacted = counts > 0.0
    return np.where(
        contacted, values / np.where(contacted, counts, 1.0), np.nan
    )


def has_estimate(value_sum, count) -> np.ndarray:
    """Boolean mask: ``True`` where an estimate exists for that entry.

    This is *the* predicate.  The pipeline uses it to decide which vertices
    reach the saved ``.npz``; the playback viewer's ``NaN`` mask is its
    complement, so the vertices painted grey on screen and the vertices absent
    from the file are the same set by construction rather than by agreement.
    """
    return ~np.isnan(attributed_mean_or_nan(value_sum, count))


def count_no_estimate(value_sum, count) -> NoEstimateCounts:
    """Count entries with no estimate, split by cause.

    Parameters
    ----------
    value_sum, count:
        The accumulator totals **already restricted to the contacted entries**.
        Restriction is the caller's job because only the caller holds the
        contacted set; passing the whole mesh would report every un-pressed
        vertex as missing data.
    """
    values, counts = _as_pair(value_sum, count)
    no_contact = counts <= 0.0
    no_estimate = np.isnan(attributed_mean_or_nan(values, counts))
    return NoEstimateCounts(
        nan_value=int(np.count_nonzero(no_estimate & ~no_contact)),
        no_contact=int(np.count_nonzero(no_contact)),
    )


def count_zero_credit(count, weight_sum) -> int:
    """Count contacted entries whose total weight is ``0``.

    These vertices **are** on the map, at exactly ``0.0``: they were pressed, and
    every press that touched them was grazing, so the attributed rate is zero.
    The number exists because that ``0.0`` is indistinguishable, on screen and in
    the ``.npz``, from the ``0.0`` of a vertex that was pressed deeply while the
    neuron stayed silent — and the two are entirely different observations.

    Zero at ``depth_weight_alpha = 0``, where every weight is exactly ``1.0``.

    Parameters
    ----------
    count, weight_sum:
        Accumulator totals restricted to the contacted entries, as for
        :func:`count_no_estimate`.  ``count`` is required rather than inferred so
        that an un-pressed vertex (``count == 0``, ``weight_sum == 0``) is not
        counted as a grazing one.
    """
    counts = np.asarray(count, dtype=np.float64)
    weights = np.asarray(weight_sum, dtype=np.float64)
    if counts.shape != weights.shape:
        raise ValueError(
            f"count_zero_credit: count and weight_sum must cover the same "
            f"vertices in the same order; got {counts.shape} and "
            f"{weights.shape}."
        )
    return int(np.count_nonzero((counts > 0.0) & (weights <= 0.0)))
