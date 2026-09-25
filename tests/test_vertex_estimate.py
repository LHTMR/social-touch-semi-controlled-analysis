"""The shared estimator: what the divisor is, and where no estimate exists.

``vertex_estimate`` is the third and last of the shared steps between the saved
maps and the on-screen heatmap (after ``touch_frame_weights`` and
``vertex_accumulator``).  It exists because the two sides disagreed: the pipeline
*raised* on a contacted vertex whose weights all came out zero, while the
playback viewer quietly painted the same vertex grey — so the viewer rendered
sessions the pipeline refused to process.

These tests pin two things.  First the **divisor**: the per-vertex total is
divided by the contact count ``N``, not by ``sum(w)``, which is what lets the
depth weighting reach the value instead of cancelling out of it.  Second the
policy for where no estimate exists at all.

The end-to-end agreement between the two callers is pinned separately, in
``tests/test_rf_depth_weighting_parity.py::TestPlaybackViewerAgreesWithTheSavedMap``.
"""

import numpy as np
import pytest

from analysis.receptive_field_mapping.data.vertex_estimate import (
    NoEstimateCounts,
    attributed_mean_or_nan,
    count_no_estimate,
    count_zero_credit,
    has_estimate,
)


class TestTheDivisorIsTheContactCount:
    """The whole point of the change: weights survive into the answer."""

    def test_the_worked_example(self):
        """Three contact points, IFF 10/20/30 Hz, weights 1.0/0.5/0.1.

        Numerator ``10*1.0 + 20*0.5 + 30*0.1 == 23`` either way.  A weighted mean
        divides by ``1.6`` and returns ``14.375``; this estimator divides by ``3``
        and returns ``23/3``.  The two differ by more than a factor of 1.8 on the
        same data, so nothing about this can be a rounding detail.
        """
        got = attributed_mean_or_nan(np.array([23.0]), np.array([3.0]))
        assert got[0] == pytest.approx(23.0 / 3.0, rel=1e-15)
        # And it is emphatically not the weighted mean.
        assert got[0] != pytest.approx(23.0 / 1.6, rel=1e-6)

    def test_shallow_contacts_pull_the_estimate_down(self):
        """A lightly-pressed vertex reports less, which a weighted mean cannot do.

        Two vertices see the *same* firing rate at every contact point; one was
        pressed fully (weights 1.0) and one only brushed (weights 0.25).  Under a
        weighted mean both report 40 Hz, because the weights cancel.  Here the
        brushed vertex reports a quarter of it — that difference is the feature.
        """
        deep_value_sum = 40.0 * 1.0 * 2       # two contact points at weight 1.0
        light_value_sum = 40.0 * 0.25 * 2     # two contact points at weight 0.25
        got = attributed_mean_or_nan(
            np.array([deep_value_sum, light_value_sum]), np.array([2.0, 2.0])
        )
        assert got[0] == pytest.approx(40.0)
        assert got[1] == pytest.approx(10.0)
        # The weighted mean these replace would have made them identical.
        weighted = np.array([deep_value_sum, light_value_sum]) / np.array([2.0, 0.5])
        assert weighted[0] == pytest.approx(weighted[1])

    def test_it_agrees_with_the_weighted_mean_when_every_weight_is_one(self):
        """The alpha = 0 baseline, at the level of this module.

        At ``depth_weight_alpha = 0`` every weight is exactly ``1.0``, so
        ``sum(w) == N`` and the two divisors are the *same operand*.  This is why
        the divisor choice cannot move an alpha = 0 map, and why the end-to-end
        parity test still means something.
        """
        value_sum = np.array([3.0, 100.0, -7.5])
        n = np.array([7.0, 3.0, 2.0])
        weight_sum_at_alpha_zero = n  # every weight 1.0 -> sum(w) == N
        assert (
            attributed_mean_or_nan(value_sum, n).tolist()
            == (value_sum / weight_sum_at_alpha_zero).tolist()
        )

    def test_the_quotient_is_the_plain_division_where_a_contact_exists(self):
        """No guard may perturb a vertex that has an estimate.

        The byte-identity claim rests on this: the substituted divisor applies
        only where the quotient would be 0/0, so every contacted vertex gets
        exactly the IEEE result of ``value_sum / count``.
        """
        values = np.array([3.0, 100.0, -7.5, 1e-18])
        counts = np.array([7.0, 3.0, 4.0, 1e18])
        got = attributed_mean_or_nan(values, counts)
        want = values / counts
        assert got.tolist() == want.tolist()


class TestGrazingIsAMeasuredZero:
    """The case that used to be 0/0 and is now a number."""

    def test_zero_total_weight_gives_zero_not_nan(self):
        """A vertex pressed only grazingly earns zero credit, so zero is its rate.

        Under the weighted mean this was ``0/0``: undefined, dropped from the map,
        counted as an exclusion.  The divisor is now the contact count, which is
        positive, so the estimator answers ``0.0`` — and that is a measurement,
        not a gap.  Reporting it as missing data would now be the dishonest
        option, because nothing is missing.
        """
        # value_sum is 0 because every contact contributed ``0 * iff``.
        got = attributed_mean_or_nan(np.array([0.0, 5.0]), np.array([4.0, 2.0]))
        assert got[0] == 0.0
        assert not np.isnan(got[0])
        assert got[1] == 2.5

    def test_it_is_kept_in_the_map(self):
        """``has_estimate`` is what decides which vertices reach the ``.npz``."""
        assert has_estimate(np.array([0.0]), np.array([4.0])).tolist() == [True]

    def test_zero_credit_is_counted_separately(self):
        """That ``0.0`` is unreadable without a count beside it.

        A vertex at 0.0 for lack of depth and a vertex at 0.0 because its neuron
        stayed silent look identical on screen and on disk.  The count is what
        keeps them apart, and it is a diagnostic about vertices that are *on* the
        map — never an exclusion tally.
        """
        count = np.array([4.0, 2.0, 3.0])
        weight_sum = np.array([0.0, 1.5, 0.0])
        assert count_zero_credit(count, weight_sum) == 2

    def test_an_uncontacted_vertex_is_not_counted_as_grazing(self):
        """``count == 0`` and ``weight_sum == 0`` is *never touched*, not *grazed*.

        Inferring grazing from ``weight_sum == 0`` alone would report the whole
        un-pressed remainder of the mesh as zero-credit.
        """
        assert count_zero_credit(np.array([0.0]), np.array([0.0])) == 0

    def test_it_is_zero_when_every_weight_is_one(self):
        """The alpha = 0 invariant: no vertex can carry zero credit there."""
        assert count_zero_credit(np.full(6, 3.0), np.full(6, 3.0)) == 0

    def test_a_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same vertices in the same order"):
            count_zero_credit(np.zeros(3), np.zeros(4))


class TestNoEstimate:
    def test_a_nan_numerator_is_nan(self):
        """No neural measurement is *no estimate* — the one remaining cause."""
        got = attributed_mean_or_nan(np.array([np.nan, 5.0]), np.array([2.0, 2.0]))
        assert np.isnan(got[0])
        assert got[1] == 2.5

    def test_an_uncontacted_vertex_is_nan(self):
        """``count == 0`` is the un-pressed remainder of the mesh."""
        got = attributed_mean_or_nan(np.array([0.0]), np.array([0.0]))
        assert np.isnan(got[0])

    def test_there_is_no_fallback_to_the_unweighted_mean(self):
        """A zero-credit entry does not borrow the mean of its neighbours.

        Falling back would put two different estimators in one map: the
        neighbouring vertices would be attributed rates and this one would not,
        and nothing on disk would say which was which.  The answer is the
        estimator's own ``0.0``, not a substituted number.
        """
        got = attributed_mean_or_nan(np.array([0.0, 40.0]), np.array([4.0, 2.0]))
        assert got[0] == 0.0
        assert got[1] == 20.0

    def test_a_shape_mismatch_raises(self):
        """Sliced differently means one vertex divided by another's count."""
        with pytest.raises(ValueError, match="same vertices in the same order"):
            attributed_mean_or_nan(np.zeros(3), np.zeros(4))

    def test_a_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            attributed_mean_or_nan(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_a_negative_count_raises(self):
        """The guard against passing ``weight_sum`` where ``count`` is expected.

        A count cannot be negative, so a negative one is proof the wrong array
        arrived.  It cannot catch every such mix-up — a non-negative weight sum
        looks like a plausible count — but a silent swap of the two divisors is
        exactly the failure this module exists to prevent, so the one detectable
        form of it raises.
        """
        with pytest.raises(ValueError, match="cannot be negative"):
            attributed_mean_or_nan(np.array([1.0]), np.array([-1.0]))


class TestHasEstimate:
    def test_it_is_the_complement_of_the_nan_mask(self):
        values = np.array([0.0, np.nan, 6.0])
        counts = np.array([0.0, 2.0, 3.0])
        assert has_estimate(values, counts).tolist() == [False, False, True]


class TestCountNoEstimate:
    def test_the_two_causes_are_counted_apart(self):
        """"No neural value" and "never contacted" are different statements.

        Both give NaN, but a caller that has restricted to the contacted set can
        only ever see the first — so seeing the second means the restriction was
        not applied, and merging the counts would hide that.
        """
        values = np.array([0.0, np.nan, 6.0, 0.0, np.nan])
        counts = np.array([0.0, 2.0, 3.0, 0.0, 5.0])
        got = count_no_estimate(values, counts)
        assert got == NoEstimateCounts(nan_value=2, no_contact=2)
        assert got.total == 4

    def test_an_uncontacted_entry_is_counted_once_not_twice(self):
        """It is NaN for *both* reasons; it belongs to the no-contact bucket.

        Otherwise ``total`` would exceed the number of excluded vertices and the
        summary JSON would over-report.
        """
        got = count_no_estimate(np.array([np.nan]), np.array([0.0]))
        assert got == NoEstimateCounts(nan_value=0, no_contact=1)
        assert got.total == 1

    def test_a_grazing_vertex_is_not_an_exclusion(self):
        """The behaviour change, stated where the summary JSON reads it.

        A contacted, entirely-grazing vertex has ``value_sum == 0`` and a positive
        count.  It has an estimate — ``0.0`` — so it must not appear in any
        no-estimate bucket.
        """
        got = count_no_estimate(np.array([0.0]), np.array([4.0]))
        assert got == NoEstimateCounts(nan_value=0, no_contact=0)
        assert got.total == 0

    def test_a_contacted_selection_with_values_counts_nothing(self):
        got = count_no_estimate(np.arange(6.0), np.ones(6))
        assert got.total == 0

    def test_an_empty_selection_counts_nothing(self):
        got = count_no_estimate(np.empty(0), np.empty(0))
        assert got == NoEstimateCounts(nan_value=0, no_contact=0)
