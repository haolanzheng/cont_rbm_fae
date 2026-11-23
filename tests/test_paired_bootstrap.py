"""Dependency-light tests runnable with unittest (and discoverable by pytest)."""
import unittest

import numpy as np

from tools.audit_paired_bootstrap import pair_max_errors, paired_bootstrap


class PairedBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            {"seed": 2, "rank": 12, "max_error": 6.0},
            {"seed": 1, "rank": 12, "max_error": 3.0},
        ]
        self.adam = [
            {"seed": 1, "rank": 12, "max_error": 1.0},
            {"seed": 2, "rank": 12, "max_error": 2.0},
        ]

    def test_proportional_pairs_remain_proportional_in_every_resample(self):
        adam = np.array([1.0, 2.0, 4.0, 9.0])
        result = paired_bootstrap(3 * adam, adam, n_resamples=500)
        self.assertEqual(result["ratio_of_medians_pool_over_adam"], 3.0)
        np.testing.assert_allclose(result["paired_percentile95"], [3.0, 3.0])

    def test_pairing_by_seed_not_list_order(self):
        seeds, pool, adam = pair_max_errors(self.pool, self.adam)
        self.assertEqual(seeds, [1, 2])
        np.testing.assert_array_equal(pool, 3 * adam)

    def test_reject_duplicate_missing_and_different_rank_partners(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pair_max_errors(self.pool + self.pool[:1], self.adam)
        with self.assertRaisesRegex(ValueError, "same nonempty seed"):
            pair_max_errors(self.pool, self.adam[:1])
        with self.assertRaisesRegex(ValueError, "same reduced rank"):
            pair_max_errors(self.pool, [dict(r, rank=10) for r in self.adam])

    def test_estimator_and_reproducibility(self):
        pool = np.array([1.0, 10.0, 11.0])
        adam = np.array([2.0, 3.0, 100.0])
        result = paired_bootstrap(pool, adam, n_resamples=100)
        self.assertEqual(result["ratio_of_medians_pool_over_adam"], 10 / 3)
        self.assertNotEqual(
            result["ratio_of_medians_pool_over_adam"], np.median(pool / adam)
        )
        self.assertEqual(result, paired_bootstrap(pool, adam, n_resamples=100))

    def test_reject_invalid_errors_and_resample_count(self):
        for pool, adam in [
            ([], []), ([1], [1, 2]), ([0], [1]), ([1], [np.nan]),
            ([[1]], [[1]]),
        ]:
            with self.subTest(pool=pool, adam=adam), self.assertRaises(ValueError):
                paired_bootstrap(pool, adam)
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1], n_resamples=0)


if __name__ == "__main__":
    unittest.main()
