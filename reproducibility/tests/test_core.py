import unittest

from reproducibility.code.ranking_transfer import predictive_preference
from reproducibility.code.warning_policies import calibrate_threshold, policy_statistic


class CoreTests(unittest.TestCase):
    def test_predictive_tie_uses_smaller_candidate(self):
        self.assertEqual(predictive_preference(0.2, 0.2, 8, 1), 1)

    def test_first_crossing_statistic(self):
        self.assertEqual(policy_statistic([0.1, 0.7, 0.2], "first_crossing"), 0.7)

    def test_threshold_respects_target(self):
        self.assertEqual(calibrate_threshold([0.1, 0.2, 0.3, 0.4], 0.25), 0.4)


if __name__ == "__main__":
    unittest.main()
