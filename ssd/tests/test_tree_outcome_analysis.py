"""CPU tests for post-hoc dynamic-tree policy attribution."""
import unittest

from tools.duet_calibration.analyze_tree_outcomes import (
    _one_tree, summarize_p1_roots, summarize_tree_rows)


class TestTreeOutcomeAnalysis(unittest.TestCase):
    def test_alternative_sibling_attributes_suffix_not_prefix(self):
        serve = {
            "phase": 1, "valid": 5,
            # root children 0/1, then children below 1 and 2.
            "par": [-1, -1, 1, 2, 2],
            "sib": [0, 1, 0, 0, 1],
            "root_start_score": 0.2, "root_context_id": 3,
        }
        walk = {
            "phase": 1, "par": serve["par"], "sib": serve["sib"],
            "path": [1, 2, 4],
        }
        row = _one_tree(serve, walk)
        self.assertEqual(row["accepted"], 3)
        self.assertEqual(row["first_child_prefix"], 0)
        self.assertEqual(row["branch_assisted"], 3)
        self.assertTrue(row["used_alternative"])
        summary = summarize_tree_rows([row])
        self.assertEqual(summary["trees_using_alternative_sibling"], 1)
        self.assertEqual(summary["accepted_sibling_order_counts"],
                         {"0": 1, "1": 2})

    def test_first_child_only_path_has_no_tree_rescue(self):
        serve = {"phase": 2, "valid": 3,
                 "par": [-1, -1, 0], "sib": [0, 1, 0]}
        walk = {"phase": 2, "par": serve["par"], "sib": serve["sib"],
                "path": [0, 2]}
        row = _one_tree(serve, walk)
        self.assertEqual(row["first_child_prefix"], 2)
        self.assertEqual(row["branch_assisted"], 0)

    def test_p1_auc_separates_reach_from_local_probability(self):
        rows = [
            {"context_id": 0, "start_score": .8,
             "context_reach": .9, "local_q": .9, "hit": 1},
            {"context_id": 1, "start_score": .2,
             "context_reach": .1, "local_q": .95, "hit": 0},
            {"context_id": 0, "start_score": .6,
             "context_reach": .8, "local_q": .8, "hit": 1},
            {"context_id": 2, "start_score": .1,
             "context_reach": .05, "local_q": .99, "hit": 0},
        ]
        result = summarize_p1_roots(rows)
        self.assertEqual(result["actual_hits"], 2)
        self.assertEqual(result["ranking_auc"]["start_score"], 1.0)
        self.assertEqual(result["ranking_auc"]["context_reach"], 1.0)
        self.assertEqual(result["ranking_auc"]["local_q"], 0.0)


if __name__ == "__main__":
    unittest.main()
