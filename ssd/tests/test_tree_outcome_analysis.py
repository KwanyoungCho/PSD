"""CPU tests for post-hoc dynamic-tree policy attribution."""
import unittest

from tools.duet_calibration.analyze_tree_outcomes import (
    _match_served_draft, _one_tree, _served_to_generated_indices,
    summarize_p1_roots, summarize_tree_rows)


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

    def test_post_rerank_wire_ids_map_back_to_generated_tree(self):
        root = {
            "piv": 0.25,
            "par": [-1, -1, 0, 0, 2, 2],
            "sib": [0, 1, 0, 1, 0, 1],
            "raw_q": [0.9, 0.01, 0.8, 0.02, 0.7, 0.03],
        }
        # cap=4 keeps generated ids [0,2,4,5], then compacts parents.
        serve = {
            "step": 2, "phase": 1, "root_rank": 0,
            "root_start_score": 0.25, "root_context_id": 3,
            "valid": 4, "par": [-1, 0, 1, 1], "sib": [0, 0, 0, 1],
        }
        self.assertEqual(_served_to_generated_indices(root, serve),
                         [0, 2, 4, 5])

    def test_served_match_does_not_assume_prompt_local_step_is_global(self):
        root_a = {"piv": 0.1, "par": [-1], "sib": [0], "raw_q": [0.8]}
        root_b = {"piv": 0.2, "par": [-1], "sib": [0], "raw_q": [0.9]}
        drafts = [
            {"trace_seq": 1, "phase": 2, "roots": [root_a],
             "root_context_ids": None},
            # A second generate() may report public step=2 while topology
            # trace_seq has continued globally to 71.
            {"trace_seq": 71, "phase": 2, "roots": [root_b],
             "root_context_ids": None},
        ]
        serve = {"step": 2, "phase": 2, "root_rank": 0,
                 "root_start_score": 0.2, "root_context_id": None,
                 "valid": 1, "par": [-1], "sib": [0]}
        draft, root, mapping = _match_served_draft(drafts, serve, 1)
        self.assertEqual(draft["trace_seq"], 71)
        self.assertIs(root, root_b)
        self.assertEqual(mapping, [0])


if __name__ == "__main__":
    unittest.main()
