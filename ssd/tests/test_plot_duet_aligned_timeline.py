import unittest

from bench.plot_duet_aligned_timeline import COLORS, pick_representative_step


def _target_step(step_id, duration_ms, status="hit_k1"):
    start = step_id * 1_000_000_000
    end = start + int(duration_ms * 1_000_000)
    return [{
        "step_id": step_id,
        "label": "target_spec_wait",
        "status": status,
        "wall_start_ns": start,
        "wall_end_ns": end,
    }]


class RepresentativeStepTest(unittest.TestCase):
    def setUp(self):
        self.target = (
            _target_step(1, 10.0)
            + _target_step(2, 20.0)
            + _target_step(3, 30.0)
        )

    def test_prefers_steps_retained_by_capped_draft_trace(self):
        draft = [
            {"step_id": 1, "label": "draft_send_response"},
            {"step_id": 3, "label": "draft_send_response"},
        ]
        self.assertEqual(
            pick_representative_step(self.target, "hit_k1", draft), 3)

    def test_falls_back_to_target_median_for_incomplete_legacy_trace(self):
        draft = [{"step_id": 9, "label": "draft_send_response"}]
        self.assertEqual(
            pick_representative_step(self.target, "hit_k1", draft), 2)

    def test_proxy_graph_breakdown_has_explicit_styles(self):
        for label in (
                "exit_proxy_launch", "exit_proxy_side",
                "chain_proxy_graph_replay", "tree_proxy_graph_replay",
                "proxy_send_enqueue"):
            self.assertIn(label, COLORS)


if __name__ == "__main__":
    unittest.main()
