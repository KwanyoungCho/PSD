import ast
import importlib.util
from pathlib import Path
import unittest

_path = Path(__file__).resolve().parents[1] / "ssd/engine/helpers/target_verify_buckets.py"
_spec = importlib.util.spec_from_file_location("target_verify_buckets_cpu", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
target_verify_buckets = _module.target_verify_buckets
select_target_verify_bucket = _module.select_target_verify_bucket


class TargetVerifyBuckets(unittest.TestCase):
    def test_default_retains_phase_caps(self):
        self.assertEqual(target_verify_buckets([14, 10]), [10, 14])
        self.assertEqual(target_verify_buckets([12, 12]), [12])

    def test_extra_shapes_do_not_remove_phase_caps(self):
        self.assertEqual(target_verify_buckets([14, 10], "2,4,2"), [2, 4, 10, 14])

    def test_exact_no_padding_all_widths(self):
        for maximum in range(1, 31):
            buckets = target_verify_buckets([maximum], "exact")
            for valid in range(1, maximum + 1):
                self.assertEqual(select_target_verify_bucket(buckets, valid), valid)

    def test_historical_g12_short_shapes_exact(self):
        counts = [2, 4, 6, 8, 10, 11, 12]
        buckets = target_verify_buckets([12], "2,4,6,8,10,11")
        self.assertEqual(buckets, counts)
        for valid in counts:
            self.assertEqual(select_target_verify_bucket(buckets, valid), valid)

    def test_sparse_fallback(self):
        buckets = target_verify_buckets([12], "2,4")
        self.assertEqual([select_target_verify_bucket(buckets, v) for v in (1, 2, 3, 4, 5, 12)], [2, 2, 4, 4, 12, 12])

    def test_fail_fast(self):
        for spec in ("0", "-1", "13", "2,", ",2", "2.0", "typo"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                target_verify_buckets([12], spec)
        for caps in ([], [0], [-1, 12]):
            with self.assertRaises(ValueError):
                target_verify_buckets(caps)
        for valid in (0, 13):
            with self.assertRaises(ValueError):
                select_target_verify_bucket([12], valid)

    def test_final_head_is_trimmed_before_computation(self):
        path = Path(__file__).resolve().parents[1] / "ssd/engine/model_runner.py"
        module = ast.parse(path.read_text())
        runner = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_run_tree_verify")
        heads = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "compute_logits"]
        self.assertTrue(any(ast.unparse(n.args[0]) == "out[:n_rows]" for n in heads))


if __name__ == "__main__":
    unittest.main()
