"""Exercise ModelRunner's initialization scope without loading model weights."""
import ast
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.overrides import _get_current_function_mode_stack
from torch.utils._device import DeviceContext


def initialization_scope():
    # Execute the actual initialization block; only model loading/capture is
    # replaced by a stub. This catches scope and dtype regressions in the caller.
    path = Path(__file__).parents[1] / 'ssd/engine/model_runner.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    start = next(i for i, n in enumerate(init.body)
                 if isinstance(n, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == 'default_dtype' for t in n.targets))
    end = next(i for i in range(start + 1, len(init.body))
               if ast.unparse(init.body[i]) == 'torch.set_default_dtype(default_dtype)')
    return compile(ast.Module(body=init.body[start:end + 1], type_ignores=[]), str(path), 'exec')


class ScopedDefaultDeviceTests(unittest.TestCase):
    def exercise(self, fail=False):
        original_dtype = torch.get_default_dtype()
        original_stack = _get_current_function_mode_stack()
        seen = []

        def setup(*args):
            modes = _get_current_function_mode_stack()
            self.assertIsInstance(modes[-1], DeviceContext)
            self.assertEqual(modes[-1].device.type, 'cuda')
            self.assertEqual(torch.get_default_dtype(), torch.float16)
            seen.append(True)
            if fail:
                raise RuntimeError('simulated model initialization failure')
            return 'stub'

        runner = SimpleNamespace(hf_config=SimpleNamespace(torch_dtype=torch.float16),
                                 is_draft=True, use_eagle=False, verbose=False,
                                 setup_and_warmup_model_and_cudagraphs=setup)
        scope = dict(torch=torch, os=os, self=runner, num_tp_gpus=1,
                     config=None, init_q=None, is_draft=True)
        try:
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'simulated'):
                        exec(initialization_scope(), scope)
                else:
                    exec(initialization_scope(), scope)
                    self.assertEqual(torch.get_default_dtype(), original_dtype)
            self.assertEqual(seen, [True])
            self.assertEqual(_get_current_function_mode_stack(), original_stack)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_default_restores_callers_mode_stack_and_dtype(self):
        self.exercise()

    def test_nested_device_scope_restores_callers_context(self):
        with torch.device('cpu'):
            self.exercise()

    def test_device_scope_restored_when_initialization_fails(self):
        self.exercise(fail=True)


if __name__ == '__main__':
    unittest.main()
