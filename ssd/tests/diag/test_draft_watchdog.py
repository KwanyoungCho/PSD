"""The target parent must not survive an unexpected draft-worker exit."""
import types
import unittest
from unittest import mock

from ssd.engine.llm_engine import LLMEngine


class _Worker:
    def __init__(self):
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        return None

    def kill(self):
        self.alive = False


class TestDraftWatchdog(unittest.TestCase):
    def test_dead_draft_terminates_target_workers_and_parent(self):
        engine = object.__new__(LLMEngine)
        engine._draft_watch_stop = types.SimpleNamespace(
            wait=lambda timeout: False)
        engine.draft_ps = types.SimpleNamespace(
            is_alive=lambda: False, exitcode=17)
        engine._exiting = False
        worker = _Worker()
        engine.ps = [worker]

        with mock.patch("ssd.engine.llm_engine.os._exit",
                        side_effect=SystemExit(17)) as exit_mock:
            with self.assertRaises(SystemExit):
                engine._watch_draft_process()

        self.assertTrue(worker.terminated)
        exit_mock.assert_called_once_with(17)


if __name__ == "__main__":
    unittest.main()
