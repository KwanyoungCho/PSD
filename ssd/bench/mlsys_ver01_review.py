"""Diagnostic-only fallback assertions; do not use these runs for TPS."""
import json
import os
from pathlib import Path
import sys

from mlsys_ver01 import main
import torch
from ssd.engine.helpers import p2_tree as PT
from ssd.engine.draft_runner import DraftRunner

original_sample = PT.tree_sample_wor


def checked_sample(logits, temperatures, c_tensor, *args, **kwargs):
    result = original_sample(logits, temperatures, c_tensor, *args, **kwargs)
    caller = sys._getframe(1).f_code.co_name
    if caller in ("run_rollout", "run_rollout_arena"):
        assert kwargs.get("greedy_only") and kwargs.get("allow_greedy")
        assert bool((temperatures == 0).all())
        assert torch.equal(result[0][:, 0], logits.argmax(-1))
        trace = Path(os.environ["PAPER_REVIEW_TRACE"])
        trace.parent.mkdir(parents=True, exist_ok=True)
        with trace.open("a") as out:
            out.write(json.dumps(dict(caller=caller, rows=logits.shape[0],
                                      first_child_argmax=True)) + "\n")
    return result


PT.tree_sample_wor = checked_sample
if os.getenv("PAPER_REVIEW_FORCE_NONE") == "1":
    DraftRunner._try_p2_executor = lambda *args, **kwargs: None

if __name__ == "__main__":
    main()
