"""Original memory/score hooks plus isolated follow-up policies."""
import importlib.util
from pathlib import Path


def install():
    path = Path(__file__).resolve().parents[1]/'duet_tree_al_full/runtime.py'
    spec = importlib.util.spec_from_file_location('_original_full_runtime', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.install()
    from policy_hook import install as policy_install
    policy_install()
    from trace_hook import install as trace_install
    trace_install()
