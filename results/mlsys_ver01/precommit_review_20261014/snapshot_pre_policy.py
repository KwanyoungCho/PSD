"""Recover the two-file policy delta and verify every byte against run hashes."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = Path('/data/chokwans99/duet-precommit-20261014/tp2_full/base.json')
run = json.loads(RAW.read_text())
checks = json.loads((HERE / 'pre_policy_suites/CHECKS_PROVENANCE.json').read_text())
data = {p: (ROOT / p).read_bytes() for p in checks['source_hashes']}
cfg = 'ssd/ssd/config.py'
text = data[cfg].decode()
start = text.index('        # Fixed-input gains did not establish')
end = text.index('\n\n    @property', start)
text = text[:start] + '''        explicit=_os_cfg.environ.get("SSD_DUET_EXIT_REPLICA")
        if explicit is not None:return explicit == "1"
        # Full-head projection can overlap target post layers without a TP
        # rendezvous. Existing single-request/alternative-proxy modes retain
        # their contract; no extra tuning parameter is introduced.
        return (self.max_num_seqs>1 and self.duet_tree_enabled
                and self.draft_async and self.num_gpus>2
                and not self.duet_exit_topm_gather and not self.duet_proxy_on_draft
                and not self.target_quant_lm_head)''' + text[end:]
data[cfg] = text.encode()
llm = 'ssd/ssd/engine/llm_engine.py'
data[llm] = data[llm].replace(b"replica_source = 'default'", b"replica_source = 'auto'")
test = 'ssd/tests/test_tree_proxy_defaults.py'
data[test] = data[test].replace(b'def test_replica_requires_explicit_opt_in',
                              b'def test_replica_default_requires_batched_async_tp_tree')
data[test] = data[test].replace(
    b'self.assertFalse(Config.duet_exit_replica.fget(self.config()))',
    b'self.assertTrue(Config.duet_exit_replica.fget(self.config()))', 1)
for path, expected in checks['source_hashes'].items():
    if hashlib.sha256(data[path]).hexdigest() != expected:
        raise AssertionError(f'Cannot reconstruct recorded source: {path}')
for path, expected in run['source_hashes'].items():
    if hashlib.sha256(data[path]).hexdigest() != expected:
        raise AssertionError(f'Run source mismatch: {path}')
for name, expected in run['adapter_hashes'].items():
    path = 'ssd/bench/' + name
    source = (ROOT / path).read_bytes()
    if name == 'mlsys_tree_followup.py':
        source = source.replace(b"replica_source='env' if 'SSD_DUET_EXIT_REPLICA' in os.environ else 'default'",
                                b"replica_source='env' if 'SSD_DUET_EXIT_REPLICA' in os.environ else 'auto'")
    if hashlib.sha256(source).hexdigest() != expected:
        raise AssertionError(f'Run adapter mismatch: {path}')
    data[path] = source
with tarfile.open(HERE / 'PRE_POLICY_SOURCE.tar.gz', 'w:gz') as tar:
    for name, content in sorted(data.items()):
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
(HERE / 'PRE_POLICY_SOURCE.json').write_text(json.dumps({
    'all_runtime_test_and_adapter_hashes_match': True,
    'files': {p: hashlib.sha256(content).hexdigest() for p, content in data.items()},
    'note': 'Reconstructed from the final source by reversing only the recorded policy/log/test changes; every byte is verified against pre-policy regression and run hashes.'}, indent=2))
print('Pre-policy runtime, tests and adapters exactly match recorded hashes.')
