"""Isolated verifier latency, not an end-to-end throughput measurement."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd.utils.verify import verify
from ssd.utils.verify_fast import verify_stochastic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = []
    for vocab in (32000, 128256):
        for b in (1, 4, 8):
            for t in (0., .7):
                torch.manual_seed(0)
                pp = torch.randn(b, 5, vocab, dtype=torch.bfloat16, device="cuda")
                qq = torch.randn(b, 4, vocab, dtype=torch.bfloat16, device="cuda")
                tokens = torch.cat((torch.zeros(b, 1, dtype=torch.long, device="cuda"),
                                    qq.argmax(-1)), dim=1)
                temps = torch.full((b,), t, device="cuda")
                args = (pp, qq, tokens, temps, temps)
                funcs = dict(reference=lambda: verify(*args, jit_speculate=True),
                             optimized=(lambda: verify_stochastic(*args)) if t else
                                       (lambda: verify(*args, jit_speculate=True, all_greedy=True)))
                for rep in range(3):
                    for name in (list(funcs) if rep % 2 == 0 else list(reversed(funcs))):
                        fn = funcs[name]
                        for _ in range(10):
                            fn()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(a.iterations):
                            fn()
                        torch.cuda.synchronize()
                        rows.append(dict(vocab=vocab, batch=b, temperature=t, repeat=rep,
                                         mode=name, ms=(time.perf_counter()-start)*1000/a.iterations))
    a.output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
                                       torch=torch.__version__, rows=rows), indent=2))


if __name__ == "__main__":
    main()
