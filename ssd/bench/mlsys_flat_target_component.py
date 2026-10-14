"""Same-input full-model target costs for padded/replanned/fixed packed trees.

Measures target-only wall latency and CUDA pre/post/head cost. This does not
predict online TPS or draft P1/P2 overlap. Node distributions are declared
synthetic shapes, with real model weights and real prompt-prefilled KV.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import hashlib
if os.getenv('PAPER_CPUSET'):
    cpus=set()
    for part in os.environ['PAPER_CPUSET'].split(','):
        ends=list(map(int,part.split('-')))
        cpus.update(range(ends[0],ends[-1]+1))
    os.sched_setaffinity(0,cpus)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
import torch
from transformers import AutoTokenizer
from ssd.config import Config
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.sampling_params import SamplingParams
from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward
from ssd.engine.helpers.packed_tree_forward import PackedTreeForward
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy
from mlsys_tree_candidates import FixedPlanPackedTree
import mlsys_flat_tree as flat
os.environ["DUET_REVIEW_FLAT"]="1"
flat.install()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    cfg = Config(model=args.model, num_gpus=1, max_num_seqs=8,
                 enforce_eager=True, max_model_len=1024,
                 max_num_batched_tokens=4096, gpu_memory_utilization=.8)
    cfg.duet_exit_layer = 21
    runner = ModelRunner(cfg, 0, None, num_tp_gpus=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prefix = tokenizer.encode('A computer system processes requests and reuses previously computed information. ' * 40)[:128]
    Sequence.block_size = runner.block_size
    seqs = [Sequence(prefix, SamplingParams()) for _ in range(8)]
    for i, seq in enumerate(seqs):
        seq.block_table = [i]
    runner.run(seqs, True)
    report = dict(status='running', model=args.model, gpu=torch.cuda.get_device_name(0),
                  cpus=sorted(os.sched_getaffinity(0)),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='fixed synthetic tree shapes, real weights and prefilled KV; no draft', cells=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in (8,):
        for shape in ('uniform', 'half_short', 'one_hit'):
            lengths = ([8] * batch if shape == 'uniform' else
                       [8] * (batch//2) + [2] * (batch-batch//2) if shape == 'half_short'
                       else [8] + [0] * (batch-1))
            full_par = [-1, -1, 0, 0, 1, 2, 3, 4]
            full_sib = [0, 1, 0, 1, 0, 0, 0, 0]
            rows = [dict(tokens=[100+i] + list(range(200, 200+n)),
                         parents=full_par[:n], prefix=len(prefix), blocks=[i])
                    for i, n in enumerate(lengths)]
            proxy = BatchedTreeProxy(batch, 8, cfg.hf_config.vocab_size,
                cfg.hf_config.torch_dtype, runner.device, 15, 8, 13)
            tok = torch.zeros(batch, 8, dtype=torch.long, device=runner.device)
            for i, n in enumerate(lengths):
                tok[i, :n] = torch.tensor(rows[i]['tokens'][1:], device=runner.device)
            proxy.prepare([full_par[:n] for n in lengths], [full_sib[:n] for n in lengths],
                          tok, torch.zeros_like(proxy.q), [.7]*batch, [.7]*batch)
            engines = {'padded': BatchedTreeForward(runner, True),
                       'replan': PackedTreeForward(runner, True),
                       'fixed': FixedPlanPackedTree(runner, True), 'flat':flat.FlatForward(runner, True)}
            fp=flat.FlatProxy(batch,8,cfg.hf_config.vocab_size,cfg.hf_config.torch_dtype,runner.device,15,8,13)
            fp.prepare([full_par[:n] for n in lengths],[full_sib[:n] for n in lengths],tok,torch.zeros_like(proxy.q),[.7]*batch,[.7]*batch)
            references = None
            for name, stream in (('padded', False), ('fixed',False), ('flat',False), ('flat',False), ('fixed',False), ('padded',False)):
                if name=='flat' and sum(lengths)+batch>=batch*9:continue
                os.environ['SSD_BATCH_TREE_PROXY_STREAM'] = str(int(stream))
                engine = engines[name]
                def invoke():
                    return engine.run(rows, 9, lambda logits,b:(fp if name=='flat' else proxy).replay(logits))
                for _ in range(3):
                    output = invoke()
                torch.cuda.synchronize()
                live = torch.tensor([j < n+1 for n in lengths for j in range(9)], device=runner.device)
                actual = (output[:sum(lengths)+batch] if name=='flat' else output.reshape(-1, output.shape[-1])[live]).clone()
                if not torch.isfinite(actual).all():
                    raise AssertionError('Nonfinite live target logits')
                if references is None:
                    references = actual.clone()
                diff = float((actual - references).abs().max())
                agree = float((actual.argmax(-1) == references.argmax(-1)).float().mean())
                # Shape-dependent GEMMs may round differently. Record numerical
                # differences rather than asserting bitwise AR identity.
                timings = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    invoke()
                    torch.cuda.synchronize()
                    timings.append((time.perf_counter()-start)*1000)
                # A separate instrumented pass keeps CUDA-event bookkeeping
                # out of the uninstrumented wall-latency measurements above.
                from ssd.engine.helpers import cudagraph_helpers as profiling
                profiling.duet_reset()
                profiling.PROFILE_DUET = True
                for _ in range(5):
                    invoke()
                torch.cuda.synchronize()
                components = {}
                for event in profiling._duet_events:
                    components.setdefault(event[1], []).append(event[3].elapsed_time(event[4]))
                profiling.PROFILE_DUET = False
                profiling.duet_reset()
                report['cells'].append(dict(batch=batch, shape=shape, lengths=lengths,
                    implementation=name, stream=stream, wall_ms=statistics.median(timings),
                    samples_ms=timings, cuda_components_ms={k:statistics.median(v) for k,v in components.items()},
                    max_logit_diff=diff, argmax_agreement=agree,
                    physical_queries=(batch*9 if name=='padded' else
                        min(batch*9,((sum(lengths)+batch+7)//8)*8) if name in ('fixed','flat') else
                        ((sum(lengths)+batch+7)//8)*8)))
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(report['cells'][-1] | {'samples_ms': []}), flush=True)
            del engines, proxy, fp, references, actual, output
            gc.collect()
            torch.cuda.empty_cache()
    report['status']='complete'
    args.output.write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
