"""Portable ver1.8 B1 measurement; performance is the default, timeline explicit.

Uses native phase fanout configuration. No sampler, verifier or per-step
algorithm replacement. See MLSYS_VER18_HANDOVER.md for the fixed protocol.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
# Run in parent and spawned draft before importing any runtime/profile module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duet_measurement_mode import configure_measurement
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument('--measurement-mode', choices=('performance','timeline'), default='performance')
_early.add_argument('--seed', type=int, default=42)
_early_args, _ = _early.parse_known_args()
configure_measurement(_early_args.measurement_mode)
os.environ['PAPER_SEED'] = os.environ['SSD_SEED'] = str(_early_args.seed)
from mlsys_measurement import configure_capture_audit, finalize_measurement
from mlsys_request_reporting import (build_requests, question_summary, safe_tps,
                                     request_reporting_supported)
configure_capture_audit()
# Spawned draft workers must receive the same RNG seed in both arms.
import torch
random.seed(int(os.environ.get("PAPER_SEED", "9101")))
torch.manual_seed(int(os.environ.get("PAPER_SEED", "9101")))

# Set in every spawned worker, before any ssd import.
CHECKOUT = Path(__file__).resolve().parents[2]
if os.getenv('PAPER_CPUSET'):
    cpus=set()
    for part in os.environ['PAPER_CPUSET'].split(','):
        ends=[int(v) for v in part.split('-')]
        cpus.update(range(ends[0],ends[-1]+1))
    os.sched_setaffinity(0,cpus)
sys.path.insert(0, str(CHECKOUT/'ssd'))
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
# Native model support: no dtype/loader/verification monkeypatches.
# Only the declared cold-prefix measurement population is shared with baseline.
if os.getenv('PAPER_COLD_PREFIX', '1') == '1':
    from ssd.engine.block_manager import BlockManager
    _allocate = BlockManager.allocate
    def _cold_allocate(self, seq):
        self.hash_to_block_id.clear()
        return _allocate(self, seq)
    BlockManager.allocate = _cold_allocate

from mlsys_capture_audit import install as install_capture_audit
install_capture_audit()

if _early_args.measurement_mode == 'timeline':
    from mlsys_phase_probe import install
    install()


# Bound CUDA event retention to one request. Flushing is between requests,
# never inside a decode step; both arms use this same profiling adapter.
if os.getenv('SSD_PROFILE_DUET', '0') == '1':
    from ssd.engine.draft_runner import DraftRunner
    from ssd.engine.helpers.cudagraph_helpers import duet_dump
    _prefill = DraftRunner.draft_async_prefill
    def _flush_previous_request(self, *args, **kwargs):
        previous = getattr(self, '_ps_profile_request', -1)
        if previous >= 0:
            duet_dump('draft', run_id=f'request_{previous:04d}')
        self._ps_profile_request = previous + 1
        return _prefill(self, *args, **kwargs)
    DraftRunner.draft_async_prefill = _flush_previous_request


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--dtype', choices=['float16','bfloat16'], default='float16')
    p.add_argument('--measurement-mode', choices=['performance','timeline'], default='performance')
    p.add_argument('--mode', choices=['tree', 'chain'], default='tree')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--input-cap', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warm-prompts', type=int, default=3)
    p.add_argument('--exit-layer', type=int, default=21)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--p1-verify', type=int, default=15)
    p.add_argument('--p2-verify', type=int, default=15)
    p.add_argument('--k1', type=int, default=10)
    p.add_argument('--k2', type=int, default=5)
    p.add_argument('--p1-nodes', type=int, default=15)
    p.add_argument('--p2-nodes', type=int, default=15)
    p.add_argument('--roots', type=int, default=4)
    p.add_argument('--siblings', type=int, default=3)
    p.add_argument('--p1-c', type=int, default=2)
    p.add_argument('--p2-c', type=int, default=3)
    p.add_argument('--passes', type=int, default=1)
    p.add_argument('--target-tp', type=int, default=1)
    p.add_argument('--ds-width', type=int, default=40)
    p.add_argument('--ps-width', type=int, default=20)
    p.add_argument('--ps-roots', type=int, default=16)
    p.add_argument('--expansion', choices=['last','frontier'], default='last')
    return p.parse_args()


def main():
    args = parse_arguments()
    if args.batch != 1 or args.target_tp != 1 or args.passes != 1:
        raise ValueError('This preset supports B1, target TP1 and one pass per seed')
    if args.input_cap != 0 or args.warm_prompts < 0:
        raise ValueError('Raw inputs must be preserved; warmup must be nonnegative')
    protocol=dict(warmup_scope='within_dataset', requested_seeds=[args.seed],
        warmup_requests=args.warm_prompts, measurement_mode=args.measurement_mode)
    os.environ['SSD_PROFILE_DIR'] = str(args.output.resolve().parent)
    os.environ.setdefault('SSD_SHM_NAME', 'duet18_'+str(os.getpid()))
    os.environ.setdefault('SSD_DIST_PORT', '49618')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if min(args.k1,args.k2,args.roots,args.siblings,args.passes)<1: raise ValueError('Positive rounds, roots, siblings and passes required')
    if args.p1_verify!=args.p1_nodes or args.p2_verify!=args.p2_nodes: raise ValueError('Generation and verification caps must be equal')
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    from ssd import LLM, SamplingParams
    from ssd.engine.llm_engine import METRICS
    from transformers import AutoTokenizer
    token = AutoTokenizer.from_pretrained(args.target)
    from mlsys_equal_dataset import load as load_inputs, attach as attach_input_identity
    prompts, indexes, input_metadata = load_inputs(args.prompts, token, args.limit, args.input_cap)
    if len(prompts) <= args.warm_prompts or not input_metadata:
        raise ValueError('Use structured raw-turn records, with measured turns after warmup')
    tree, spec = args.mode == 'tree', True
    report_requests = request_reporting_supported(args.mode)
    kwargs = dict(dtype=args.dtype, num_gpus=args.target_tp+1 if spec else args.target_tp, max_num_seqs=args.batch,
        max_model_len=2048, max_num_batched_tokens=args.batch*2048,
        gpu_memory_utilization=.45, draft_gpu_memory_utilization=.35, speculate=spec, draft_async=spec and args.mode != 'sd',
        jit_speculate=True, speculate_k=args.k1+args.k2, async_fan_out=args.roots+1)
    if spec: kwargs.update(draft=args.draft, duet_enabled=args.mode in ('tree','chain'), duet_exit_layer=args.exit_layer,
        duet_phase1_k=args.k1, duet_phase2_k=args.k2, duet_draft_fan_out=args.roots,
        duet_p1_tree_policy='on' if tree else 'off', duet_p2_tree_policy='on' if tree else 'off',
        duet_p1_tree_max_nodes=args.p1_nodes, duet_p1_tree_verify_nodes=args.p1_verify,
        duet_p2_tree_max_nodes=args.p2_nodes, duet_p2_tree_verify_nodes=args.p2_verify,
        duet_tree_c_tensor=args.siblings, duet_p1_tree_c_tensor=args.p1_c,
        duet_p2_tree_c_tensor=args.p2_c, duet_p1_roots_per_position=args.roots,
        duet_p1_forward_width=args.ds_width, duet_p2_forward_width=args.ps_width,
        duet_p2_root_count=args.ps_roots, duet_tree_expansion=args.expansion)
    if os.getenv('PAPER_BASELINE', '0') != '1':
        kwargs['greedy_only' if args.temperature == 0 else 'stochastic_only'] = True
    arm = os.environ.get('PAPER_ARM', 'base')
    report = dict(status='running', arm=arm, args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        paper_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip(),
        paper_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True),
        source_hashes={str(f.relative_to(CHECKOUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (CHECKOUT/'ssd/ssd').rglob('*.py')},
        adapter_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('mlsys_measurement.py')]},
        kwargs=kwargs, experiment_protocol=protocol, prompt_ids=prompts, question_indexes=indexes,
        env={k:v for k,v in os.environ.items() if k.startswith(('SSD_','PAPER_','DUET_REVIEW_')) or k=='CUDA_VISIBLE_DEVICES'},
        groups=[], warm_groups=[], steps=[], prefill=[], outputs=[],
        request_records=[] if report_requests else None,
        input_metadata=input_metadata,
        request_scope=('paper flattened raw turns; no generated conversation history'
                       if input_metadata else 'first turn of each question; repeated passes are not independent questions'))
    reporting_source = Path(__file__).with_name('mlsys_request_reporting.py')
    report['adapter_hashes'][reporting_source.name] = hashlib.sha256(reporting_source.read_bytes()).hexdigest()
    data_source=Path(__file__).with_name('mlsys_equal_dataset.py')
    report['adapter_hashes'][data_source.name]=hashlib.sha256(data_source.read_bytes()).hexdigest()
    def save():
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report));temp.replace(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);save()
    llm=None
    try:
        llm=LLM(args.target, **kwargs)
        if llm.config.sampler_x is not None:raise ValueError('Sampling rescaling must remain disabled')
        if tree:
            from ssd.engine.helpers.batch_tree_forward import proxy_stream_enabled
            report['resolved_tree_runtime'] = dict(
                target_tp=llm.num_tp_gpus, batch=llm.config.max_num_seqs,
                exit_replica=llm.config.duet_exit_replica,
                replica_source='env' if 'SSD_DUET_EXIT_REPLICA' in os.environ else 'default',
                proxy_stream=proxy_stream_enabled(llm.config, llm.num_tp_gpus),
                stream_source='env' if 'SSD_BATCH_TREE_PROXY_STREAM' in os.environ else 'auto')
        # Capture scheduler output without changing admission or model timing.
        current={}; group_sequences={}; terminal_reasons={}
        original_schedule=llm.scheduler.schedule
        def schedule():
            seqs, pref=original_schedule()
            current.update(seqs=list(seqs),prefill=pref, before=[s.num_tokens for s in seqs])
            for s in seqs:
                group_sequences[s.seq_id] = s
            if pref is None:
                # Explicit scheduler context-guard return; never infer this
                # reason merely from an empty output or a zero time.
                for s in seqs:
                    terminal_reasons[s.seq_id] = 'context_limit_guard'
            return seqs,pref
        llm.scheduler.schedule=schedule
        original_step=llm.step
        group_steps=[];group_prefill=[]
        def step(which):
            et=len(METRICS['phase_events']);tv=len(METRICS['target_verify_times'])
            dt=METRICS['decode_total_time'];pt=METRICS['prefill_total_time']
            audit_start=time.monotonic_ns()
            result=original_step(which)
            audit_end=time.monotonic_ns()
            seqs=current['seqs'];pref=current['prefill']
            if pref is True:
                group_prefill.append(dict(seconds=METRICS['prefill_total_time']-pt,batch=len(seqs)))
            elif pref is False:
                events=copy.deepcopy(METRICS['phase_events'][et:])
                # Review packed execution uses an adapter, not a production
                # config flag. Record the shape actually run by that adapter.
                queries=getattr(llm.model_runner,'_review_physical_queries',None)
                if queries is not None:
                    for event in events:event['physical_batch_queries']=queries
                emitted=[s.num_tokens-n for s,n in zip(seqs,current['before'])]
                cap=any(s.num_completion_tokens>=s.max_new_tokens for s in seqs)
                clipped=any(e['accepted_len']!=n for e,n in zip(events,emitted)) if spec else False
                group_steps.append(dict(audit_start_ns=audit_start,audit_end_ns=audit_end,seconds=METRICS['decode_total_time']-dt,batch=len(seqs),
                    cap=cap,clipped=clipped,emitted=sum(emitted),events=events,
                    seq_ids=[s.seq_id for s in seqs], emitted_by_request=emitted,
                    verify=METRICS['target_verify_times'][tv:]))
            return result
        llm.step=step
        # Explicit fixed-size groups keep the old scheduler's resident set
        # bounded without importing the new scheduler into the baseline.
        phases=[('warm',prompts[:args.warm_prompts])]+[(f'measured_{i}',prompts) for i in range(args.passes)]
        profile_request = 0
        for phase, data in phases:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            for start in range(0,len(data),args.batch):
                if protocol['warmup_scope']=='within_dataset' and phase.startswith('measured') and start<args.warm_prompts:continue
                group_steps.clear();group_prefill.clear()
                group_sequences.clear();terminal_reasons.clear()
                if args.batch != 1:raise ValueError('This frozen protocol adapter supports B1 only')
                effective_output_cap=min(args.output_tokens,llm.config.max_model_len-len(data[start]))
                if effective_output_cap < 1:raise ValueError('Input leaves no output capacity')
                params=SamplingParams(temperature=args.temperature,draft_temperature=args.temperature,max_new_tokens=effective_output_cap,ignore_eos=False)
                torch.cuda.synchronize();t=time.perf_counter()
                outputs,metrics=llm.generate(data[start:start+args.batch],params,use_tqdm=False)
                torch.cuda.synchronize();wall=time.perf_counter()-t
                if os.getenv('SSD_PROFILE_DUET','0')=='1':
                    duet_dump('target_rank0', run_id=f'request_{profile_request:04d}')
                    profile_request += 1
                records=dict(phase=phase,index=start,n=len(outputs),wall=wall,steps=copy.deepcopy(group_steps),
                    prefill=copy.deepcopy(group_prefill),outputs=[o['token_ids'] for o in outputs])
                records['native_performance']=copy.deepcopy(metrics.get('performance'))
                records['request_records'] = build_requests(group_sequences, group_steps,
                    terminal_reasons, indexes[start:start+len(outputs)],
                    records['outputs'], phase, llm.config.max_model_len, spec) if report_requests else None
                if report_requests:
                    attach_input_identity(records['request_records'], input_metadata[start:start+len(outputs)], phase)
                    for request, sid in zip(records['request_records'], sorted(group_sequences)):
                        request['seq_id']=sid
                        request['effective_output_cap']=effective_output_cap
                        request['target_temperature']=params.temperature
                        request['draft_temperature']=params.draft_temperature
                        request['ignore_eos']=params.ignore_eos
                report['warm_groups' if phase=='warm' else 'groups'].append(records)
                if phase.startswith('measured'):
                    report['steps'].extend(records['steps']);report['prefill'].extend(records['prefill']);report['outputs'].extend(records['outputs'])
                    if report_requests:
                        report['request_records'].extend(records['request_records'])
                if start%16==0:save();print('PAPER_GROUP',arm,phase,start,flush=True)
        included=[s for s in report['steps'] if s['batch']==args.batch and not s['cap'] and not s['clipped']]
        events=[e for s in included for e in s['events']]
        def mean(v):
            v=[x for x in v if x is not None]
            return sum(v)/len(v) if v else None
        report['summary']=dict(steps=len(included),all_steps=len(report['steps']),
            tokens=sum(s['emitted'] for s in included),seconds=sum(s['seconds'] for s in included),
            tps=safe_tps(sum(s['emitted'] for s in included),
                         sum(s['seconds'] for s in included),len(included)),
            verify_ms=mean([v*1000 for s in included for v in s['verify']]),
            step_ms=mean([s['seconds']*1000 for s in included]),
            al=mean([e['accepted_len'] for e in events]),hit=mean([e.get('cache_hit') for e in events]),
            actual_output_tokens=sum(map(len,report['outputs'])),
            emitted_decode_tokens=sum(s['emitted'] for s in report['steps']))
        if spec and report['summary']['actual_output_tokens']!=report['summary']['emitted_decode_tokens']:
            raise AssertionError('Actual output/decode token accounting mismatch')
        finalize_measurement(report)
        if args.measurement_mode == 'performance':
            from duet_question_mean import summarize
            report['question_mean'] = summarize(report['groups'])
            report['question_mean_unfiltered'] = summarize(report['groups'],filtered=False)
            for name,key in [('QUESTION_MEAN.json','question_mean'),('QUESTION_MEAN_RAW.json','question_mean_unfiltered')]:
                args.output.with_name(name).write_text(json.dumps(report[key],ensure_ascii=False,indent=2)+'\n')
        report['request_population'] = (question_summary(report['request_records'])
            if report_requests else dict(status='unavailable',
                reason='AR prefill tokens / synchronous SD lack the async decode-event contract; raw and aggregate metrics retained'))
        report['status']='complete';save();print('PAPER_COMPLETE',json.dumps(report['summary']),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        if llm:llm.exit(hard=False)

if __name__=='__main__':main()
