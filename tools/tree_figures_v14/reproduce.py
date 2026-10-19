"""Portable orchestration around the unchanged, frozen v7 observer/harness.

This launcher is new; it has not launched a second GPU campaign locally.
Native and observed full runs are paired ON THE RECEIVING SERVER. No engine
source, model policy, candidate rule, or AL estimator is rewritten here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

BUNDLE = Path(__file__).resolve().parent
COMMIT = 'dedc85ccb299df2d76fcf34d50f05f2d8747a32a'
SEEDS = (42, 123, 2026)


def input_spec():
    return json.loads((BUNDLE / 'INPUT_SPEC.json').read_text())


def input_digest(rows):
    value = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode()).hexdigest()


def validate_inputs(rows):
    spec = input_spec()
    if (len(rows) != spec['full_records']
            or len({str(x['question_id']) for x in rows}) != spec['questions']
            or input_digest(rows) != spec['full_semantic_sha256']):
        raise ValueError('External inputs differ from the frozen original Spec-Bench records/order/token IDs')
    small = rows[:spec['smoke_records']]
    if input_digest(small) != spec['smoke_semantic_sha256']:
        raise ValueError('Smoke subset differs from the original nine turns')
    return small


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def bundle_check():
    expected = json.loads((BUNDLE / 'SHA256.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Bundle file changed: {name}')


def engine_check(checkout):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=checkout, text=True).strip()
    if git('rev-parse', 'HEAD') != COMMIT:
        raise RuntimeError('Use the exact clean MLSys-ver1.4 commit ' + COMMIT)
    if git('status', '--porcelain'):
        raise RuntimeError('Engine checkout must be clean; put output outside it')


def adapted_analysis():
    source = (BUNDLE / 'tools/original_analyze_render.py').read_text()
    edits = {
        'import argparse,gzip,json,sys': 'import argparse,gzip,json,sys,os',
        "HERE=Path(__file__).resolve().parents[1]": "HERE=Path(os.environ['FIGURE_REPORT_ROOT'])",
        "RAW=Path('/data/chokwans99/duet-v14-tree-figures-20261019')": "RAW=Path(os.environ['V14_REVIEW_RAW'])",
        "Path('/data/chokwans99/duet-v14-residual-review-20261019/runs')": "Path(os.environ['FIGURE_NATIVE_REFERENCE_ROOT'])",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError('Unexpected original analyzer layout')
        source = source.replace(old, new)
    return source


def adapted_check():
    source = (BUNDLE / 'tools/original_check_expected.py').read_text()
    source = source.replace("ROOT=Path('/data/chokwans99/duet-v14-tree-figures-20261019')",
                            "ROOT=Path(os.environ['V14_REVIEW_RAW'])")
    source = source.replace("'/home/chokwans99/PSD-MLSys-ver1.4/ssd'",
                            "str(Path(os.environ['PAPER_CHECKOUT'])/'ssd')")
    return source


def init(args):
    bundle_check()
    checkout = Path(args.checkout).resolve()
    engine_check(checkout)
    pairs = [[int(x) for x in p.split(',')] for p in args.pairs]
    flat = [x for pair in pairs for x in pair]
    if not 1 <= len(pairs) <= 3 or any(len(p) != 2 for p in pairs) or len(set(flat)) != len(flat):
        raise ValueError('Specify 1..3 disjoint pairs; target GPU first, draft GPU second')
    cpus = args.cpu_sets or [''] * len(pairs)
    if len(cpus) != len(pairs):
        raise ValueError('One --cpu-sets item per GPU pair')
    for model in (args.target, args.draft):
        if not (Path(model) / 'config.json').exists():
            raise ValueError('Use a local model snapshot directory containing config.json: ' + model)
    input_path = Path(args.inputs_full).resolve()
    inputs = json.loads(input_path.read_text())
    smoke_inputs = validate_inputs(inputs)
    work = Path(args.work).resolve()
    if work.is_relative_to(checkout):
        raise ValueError('Put output outside the clean engine checkout')
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('Init needs an empty output directory: ' + str(work))
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUNDLE / 'payload', work / 'raw')
    write(work / 'raw/INPUTS_FULL.json', inputs)
    write(work / 'raw/INPUTS_SMOKE.json', smoke_inputs)
    report = work / 'report'
    (report / 'tools').mkdir(parents=True)
    (report / 'tools/analyze_render.py').write_text(adapted_analysis())
    (report / 'tools/check_expected.py').write_text(adapted_check())
    config = dict(checkout=str(checkout), target=str(Path(args.target).resolve()),
                  draft=str(Path(args.draft).resolve()), pairs=pairs, cpu_sets=cpus,
                  cuda_arch=args.cuda_arch, work=str(work), seeds=list(SEEDS),
                  inputs_source=str(input_path), inputs_semantic_sha256=input_digest(inputs),
                  commit=COMMIT, warning='Quality/structure study; no TPS interpretation')
    write(work / 'CONFIG.json', config)
    (work / 'CONFIG.sha256').write_text(hashlib.sha256((work / 'CONFIG.json').read_bytes()).hexdigest() + '\n')
    print(json.dumps(config, indent=2))


def context(work):
    bundle_check()
    work = Path(work).resolve()
    config = json.loads((work / 'CONFIG.json').read_text())
    if hashlib.sha256((work / 'CONFIG.json').read_bytes()).hexdigest() != (work / 'CONFIG.sha256').read_text().strip():
        raise RuntimeError('Prepared server configuration changed; init a new output directory')
    engine_check(config['checkout'])
    raw = work / 'raw'
    for p in (BUNDLE / 'payload').rglob('*'):
        if p.is_file():
            rel = p.relative_to(BUNDLE / 'payload')
            if (raw / rel).read_bytes() != p.read_bytes():
                raise RuntimeError('Prepared payload changed: ' + str(rel))
    inputs = json.loads((raw / 'INPUTS_FULL.json').read_text())
    small = validate_inputs(inputs)
    if json.loads((raw / 'INPUTS_SMOKE.json').read_text()) != small:
        raise RuntimeError('Prepared smoke input changed')
    os.environ.update(V14_REVIEW_RAW=str(raw), VER15_CACHE_ROOT=str(work / 'cache'),
                      VER15_TARGET=config['target'], VER15_DRAFT=config['draft'],
                      PAPER_CHECKOUT=config['checkout'], SSD_CUDA_ARCH=config['cuda_arch'],
                      FIGURE_REPORT_ROOT=str(work / 'report'),
                      FIGURE_NATIVE_REFERENCE_ROOT=str(work / 'references'),
                      PYTHONDONTWRITEBYTECODE='1')
    return work, config


def check(work):
    work, config = context(work)
    inputs = json.loads((work / 'raw/INPUTS_FULL.json').read_text())
    if len(inputs) != 560 or len({str(x['question_id']) for x in inputs}) != 480:
        raise RuntimeError('Full input identity/count mismatch')
    subprocess.run([sys.executable, '-B', str(work / 'report/tools/check_expected.py')], check=True)
    print('CPU checks passed. No GPU inference has been launched.')


def load_runner(work, config):
    spec = importlib.util.spec_from_file_location('frozen_v7_runner', work / 'raw/run_once_helper.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PAIRS = config['pairs']
    module.CPUSETS = config['cpu_sets']
    return module


def job(work, config, seed, tag, observed, smoke=False):
    value = json.loads((work / 'raw/PLAN.json').read_text())['job']
    value.update(checkout=config['checkout'], tag=tag, shapes=observed,
                 figure_stride=1 if smoke else 8, smoke=smoke)
    value['config']['seed'] = seed
    if smoke:
        value['config'].update(prompts='INPUTS_SMOKE.json', output_tokens=96)
    return value


def launch(runner, value, slot):
    result = runner.run(value, slot)
    if result['status'] != 'complete' or not result.get('source_unchanged'):
        raise RuntimeError('Run not complete/valid; inspect its process.log and status.json: ' + repr(result))
    return result


def parity(work, native_tag, observed_tag, count):
    raw = work / 'raw/runs'
    a = json.loads((raw / native_tag / 'run.json').read_text())
    b = json.loads((raw / observed_tag / 'run.json').read_text())
    if any(x['paper_commit'] != COMMIT or x['paper_dirty'] or x['status'] != 'complete' for x in (a, b)):
        raise RuntimeError('Run provenance/status mismatch')
    ga = a['warm_groups'] + a['groups']; gb = b['warm_groups'] + b['groups']
    signature = lambda gs: [[(e['source'], e['accepted_len']) for s in g['steps'] for e in s['events']] for g in gs]
    if len(ga) != count or len(gb) != count or a['prompt_ids'] != b['prompt_ids']:
        raise RuntimeError('Native/observer input count/IDs mismatch')
    if [g['outputs'] for g in ga] != [g['outputs'] for g in gb] or signature(ga) != signature(gb):
        raise RuntimeError('Observer changed native output/source/AL on THIS server')
    write(work / 'report' / (observed_tag + '_PARITY.json'),
          dict(status='pass',native_tag=native_tag,observed_tag=observed_tag,turns=count,
               outputs_identical=True,all_source_AL_events_identical=True))


def smoke(work):
    check(work)
    work, config = context(work)
    runner = load_runner(work, config)
    launch(runner, job(work, config, 42, 'native_smoke', False, True), 0)
    launch(runner, job(work, config, 42, 'observed_smoke', True, True), 0)
    parity(work, 'native_smoke', 'observed_smoke', 9)
    import gzip
    import torch
    folder = work / 'raw/runs/observed_smoke'
    with gzip.open(folder / 'paired.jsonl.gz', 'rt') as f:
        records = [json.loads(line) for line in f]
    mc = [x['native_mc'] for row in records for x in row['outcomes'].values() if 'native_mc' in x]
    if len(mc) < 6 or any(x['status'] != 'pass' for x in mc):
        raise RuntimeError('Need both phases and all three real-model expectation checks')
    diag = folder / 'ROOT_PREFIX_DIAG.pt'
    audit = None
    if diag.exists():
        audit = torch.load(diag, weights_only=True, map_location='cpu')
        if not audit['prefix_glue_masks_equal'] or audit['native_width_replay_max_tv'] > .001:
            raise RuntimeError('Root prefix/native-width replay audit failed')
    write(work / 'report/SMOKE_CHECKS.json',
          dict(status='pass', native_residual_verifier=True, native_MC=mc,
               native_outputs_identical=9, prefix_diag_written=audit is not None))
    print('Smoke passed: native outputs/source/AL identical; full comparison may now run.')


def analyze(work):
    work, config = context(work)
    tags = []
    for seed in SEEDS:
        native, observed = f'native_full_seed{seed}', f'observed_full_seed{seed}'
        for tag in (native, observed):
            status = json.loads((work / 'raw/runs' / tag / 'status.json').read_text())
            if status['status'] != 'complete' or not status['source_unchanged']:
                raise RuntimeError('Incomplete/invalid run: ' + tag)
        parity(work, native, observed, 560)
        dst = work / 'references' / f'full_native_v14_s{seed}_v3'
        dst.mkdir(parents=True, exist_ok=True)
        link = dst / 'run.json'
        source = work / 'raw/runs' / native / 'run.json'
        if not link.exists():
            link.symlink_to(source)
        if link.resolve() != source:
            raise RuntimeError('Native reference points to a different run')
        tags.append(observed)
    subprocess.run([sys.executable, '-B', str(work / 'report/tools/analyze_render.py'), *tags], check=True)
    write(work / 'report/STATUS.json', dict(state='complete', seeds=list(SEEDS),
          native_parity_turns_per_seed=560, runtime_modified=False,
          result='paper/FIGURES_1_2_3.pdf', online_TPS_claim=False))


def full(work):
    work, config = context(work)
    if json.loads((work / 'report/SMOKE_CHECKS.json').read_text())['status'] != 'pass':
        raise RuntimeError('Run and pass smoke first')
    runner = load_runner(work, config)
    lock = threading.Lock(); completed = []
    def worker(slot):
        for seed in SEEDS[slot::len(config['pairs'])]:
            for observed, prefix in ((False, 'native'), (True, 'observed')):
                launch(runner, job(work, config, seed, f'{prefix}_full_seed{seed}', observed), slot)
            parity(work, f'native_full_seed{seed}', f'observed_full_seed{seed}', 560)
            with lock:
                completed.append(seed)
                write(work / 'report/STATUS.json', dict(state='full_running',completed=list(completed)))
    write(work / 'report/STATUS.json', dict(state='full_running',completed=[]))
    try:
        with ThreadPoolExecutor(max_workers=len(config['pairs'])) as pool:
            for future in as_completed([pool.submit(worker, i) for i in range(len(config['pairs']))]):
                future.result()
        analyze(work)
    except BaseException as exc:
        write(work / 'report/STATUS.json', dict(state='failed', completed=completed, error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('init')
    for name in ('checkout','target','draft','work','cuda-arch','inputs-full'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--pairs', nargs='+', required=True)
    p.add_argument('--cpu-sets', nargs='+')
    for name in ('check','smoke','full','analyze'):
        commands.add_parser(name).add_argument('--work',required=True)
    args = parser.parse_args()
    if args.command == 'init':init(args)
    else:globals()[args.command](args.work)


if __name__ == '__main__':
    main()
