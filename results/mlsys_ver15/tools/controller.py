"""Idle local GPU pairs, unlimited seeds, validation before advancing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['VER15_RAW'])
REPORT = Path(os.environ['VER15_REPORT'])
PYTHON = sys.executable
PREFIX = (42, 123, 2026)
KEYS = ('tps', 'al', 'hit_pct', 'share_pct', 'step_ms')


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def next_seed(completed):
    for seed in PREFIX:
        if seed not in completed:
            return seed
    seed = 0
    while seed in completed or seed in PREFIX:
        seed += 1
    return seed


def ordered(seeds):
    return sorted(seeds, key=lambda s: (0, PREFIX.index(s)) if s in PREFIX else (1, s))


def fmt(value):
    return '—' if value is None else f'{value:.3f}'


def render(queue):
    infos = {}
    for path in (REPORT/'validated').glob('seed*.json'):
        value = json.loads(path.read_text())
        assert value['checks']['status'] == 'complete'
        infos[value['info']['seed']] = value['info']
    infos = {s: infos[s] for s in ordered(infos)}
    totals = {}
    for source in ('1', '2', '0', 'all'):
        totals[source] = {}
        for key in KEYS:
            values = [v['sources'][source][key] for v in infos.values() if v['sources'][source][key] is not None]
            totals[source][key] = dict(mean=mean(values) if values else None,
                sample_std=stdev(values) if len(values) > 1 else None, n=len(values))
    rows = [dict(seed=s, state='complete_reused' if v.get('reused_from') else 'complete',
        gpus=v['gpus'], turns_lower_bound=560, validated=True) for s, v in infos.items()]
    for active in queue.get('active', {}).values():
        row = dict(seed=active['seed'], state='initializing', gpus=active['gpus'], turns_lower_bound=0, validated=False)
        log = Path(active.get('raw_root',str(ROOT)))/'runs'/f"seed{active['seed']}"/'process.log'
        if log.exists():
            matches = re.findall(r'PAPER_GROUP\s+\S+\s+measured_0\s+(\d+)', log.read_text(errors='replace'))
            row.update(state='running', turns_lower_bound=int(matches[-1])+1 if matches else 0)
        rows.append(row)
    compact = {}
    for seed, info in infos.items():
        compact[str(seed)] = {key: info[key] for key in ('seed','gpus','sources','exclusion_counts',
            'zero_decode_uids','context_guard_ended_uids','parameters','reporting_policy')}
        compact[str(seed)].update(valid_questions=info['question_mean']['questions'],
            valid_turns=info['question_mean']['turns'], validated_record=str(REPORT/'validated'/f'seed{seed}.json'),
            reused_from=info.get('reused_from'))
    dump(REPORT/'STATUS.json', dict(queue, rows=rows, complete_validated_seeds=list(infos)))
    dump(REPORT/'SUMMARY.json', dict(status='rolling_not_final', reporting_policy='turn_then_question_mean_existing_step_filter_v1',
        per_seed=compact, all_completed_seeds_equal_weight=totals,
        note='All completed seeds, equal question then equal seed weights; no best-seed selection. Full unfiltered question means and pooled diagnostics remain in validated/seedN.json.'))
    with (REPORT/'SUMMARY.csv').open('w', newline='') as stream:
        fields = ['seed','case',*KEYS,'metric_questions','metric_turns','long_steps_excluded','reused']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for seed, info in infos.items():
            for s in info['sources'].values():
                writer.writerow(dict(seed=seed,case=s['label'],**{k:s[k] for k in KEYS},
                    metric_questions=s['metric_questions']['al'],metric_turns=s['metric_turns']['al'],
                    long_steps_excluded=info['exclusion_counts']['long_step'],reused=bool(info.get('reused_from'))))
    lines = ['# MLSys-ver1.5 점수순 tree / target-sampling 연속 seed 평가', '',
        '사용자가 중단하라고 할 때까지 seed별 전체 데이터 평가를 이어갑니다. 완료 및 검증된 결과만 성능 평균에 포함합니다.', '',
        '- 모델: LayerSkip-Llama2-7B + JackFram/llama-68m, BF16, B=1, Target/Draft T=0.7, nucleus 절단 없음.',
        '- Exit21(0-based). P1: K1=10, 위치당 root4개, W1=56, M1=23, C=3. P2: K2=5, root24개, W2=28, M2=15.',
        '- 비동기 proxy 전송만 켬. Exit replica·별도 proxy stream·profile OFF. 동일한 전체 page prewarm, draft KV fraction 0.35.',
        '- 매 seed Spec-Bench480문항/560턴 전체 1회. 원본 입력과 순서 유지, 추가 template·truncation 없음. 문맥2048, 출력 min(1024,2048−입력), 자연 EOS·기존 문맥 보호 유지.',
        '- 첫3요청은 560턴 안에서 warmup으로 1회만 실행하고 성능 제외(대상557턴). cap/clip·부분 batch·같은 요청/batch/source-mix 내 2×중앙값 초과 step 제외 규칙 유지.',
        '- 기본 집계: 요청별 비율 → 문항 내 유효 턴 평균 → 같은 가중치 문항 평균 → 같은 가중치 seed 평균. 조건부 AL·latency는 해당 source가 발생한 유효 턴/문항만 포함.',
        '- v1.5는 최종 경로 점수 top-M + target 분포 직접 샘플링 검증입니다. v1.4 seed 결과를 재사용하지 않습니다. root 점수 수식은 기존 휴리스틱 그대로이며 새 검증에 최적화한 설정은 아닙니다.',
        '- 진행 순서: 42 → 123 → 2026 → 0 → 1 → 2 → …, 뒤에서 42·123·2026은 중복 실행하지 않습니다.',
        '- 로컬 (4,5)·(6,7)·(1,2)·(0,3) 중 비어 있는 쌍을 사용합니다. 외부 작업은 종료하지 않습니다. 디스크35GiB 미만은 새 seed를 보류합니다.', '',
        f"상태: **{queue['state']}** / 다음 seed: {queue['next_seed']} / 갱신: {queue['updated_local']}", '',
        '| Seed | 상태 | 저장 턴 하한 | GPU | 검증 |', '|---|---|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['state']} | {row['turns_lower_bound']}/560 | {row['gpus']} | {row['validated']} |")
    lines += ['', '| Seed | TPS | AL | 전체 hit (%) | Step (ms) | P1 rate (%) / AL | P2 rate (%) / AL | Miss rate (%) / AL | 유효 문항/턴 | 긴 step 제외 |',
        '|---|---:|---:|---:|---:|---|---|---|---|---:|']
    for seed, info in infos.items():
        ss=info['sources']; a=ss['all']; parts=[f"{fmt(ss[k]['share_pct'])} / {fmt(ss[k]['al'])}" for k in ('1','2','0')]
        lines.append(f"| {seed}{' (재사용)' if info.get('reused_from') else ''} | " + ' | '.join(fmt(a[k]) for k in ('tps','al','hit_pct','step_ms')) +
            ' | ' + ' | '.join(parts) + f" | {info['question_mean']['questions']}/{info['question_mean']['turns']} | {info['exclusion_counts']['long_step']} |")
    if infos:
        a=totals['all']
        def fmean(k):
            v=a[k];return fmt(v['mean']) + (f" ± {v['sample_std']:.3f}" if v['sample_std'] is not None else '')
        parts=[f"{fmt(totals[s]['share_pct']['mean'])} / {fmt(totals[s]['al']['mean'])}" for s in ('1','2','0')]
        lines.append(f'| 완료 {len(infos)} seed 평균 ± 표준편차 | '+ ' | '.join(fmean(k) for k in ('tps','al','hit_pct','step_ms'))+' | '+' | '.join(parts)+' | — | — |')
    lines += ['', '| Seed | P1 hit step (ms) | P2 hit step (ms) | Miss step (ms) | Cap/clip 제외 step | 전체 step |', '|---|---:|---:|---:|---:|---:|']
    for seed, info in infos.items():
        lines.append(f'| {seed} | '+' | '.join(fmt(info['sources'][s]['step_ms']) for s in ('1','2','0'))+
            f" | {info['exclusion_counts']['cap_or_clip']} | {info['exclusion_counts']['all_steps']} |")
    lines += ['', '무디코딩·문맥 보호·조건부 지표의 분모와 원시 집계는 각 validated/seedN.json에 보존합니다. 이전 파라미터의 seed는 이 표에 섞지 않습니다.', '',
        f'Raw: `{ROOT}`', '', f'진행 중 seed를 마친 뒤 신규 배정을 멈추려면 `touch {ROOT}/STOP`. 이후 자동 재개하지 않습니다. 직접 종료 요청을 받으면 소유 프로세스 확인 후 처리합니다.', '']
    (REPORT/'REPORT.md').write_text('\n'.join(lines))


def idle(snapshot, pair):
    uuids = {line.split(',')[1].strip() for line in snapshot['gpu']
             if int(line.split(',')[0]) in pair}
    return not any(line.split(',')[0].strip() in uuids
                   for line in snapshot['processes'] if line)


def main():
    os.environ['TZ']='Asia/Seoul';time.tzset()
    if socket.gethostname().split('.')[0]!='orion':raise RuntimeError('Local orion only')
    root_release=json.loads((ROOT/'RELEASE.json').read_text())
    assert root_release['validation_passed'] and root_release['pushed']
    checkout=Path(os.environ['VER15_CHECKOUT'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==root_release['commit']
    with (ROOT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import run_once_helper as runner
        plan=json.loads((ROOT/'PLAN.json').read_text())
        completed={int(p.stem[4:]) for p in (REPORT/'validated').glob('seed*.json')}
        queue=dict(state='starting',active={},completed=ordered(completed),
            gpu_pairs=plan['gpu_pairs'],pid=os.getpid(),until_user_stop=True,
            stop_file=str(ROOT/'STOP'),waiting_pairs=[],release_commit=root_release['commit'])
        futures={};failed=False
        def update(state):
            active={str(slot):dict(seed=v['seed'],gpus=plan['gpu_pairs'][slot],raw_root=str(ROOT))
                    for slot,v in futures.items()}
            queue.update(state=state,active=active,completed=ordered(completed),
                next_seed=next_seed(completed|{v['seed'] for v in futures.values()}),
                updated_local=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
            dump(ROOT/'QUEUE.json',queue);render(queue)
        update('running')
        with ThreadPoolExecutor(max_workers=len(plan['gpu_pairs'])) as pool:
            while True:
                for slot in list(futures):
                    v=futures[slot]
                    if not v['future'].done():continue
                    try:
                        result=v['future'].result()
                        if result['status']=='busy':del futures[slot];continue
                        if result['status']!='complete':raise RuntimeError(str(result))
                        tag=f"seed{v['seed']}"
                        with (ROOT/'runs'/tag/'validation.log').open('w') as log:
                            subprocess.run([PYTHON,'-B',str(Path(__file__).with_name('validate.py')),tag],
                                stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ))
                        checked=json.loads((REPORT/'validated'/f'{tag}.json').read_text())
                        assert checked['checks']['status']=='complete'
                        completed.add(v['seed'])
                    except BaseException:
                        failed=True;queue['error']=traceback.format_exc()
                        (ROOT/'FAILURE.txt').write_text(queue['error'])
                        (ROOT/'STOP').write_text('Validation/execution failed; no new claims.\n')
                    del futures[slot]
                stop=(ROOT/'STOP').exists()
                if stop and not futures:update('failed' if failed else 'stopped_by_user');break
                queue['waiting_pairs']=[]
                disk_ok=shutil.disk_usage(ROOT).free>=35*1024**3
                if not stop and disk_ok:
                    snapshot=runner.state()
                    for slot,pair in enumerate(plan['gpu_pairs']):
                        if slot in futures:continue
                        if not idle(snapshot,pair):
                            queue['waiting_pairs'].append(dict(gpus=pair,reason='external_gpu_process'));continue
                        seed=next_seed(completed|{v['seed'] for v in futures.values()})
                        job=copy.deepcopy(plan['job_template']);job['tag']=f'seed{seed}';job['config']['seed']=seed
                        futures[slot]=dict(seed=seed,future=pool.submit(runner.run,job,slot))
                update('draining' if stop else 'waiting_for_disk' if not disk_ok else 'running')
                time.sleep(15)


if __name__=='__main__':main()
