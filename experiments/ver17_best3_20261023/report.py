"""Report the six completed full runs with explicit question populations."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from question_mean import summarize, KEYS

CODE=Path(__file__).resolve().parent
OUT=CODE.parents[1]/'results/mlsys_ver17/best3_20261023'
SEEDS=(2,0,10)
ARMS=('before','after')


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def fmt(value,digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def avg(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows):
    return {source:{key:dict(mean=avg(row['sources'][source][key] for row in rows),
            sd=statistics.stdev([row['sources'][source][key] for row in rows])
               if len([row for row in rows if row['sources'][source][key] is not None])>1 else None)
            for key in KEYS} for source in ('1','2','0','all')}


def comparison_table(aggregates):
    fields=[('1','share_pct'),('1','al'),('2','share_pct'),('2','al'),
            ('all','hit_pct'),('all','al'),('all','step_ms'),('all','tps')]
    lines=['| 버전 / 표본 | P1 % | P1 AL | P2 % | P2 AL | Hit % | AL | Step ms | TPS |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,agg in aggregates:
        cells=[]
        for source,key in fields:
            item=agg[source][key];digits=3 if key=='al' else 2
            cells.append(fmt(item['mean'],digits)+' ± '+fmt(item['sd'],digits))
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    return lines


def wall_summary(run, rows, common):
    """Same selected steps/questions, using the independent outer CPU clock."""
    offset=0;by_question=defaultdict(list);checked=0
    for index,group in enumerate(run['warm_groups']+run['groups']):
        steps=group['steps'];ww=rows[offset:offset+len(steps)];offset+=len(steps)
        assert len(ww)==len(steps)
        assert all(s['seconds']==w['native_seconds'] for s,w in zip(steps,ww))
        checked+=len(steps)
        rec=group['request_records'][0];qid=str(rec['question_id'])
        if index<len(run['warm_groups']) or qid not in common:
            continue
        selected=[w for s,w in zip(steps,ww) if s['performance_included']]
        if rec['completion_tokens']<2 or rec['decode_total_time']<=0 or not selected:
            continue
        by_question[qid].append(statistics.mean(w['wall_seconds'] for w in selected)*1000)
    assert offset==len(rows)
    assert set(by_question)==common
    return dict(step_ms=statistics.mean(statistics.mean(v) for v in by_question.values()),
                checked_steps=checked,questions=len(by_question),
                definition='same filtered complete common questions, independent CPU wall around LLMEngine.step')


def metric_table(summaries):
    lines=['| Seed | 버전 | P1 발생률 % | P1 AL | P2 발생률 % | P2 AL | Miss % | Miss AL | 전체 hit % | 전체 AL | Step ms | TPS |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for arm in ARMS:
            r=summaries[f'{arm}_s{seed}']['sources'];a=r['all']
            values=[r['1']['share_pct'],r['1']['al'],r['2']['share_pct'],r['2']['al'],
                    r['0']['share_pct'],r['0']['al'],a['hit_pct'],a['al'],a['step_ms'],a['tps']]
            lines.append('| '+str(seed)+' | '+('v1.6' if arm=='before' else 'v1.7')+' | '+
                         ' | '.join(fmt(v,3 if i in (1,3,5,7) else 2) for i,v in enumerate(values))+' |')
    return lines


def final_report():
    state=json.loads((OUT/'STATUS.json').read_text())
    tags=[f'{arm}_s{seed}' for seed in SEEDS for arm in ARMS]
    assert all(state['done'].get(t,{}).get('status')=='complete' for t in tags)
    inputs=json.loads((CODE/'INPUTS_FULL.json').read_text())
    required=defaultdict(set)
    for r in inputs[3:]:required[str(r['question_id'])].add(r['uid'])
    native={};checks={};eligible=[]
    for tag in tags:
        folder=OUT/'runs'/tag
        native[tag]=json.loads((folder/'QUESTION_MEAN.json').read_text())
        checks[tag]=json.loads((folder/'CHECKS.json').read_text())
        valid=set(native[tag]['valid_turn_uids'])
        eligible.append({qid for qid,uids in required.items() if uids<=valid})
    common=set.intersection(*eligible)
    assert common
    matched={};raw={};available_raw={};outer_wall={};measurement={}
    for tag in tags:
        run=json.loads((OUT/'runs'/tag/'run.json').read_text())
        outer_wall[tag]=wall_summary(run,json.loads((OUT/'runs'/tag/'WALL_STEPS.json').read_text()),common)
        measurement[tag]={key:run['measurement'][key] for key in
                          ('policy','included_steps','excluded_steps','exclusion_reasons')}
        selected=[g for g in run['groups'] if str(g['request_records'][0]['question_id']) in common]
        matched[tag]=summarize(selected)
        raw[tag]=summarize(selected,filtered=False)
        available_raw[tag]=json.loads((OUT/'runs'/tag/'QUESTION_MEAN_RAW.json').read_text())
        assert set(matched[tag]['by_question'])==common
        assert len(matched[tag]['valid_turn_uids'])==sum(len(required[q]) for q in common)
        del run
    aggs={arm:aggregate([matched[f'{arm}_s{s}'] for s in SEEDS]) for arm in ARMS}
    deltas={source:{key:aggs['after'][source][key]['mean']-aggs['before'][source][key]['mean']
                   for key in KEYS} for source in ('1','2','0','all')}
    historical=json.loads((OUT/'HISTORICAL_METRICS.json').read_text())
    raw_aggs={arm:aggregate([available_raw[f'{arm}_s{s}'] for s in SEEDS]) for arm in ARMS}
    common_raw_aggs={arm:aggregate([raw[f'{arm}_s{s}'] for s in SEEDS]) for arm in ARMS}
    paired={str(seed):{source:{key:matched[f'after_s{seed}']['sources'][source][key]-
             matched[f'before_s{seed}']['sources'][source][key] for key in KEYS}
             for source in ('1','2','0','all')} for seed in SEEDS}
    result=dict(status='complete',selection='historical best-3 seeds 2,0,10 fixed before this campaign; no reselection',
        common_complete_question_ids=sorted(common,key=int),common_questions=len(common),
        common_turns=sum(len(required[q]) for q in common),matched=matched,
        all_valid=native,common_unfiltered=raw,available_unfiltered=available_raw,
        available_unfiltered_aggregate=raw_aggs,common_unfiltered_aggregate=common_raw_aggs,
        historical=historical,aggregate=aggs,after_minus_before=deltas,paired_seed_differences=paired,
        outer_wall=outer_wall,measurement=measurement,checks=checks)
    dump(OUT/'RESULTS.json',result)
    compact=dict(result)
    for population in ('matched','all_valid','common_unfiltered','available_unfiltered'):
        compact[population]={tag:{k:v for k,v in row.items() if k!='by_question'}
                             for tag,row in result[population].items()}
    compact['detail_note']='Per-question detail is retained locally in RESULTS.json; source hashes and all metric denominators are included here.'
    dump(OUT/'SUMMARY.json',compact)
    csvrows=[]
    for population,ss in [('common_filtered',matched),('available_filtered',native),
                          ('common_unfiltered',raw),('available_unfiltered',available_raw)]:
        for tag,r in ss.items():
            for source in ('1','2','0','all'):
                v=r['sources'][source]
                csvrows.append(dict(population=population,run=tag,source=source,
                    questions=r['questions'],turns=r['turns'],**{k:v[k] for k in KEYS}))
    with (OUT/'METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
    lines=['# DUET 7B: v1.6 → v1.7 · 동일 best-3 비교','',
        '완료: 6/6 실행, 각 실행 전체 Spec-Bench 480문항·560턴. 기존 논문 표에서 선택했던 best-3 seed 2·0·10을 그대로 고정했다. 이번 결과에서 seed를 다시 고르지 않았다.',
        '',f'주 비교는 여섯 실행에서 모든 측정 턴이 유효한 **공통 {len(common)}문항·{result["common_turns"]}턴**이다. 실행별 유효 표본 전체를 사용한 결과와 필터 없는 민감도도 아래/SUMMARY.json에 별도로 보존했다.',
        '', 'BF16, v1.6 `4c7e4338` → v1.7 `0654bd68`, B1, target TP1 + draft TP1. K1=10/K2=5, Exit21, P1 W56·위치당4·G=M23, P2 R24·W28·G=M15, C3는 고정했다. P1/P2 모두 동적 tree on, 확장은 last다. async proxy send on, replica/proxy side stream off, graph logits on, profile off.',
        '', '입력 raw prompt·원본 token ID·순서를 유지하고 문맥2048/출력1024, target·draft T0.7, top-p1, 자연 EOS와 기존 문맥 보호를 적용했다. 첫3턴은 560턴 안에 포함된 warmup이며 성능에서 제외했다.',
        '', '## 같은 입력 문항에서의 전후 비교','']
    lines+=metric_table(matched)
    lines+=['','| 버전 | P1 발생률 % | P1 AL | P2 발생률 % | P2 AL | 전체 hit % | 전체 AL | Step ms | TPS |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields=[('1','share_pct'),('1','al'),('2','share_pct'),('2','al'),('all','hit_pct'),('all','al'),('all','step_ms'),('all','tps')]
    for arm in ARMS:
        cells=[]
        for source,key in fields:
            cell=aggs[arm][source][key]
            digits=3 if key=='al' else 2
            cells.append(fmt(cell['mean'],digits)+' ± '+fmt(cell['sd'],digits))
        lines.append('| '+('v1.6' if arm=='before' else 'v1.7')+' | '+' | '.join(cells)+' |')
    a=aggs['after']['all'];b=aggs['before']['all']
    step_pct=100*(a['step_ms']['mean']/b['step_ms']['mean']-1)
    tps_pct=100*(a['tps']['mean']/b['tps']['mean']-1)
    lines+=['',f'동일 공통 표본의 3-seed 평균 변화: **step {step_pct:+.2f}%**, **TPS {tps_pct:+.2f}%**, AL {deltas["all"]["al"]:+.3f}, hit {deltas["all"]["hit_pct"]:+.2f}%p. ±는 seed 간 표본 표준편차다.',
        '', '## 경로별 step latency','',
        '| Seed | 버전 | P1 ms | P2 ms | Miss ms | 전체 ms |',
        '|---|---|---:|---:|---:|---:|']
    for seed in SEEDS:
        for arm in ARMS:
            r=matched[f'{arm}_s{seed}']['sources']
            lines.append('| '+str(seed)+' | '+('v1.6' if arm=='before' else 'v1.7')+' | '+
                         ' | '.join(fmt(r[s]['step_ms']) for s in ('1','2','0','all'))+' |')
    lines+=['','전체 `LLMEngine.step()` 호출을 별도 CPU 시계로 측정한 값도 같은 공통 문항·같은 제외 step으로 계산했다. GPU profile이나 추가 synchronize는 사용하지 않았다.','',
            '| 버전 | 내부 step ms | 전체 호출 wall ms | 차이 ms |',
            '|---|---:|---:|---:|']
    for arm in ARMS:
        native_ms=aggs[arm]['all']['step_ms']['mean']
        wall_ms=statistics.mean(outer_wall[f'{arm}_s{s}']['step_ms'] for s in SEEDS)
        lines.append(f'| {arm} | {native_ms:.4f} | {wall_ms:.4f} | {wall_ms-native_ms:.4f} |')
    lines+=['', '## 실행별 유효 전체 표본','']+metric_table(native)
    lines+=['','| 실행 | 유효 문항 | 유효 턴 | 문맥 보호로 무생성 | 긴 step 제외 | 전체 raw steps |',
            '|---|---:|---:|---:|---:|---:|']
    for tag in tags:
        c=checks[tag];n=native[tag]
        lines.append(f'| {tag} | {n["questions"]} | {n["turns"]} | {len(c["context_guard_no_decode_uids"])} | {c["long_excluded"]} | {c["raw_steps"]} |')
    lines+=['','## 필터 없는 값과 과거 논문 표','',
        '아래 표는 첫3턴을 제외한 뒤 긴 step 및 cap/clip step을 제거하지 않는 기존 논문 표 방식이다. 출력2토큰 미만 등 미정의 턴은 제외하고 실행별 유효 문항 전체를 사용한다. 위의 공통 표본·필터 적용 수치와 직접 섞지 않는다.',
        '과거 v1.4의 best-3 선택 및 실행 시점은 이미 정해져 있었고, 해당 원시 입력 ID와 seed를 이번에도 유지했다. 과거와 현재 사이에는 graph 구현·측정 시점 등 차이도 있으므로 v1.4와의 차이를 이번 오류 수정만의 효과라고 해석하지 않는다. 수정 효과의 대조군은 이번에 같이 실행한 v1.6이다.','']
    lines+=comparison_table([('과거 v1.4 · 원 논문 표',historical['aggregate']),
                              ('v1.6 · 이번 대조군',raw_aggs['before']),
                              ('v1.7 · 이번 수정판',raw_aggs['after'])])
    lines+=['','같은 공통 문항에 대한 필터 없는 민감도:','']
    lines+=comparison_table([('v1.6 · 공통 문항',common_raw_aggs['before']),
                              ('v1.7 · 공통 문항',common_raw_aggs['after'])])
    lines+=['','## 집계와 검증 범위','',
        '- 턴 내 지표 비율 → 문항 내 유효 턴 평균 → 문항 평균 → 3-seed 동일 가중 평균이다. P1/P2 발생률의 분모는 전체 step이고, P1/P2 AL·latency는 해당 경로가 실제 발생한 표본의 조건부 평균이다. 경로별 유효 문항/턴 수는 SUMMARY.json에 있다.',
        '- 기존 cap/clip 제외와 요청별·실제 batch/source 구성별 중앙값 2배를 넘는 step 제외를 시간·토큰·AL·hit에 함께 적용했다. 제외 개수와 원시는 보존했다. profile을 켜거나 구간 시간을 임의로 빼지 않았다.',
        '- 출력2토큰 미만/무생성 턴의 TPS 등 미정의 지표를 0으로 대체하지 않았다. 기존 문항 평균과 같은 유효성 규칙을 적용했다. 문맥 보호로 무생성인 UID, 제외 UID와 이유, 필터 없는 지표는 JSON에 있다.',
        '- P1 직접 자식 제외(B1/B>1), phase/context/lane RNG 분리, proxy 점수 wire와 zero padding, B>1 guard 큐 복원을 함께 적용했다. 검증 수식과 draft q는 유지했다. 개별 수정의 효과를 분리한 실험은 아니다.',
        '- 기존 best-3는 과거 결과에서 상위 성능으로 선택된 seed다. 이번에는 요청대로 같은 세 값을 유지했지만 RNG 스트림 분리로 실제 난수열은 바뀐다. 이 세 결과만으로 모든 seed의 평균 효과를 확정하지 않으며, 유리한 새 seed를 다시 선택하지 않았다.',
        '- 세 seed는 서로 다른 PIX 쌍 [1,2]·[4,5]·[6,7]에서 실행했으며 같은 seed의 전후는 같은 쌍이다. seed0은 변경군을 먼저 실행해 순서 편향을 일부 통제했다. 관측 분산에는 stochastic 출력·GPU 쌍·시간 변동이 남는다.',
        '- 원시 입력/출력·source와 hit의 일관성·실제 parameter/KV/HF dtype·graph 실행·양 phase tree on·런타임/측정 helper SHA 불변·선점 없음·프로세스 종료0을 실행별 CHECKS.json으로 확인했다.',
        '', '[실험 조건](PROTOCOL.json) · [비교 요약/분모](SUMMARY.json) · [CSV](METRICS.csv)',
        '문항별 상세는 로컬 RESULTS.json, 원시 실행은 runs/ 아래에 보존한다. 원격에는 요약과 검증 근거를 포함한다.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def progress_report():
    state=json.loads((OUT/'STATUS.json').read_text())
    if state['status'] in ('complete','measurements_complete'):
        return
    lines=['# DUET 7B v1.6 → v1.7 · 진행 상태','',
        f'갱신: {datetime.now().isoformat(timespec="seconds")} · 상태: {state["status"]}',
        '', '전체 Spec-Bench 480문항·560턴/seed, 출력1024, BF16, profile off. 양 phase tree on. 최종 비교는 기존 best-3 seed2·0·10의 같은 입력 문항 평균으로 계산한다.',
        '', '| 실행 | 상태 | GPU | 최근 저장/진행 |', '|---|---|---|---|']
    for seed in SEEDS:
        for arm in ARMS:
            tag=f'{arm}_s{seed}';row=state['done'].get(tag) or state['active'].get(tag,{})
            status=row.get('status','queued')
            path=OUT/'runs'/tag/'process.log';last=''
            if path.exists():
                with path.open('rb') as f:
                    rr=[]
                    for line in f:
                        if line.startswith((b'PAPER_GROUP',b'PAPER_COMPLETE')):
                            rr=[line.decode(errors='replace').strip()]
                last=rr[-1][:90] if rr else '모델/graph 준비 중'
            lines.append(f'| {tag} | {status} | {row.get("gpus",[])} | {last} |')
    lines+=['','완료·검증된 여섯 실행이 모두 모이면 전후 P1/P2 발생률·AL·hit·step·TPS와 제외 개수 표가 자동 생성된다. 현재 부분 결과를 전체 성능으로 해석하지 않는다.',
            '', '[조건](PROTOCOL.json) · [진행 JSON](STATUS.json)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--progress',action='store_true');parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        while True:
            state=json.loads((OUT/'STATUS.json').read_text())
            if state['status'] in ('complete','measurements_complete'):break
            progress_report()
            if state['status'] in ('failed','stopped'):break
            time.sleep(30)
    elif args.progress:progress_report()
    else:final_report()
