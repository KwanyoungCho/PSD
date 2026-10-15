"""Source/measurement audit and incremental PS preparation report."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = Path('/data/chokwans99/duet-ps-device-20261015')


def read(path):
    return json.loads(path.read_text())


def candidate_file(rel):
    saved = OUT/'candidate_source'/rel
    return saved if saved.exists() else ROOT/rel


def main():
    checks = read(OUT/'checks/CHECKS.json')
    proof = read(OUT/'checks/CHECKS_PROVENANCE.json')
    assert proof['source_unchanged'] and len(checks) == 40
    assert all(r['returncode'] == 0 for r in checks)
    b = read(OUT/'BREAKDOWN.json')
    sources = {}
    runs = {}
    for path in sorted((RAW/'runs').glob('*/run.json')):
        r = read(path)
        assert r['status'] == 'complete', path
        role = path.parent.name.split('_')[0]
        if role in sources:
            assert sources[role] == r['source_hashes'], path
        sources[role] = r['source_hashes']
        runs[path.parent.name] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            commit=r['paper_commit'], dirty=r['paper_dirty'],
            kwargs=r['kwargs'], env=r['env'], args=r['args'],
            measurement_status=r['measurement_status'])
    assert sources['before'] == read(OUT/'BASELINE.json')['runtime_hashes']
    for rel, digest in sources['after'].items():
        assert hashlib.sha256(candidate_file(rel).read_bytes()).hexdigest() == digest, rel
        assert proof['source_hashes'][rel] == digest, rel
    changed = [k for k in sources['after'] if sources['after'][k] != sources['before'].get(k)]
    assert set(changed) == {'ssd/ssd/engine/draft_runner.py', 'ssd/ssd/engine/helpers/p2_prepare.py'}
    patch = ''
    for rel in changed:
        patch += ''.join(difflib.unified_diff(
            (RAW/'baseline'/rel).read_text().splitlines(keepends=True),
            candidate_file(rel).read_text().splitlines(keepends=True),
            fromfile='a/'+rel, tofile='b/'+rel))
    (OUT/'RUNTIME_INCREMENTAL.patch').write_text(patch)
    (OUT/'PROVENANCE.json').write_text(json.dumps(dict(
        runtime_changed=changed, sources=sources, runs=runs,
        baseline='Previous PS optimization, not clean 0916871',
        candidate_not_adopted=True, saved_source='candidate_source/',
        uncommitted=True), indent=2))
    u0 = read(RAW/'runs/before_unprofiled/run.json')
    u1 = read(RAW/'runs/after_unprofiled/run.json')
    assert u0['outputs'] == u1['outputs'] and u0['kwargs'] == u1['kwargs']
    assert u0['summary']['al'] == u1['summary']['al']
    assert u0['summary']['hit'] == u1['summary']['hit']
    u = dict(before=u0['summary'], after=u1['summary'], output_exact=len(u0['outputs']),
             preparation_before=u0['measurement']['preparation_all_decode'],
             preparation_after=u1['measurement']['preparation_all_decode'])
    assert u['preparation_before']['affected_steps'] == u['preparation_after']['affected_steps'] == 0
    (OUT/'UNPROFILED.json').write_text(json.dumps(u, indent=2))
    fallback = read(OUT/'FALLBACK.json')
    assert fallback['output_exact'] == 8
    assert ast.literal_eval(fallback['before_fallback_counters'][0]) == ast.literal_eval(fallback['after_fallback_counters'][0])
    ps = b['components']['ps_total']
    ci = ps['paired_question_bootstrap_delta_ci95']
    keys = ['build','dispatch','prepare1','prepare2']
    prep = {arm:sum(b['components'][k][arm+'_ms'] for k in keys) for arm in ('before','after')}
    names = dict(build='PS root/layout 구성', dispatch='Slot 계산·유효성 검사·dispatch 간격',
        prepare1='PS root 메타데이터 준비',prepare2='Graph 입력 버퍼 준비',
        graph='PS graph: forward + tree 선택',other='기타 간격·출력 변환·cache merge',
        ps_total='**PS 전체**',target_forward='Target forward',p1_graph='DS graph',prep_total='준비 네 구간 합 (중복 합산 금지)',ps_overhead='PS 전체에서 graph를 뺀 시간 (중복 합산 금지)')
    table = ['| 구간 | 직전 최적화 ms | 이번 수정 ms | 변화 |','|---|---:|---:|---:|']
    for k,v in b['components'].items():
        table.append(f'| {names[k]} | {v["before_ms"]:.3f} | {v["after_ms"]:.3f} | {v["change_pct"]:+.2f}% |')
    repeats = ['| 반복 | 실행 순서 | 이전 PS ms | 이후 PS ms |','|---|---|---:|---:|']
    for p in b['pairs']:
        v = p['breakdown_ms']['ps_total']
        repeats.append(f'| {p["rep"]} | '+('A→B' if p['rep']==1 else 'B→A')+f' | {v["before"]:.3f} | {v["after"]:.3f} |')
    source_table = ['| 현재 검증 source | step 쌍 | 이전 PS ms | 이후 PS ms | 차이 95% 구간 ms |','|---|---:|---:|---:|---:|']
    for k,name in [('0','miss'),('1','DS hit'),('2','PS hit')]:
        v = b['per_source'][k]; p = v['breakdown']['ps_total']; c = v['ps_delta_ci95']
        source_table.append(f'| {name} | {v["n"]} | {p["before"]:.3f} | {p["after"]:.3f} | {c[0]:+.3f}~{c[1]:+.3f} |')
    decision = '채택: PS 전체의 추가 감소를 확인했다.' if ci[1] < 0 and b['components']['ps_overhead']['paired_question_bootstrap_delta_ci95'][1] < 0 else '미채택: PS 전체와 graph 외 overhead가 함께 줄었다는 증거가 부족하다.'
    text = f'''# PS의 불필요한 CPU readback 제거 — 추가 최적화

2026-10-15. MLSys-ver1.1 기반의 추가 후보 실험. **이번 후보는 미채택하고 직전 PS 최적화 runtime을 유지한다.** 새 커밋·push 없음.

## 결론

**{decision}** 직전 최적화 버전과 직접 비교해 PS 전체 **{ps['before_ms']:.3f}→{ps['after_ms']:.3f}ms ({ps['change_pct']:+.2f}%)**.
준비 네 구간 합은 **{prep['before']:.3f}→{prep['after']:.3f}ms**다.
PS 전체 차이의 문항 단위 paired bootstrap 95% 구간은 **{ci[0]:+.3f}~{ci[1]:+.3f}ms**다.

기능 검증은 통과했지만, `phase2_build`의 감소가 PS 전체와 graph 외 overhead의 확실한 감소로 이어지지 않았다. 평균만 보고 이 후보를 성능 개선으로 추가하지 않았다. `candidate_source/`와 patch에 구현을 보존하고 현재 runtime은 baseline 소스와 일치하도록 복원했다. [복원 소스 검증](RESTORED.json)을 확인한다.

기준은 clean `0916871`이 아니라 [앞선 PS 준비 최적화](../ps_preparation_20261015/REPORT.md)까지 포함한 runtime이다.
두 단계 실험의 다른 실행에서 얻은 절대 시간을 이어 붙여 누적 개선율을 계산하지 않는다.

## 원인과 변경

1. B1 tree hit의 PS 준비에서 GPU fan-out을 `.tolist()`로 읽고 Python seed 위치 목록을 만들었다. 실제 captured executor는 이 목록의 **길이만** 사용했다. 이제 root 수는 이미 확정된 tensor shape에서 가져오고, 위치 매핑과 조상 mask는 GPU에 유지한다.
2. 같은 request의 문맥 길이는 DS 실행 준비에서 이미 CPU로 읽었다. PS에서 같은 tensor를 다시 `int(...)`로 읽던 부분을 없애고 DS에서 보관한 Python 정수를 전달한다.
3. Eager fallback이나 진단 도구가 실제 위치 목록을 필요로 하면 그 경로에서만 host fan-out과 seed 위치를 만든다. 매 step 목록을 `None`으로 초기화하므로 이전 요청의 값이 남지 않는다. 정상 captured 경로를 막는 새 사용자 옵션은 없다.

작은 데이터를 읽는 데에도 GPU의 선행 작업이 끝나기를 기다리는 비용이 포함될 수 있다. 따라서 제거한 데이터 크기만으로 절감 시간을 예측하지 않는다. 또한 `phase2_build`만 줄어도 뒤 dispatch로 대기가 이동할 수 있으므로 **PS 전체 완료 시간까지** 비교했다.

`metadata_ints.F`는 이 P2 tree 경로에서 사용하지 않아 0을 전달한다. Legacy chain 경로는 host fan-out과 원래 F를 유지한다. Graph 거절 후 arena fallback도 목록을 복원한다. 후보 점수, 순서, budget, tree 정책, graph 내부 수학 연산, RNG 소비, 여섯 metadata 유효성 검사는 변경하지 않았다.

## 직접 breakdown

'''+ '\n'.join(table)+'''

PS 전체는 proxy 수신 완료부터 cache merge 완료까지의 draft process 구간이다. Dispatch는 build 끝부터 첫 prepare 시작까지의 관찰 간격이며, CPU 단독 계산 시간으로 부르지 않는다. Graph는 forward뿐 아니라 tree 선택/sampling도 포함한다. 중첩 event를 이중 합산하지 않는다.

'''+ '\n'.join(repeats)+'\n\n'+ '\n'.join(source_table)+f'''

Source는 현재 verification 입력의 출처이고 PS 시간은 다음 cache를 만드는 시간이다. 이번 readback 제거는 **B1 tree-hit** 경로에 적용되며 miss와 B>1 전체 개선을 주장하지 않는다.

## 검증과 측정 조건

- 후보 runtime의 **{sum(r['tests'] for r in checks)} tests / {len(checks)} suites**, 실패·오류·skip 0. 검사 전후 runtime/bench/test hash 동일. [검사 결과](checks/CHECKS.json), [소스 증거](checks/CHECKS_PROVENANCE.json). 현재 유지한 runtime은 앞선 382-test 검증 소스와 정확히 일치하며, 이번 386개 검사는 보관된 후보 소스의 결과다.
- Profile: 동일 24턴 × AB/BA 두 쌍, 출력 **48/48 동일**, step source·AL·cache hit 동일. {b['matched_step_pairs']:,}개 matched step 쌍, 통계 기여 문항 {b['questions']}개. Full dataset의 AL 개선 평가가 아니다.
- Profile OFF: 별도 48턴 출력 **48/48 동일**, AL {u['after']['al']:.6f}, hit {100*u['after']['hit']:.4f}%로 전후 동일. Graph 준비 영향 step 양 arm 0. [보조 측정](UNPROFILED.json).
- Graph가 두 번에 한 번 거절되도록 강제해 실제 fallback **148회씩** 실행, 8턴 출력 **8/8 동일**. [fallback 증거](FALLBACK.json). 이 실행은 회귀 검사와 겹쳐 성능 근거로 사용하지 않는다.
- Llama2-7B + AMD135M, BF16, B1, T0.7, seed9101, RTX4090 GPU4/5. CPU16–23/48–55. Exit20, K1/K2=6/3, G=M=12/8, C3, DS 폭26, PS 폭15/root10, frontier. 파라미터 탐색 없이 고정했다.
- SpecBench raw turns 파일에서 균등 간격 선택, 입력 상한512/출력96/문맥2048, warmup4. 양 arm 동일 cold-prefix 및 graph prewarm all.
- 성능 arm은 전체 회귀 종료 후 순차 실행했다. GPU clock은 고정하지 않았으며 telemetry를 보존했다. 다른 GPU 작업의 시스템 자원 영향까지 완전히 통제한 실험은 아니다.

## 그림·재현·남은 비용

- [구간별 전후 PDF](BREAKDOWN_COMPARISON.pdf), [PNG](BREAKDOWN_COMPARISON.png), [수치·CI](BREAKDOWN.json).
- [동일 UID/step의 native timeline 전후](native/NATIVE_BEFORE_AFTER.pdf): DS hit, PS hit, miss 각각 한 쌍. Draft 점선은 forward 횟수 표시이며 실측 forward 경계가 아니다.
- [직전 최적화 대비 후보 runtime patch](RUNTIME_INCREMENTAL.patch), [실행별 provenance](PROVENANCE.json). 후보가 변경한 runtime은 `draft_runner.py`, `helpers/p2_prepare.py` 두 파일이다. 현재 runtime에 이 patch가 적용됐다는 뜻은 아니다.
- Raw는 `RAW_RESULTS.tar.gz`와 `{RAW}`에 보관한다. 재분석: `python {OUT.relative_to(ROOT)}/analyze.py --raw {RAW}/runs --output {OUT.relative_to(ROOT)} --reps 1 2`.
- 남은 metadata 유효성 검사 readback, graph 입력 버퍼 복사, slot 구성, cache merge 등은 남아 있다. CPU–GPU 동기화가 모두 제거됐다는 결과가 아니다. 이 결과의 수치를 70B/Blackwell/B>1에 일반화하지 않는다.
'''
    (OUT/'REPORT.md').write_text(text)
    print(decision, ps)


if __name__ == '__main__':
    main()
