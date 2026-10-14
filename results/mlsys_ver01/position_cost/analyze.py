"""Aggregate fixed-work evidence and append it to the single review report."""
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess

HERE=Path(__file__).parent
ROOT=HERE.parents[2]


def stat(xs):
    return dict(median=st.median(xs),min=min(xs),max=max(xs),runs=xs)


def main():
    compute=[json.loads((HERE/f'compute_r{i}.json').read_text()) for i in (1,2,3)]
    wires=[json.loads((HERE/f'transfer_ack_r{i}.json').read_text()) for i in (1,2,3)]
    checks=json.loads((HERE/'CHECKS.json').read_text())
    assert checks['status']=='PASS'
    assert len({x['source_sha256'] for x in compute})==1
    assert len({x['source_sha256'] for x in wires})==1
    result=dict(scope='Fixed tensor/communication microbench, not full model or TPS comparison.',compute=[],wire=[],checks=checks,
        wire_protocol='rank1 draft sends payload; rank0 target returns 8-byte ACK. Report complete round-trip transaction, never call it one-way.',
        source_files={},paper_baseline='a82f7d24fb36827a9a81a3567f344dccb71f193e')
    for j,c in enumerate(compute[0]['cells']):
        out={k:v for k,v in c.items() if k!='timing'}
        out['timing_us']={k:stat([x['cells'][j]['timing'][k]['median_us'] for x in compute]) for k in c['timing']}
        pairs=[('tree_ladder_mix','tree_ladder_hat')] if 'kind' in c else [
            ('position_dense_mix','position_dense_hat'),('position_fused_mix','position_dense_hat'),
            ('proxy_mix','proxy_hat'),('complement_mix','complement_hat'),('residual_mix','residual_hat'),
            ('proxy_hat','proxy_hat_scalar_q')]
        out['paired_delta_us']={a+' minus '+b:stat([x['cells'][j]['timing'][a]['median_us']-x['cells'][j]['timing'][b]['median_us'] for x in compute]) for a,b in pairs}
        result['compute'].append(out)
    for j in range(0,len(wires[0]['ranks'][1]['cells']),4):
        c=wires[0]['ranks'][1]['cells'][j]
        out={k:c[k] for k in ('vocab','batch','positions')}
        out['payloads']={}
        for off in range(4):
            row=wires[0]['ranks'][1]['cells'][j+off]
            out['payloads'][row['payload']]=dict(bytes=row['bytes'],gpu_us=stat([x['ranks'][1]['cells'][j+off]['timing']['gpu_us']['median'] for x in wires]),wall_us=stat([x['ranks'][1]['cells'][j+off]['timing']['wall_us']['median'] for x in wires]))
        out['full_minus_scalar_us']=stat([x['ranks'][1]['cells'][j]['timing']['gpu_us']['median']-x['ranks'][1]['cells'][j+1]['timing']['gpu_us']['median'] for x in wires])
        result['wire'].append(out)
    # Preserve the original send/recv measurement separately from ACK timing.
    one_way=[json.loads((HERE/f'transfer_r{i}.json').read_text()) for i in (1,2)]
    result['one_way_receive']=[]
    for j in range(0,len(one_way[0]['ranks'][0]['cells']),4):
        c=one_way[0]['ranks'][0]['cells'][j]
        out={k:c[k] for k in ('vocab','batch','positions')}
        out['full_us']=stat([x['ranks'][0]['cells'][j]['timing']['gpu_us']['median'] for x in one_way])
        out['scalar_us']=stat([x['ranks'][0]['cells'][j+1]['timing']['gpu_us']['median'] for x in one_way])
        out['full_minus_scalar_us']=stat([x['ranks'][0]['cells'][j]['timing']['gpu_us']['median']-x['ranks'][0]['cells'][j+1]['timing']['gpu_us']['median'] for x in one_way])
        result['one_way_receive'].append(out)
    runtime=json.loads((HERE.parent/'extensions/FINAL_RUNTIME_HASHES.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in runtime.items())
    result['runtime_unchanged_files']=len(runtime)
    result['runtime_hash_manifest']=str((HERE.parent/'extensions/FINAL_RUNTIME_HASHES.json').relative_to(ROOT))
    result['runtime_hash_manifest_sha256']=hashlib.sha256((ROOT/result['runtime_hash_manifest']).read_bytes()).hexdigest()
    for p in [*HERE.glob('*.py'),ROOT/'ssd/ssd/engine/helpers/batch_tree_sampling.py',ROOT/'ssd/ssd/engine/helpers/p2_tree.py']:
        result['source_files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')

    parts=[r'''
<a id="position-weight-cost-review"></a>

#### 2026-10-13 위치 수식의 추가 비용: scalar 확률과 전체 분포 전송

**결론:** Chain의 관측 위치 가중치 `h_hat`만 보면 위치별 `e(y), q(y)`면 충분하다. 그러나 논문 원본은 전체 `q`를 residual 후보 선정과 확률적 검증의 correction에도 사용한다. `h_bar`를 빼는 것만으로 해당 wire를 scalar로 교체할 수 없다. 현재처럼 `e,q`가 target에 있는 구조에서는 `h_bar`의 추가 전송량/메시지는 **0**이고, 추가 비용은 overlap reduction과 위치분포 혼합이다. 계산이 늘어 proxy 전송 시작이 늦어질 수는 있다.

이번에는 inference runtime을 변경하지 않고 component benchmark와 수치 검사를 수행했다. Graph 준비·warmup·메모리 할당은 측정에서 제외했다. 새로운 알고리즘을 production에 채택하거나 전체 모델/TPS를 재측정한 결과는 아니다.

##### A. 수식과 필요한 정보

\[
\widehat\alpha_i=\min(1,e_i(y_i)/q_i(y_i)),\quad
\widehat h_i=\left(\prod_{j<i}\widehat\alpha_j\right)(1-\widehat\alpha_i).
\]

마지막 bonus 위치는 모든 수락률의 곱이다. 위치 가중치만 계산하면 전체 vocabulary 확률을 상대 GPU로 보낼 필요가 없다. Draft가 `q(y)`를 보내고 target이 자신의 `e(y)`를 사용하면 된다. 그러나 한 확률의 계산이 곧 한 logit의 계산은 아니다.

\[
e(y)=\exp\{z_y/T-\operatorname{logsumexp}_v(z_v/T)\}.
\]

정확한 softmax 분모는 전체 vocabulary를 읽어야 한다. 이미 sampler/head에서 계산한 확률 또는 정규화 상수가 있으면 재사용할 수 있다. 단순히 LM head의 한 행만 계산하고 이를 확률로 사용할 수는 없다.

\[
\bar\alpha_i=\sum_v\min(e_i(v),q_i(v)),\quad
\widetilde h=.75\widehat h+.25\bar h.
\]

이 overlap에는 전체 분포의 정보가 필요하다. `e=(.2,.5,.3)`, `q_A=(.4,.5,.1)`, `q_B=(.4,.1,.5)`, `y=0`이면 `q_A(y)=q_B(y)=.4`, 두 관측 수락률은 .5이지만 평균 수락률은 .8/.6이다. 같은 scalar만으로 두 값을 구분할 수 없다.

또한 p=e라고 두면 실제 correction은 q_A에서 token2에 질량1, q_B에서 token1에 질량1이다. 즉, 수락 판단에 충분한 scalar가 correction 분포에도 충분한 것은 아니다. 이 반례는 `CHECKS.json`에 수치 검증을 보존했다.

##### B. 논문 원본에서 실제 무엇을 보내는가?

기준은 `feat/duet-p2tree-g0@a82f7d2`의 변경 없는 `/home/chokwans99/PSD-paper-review`다.

| 방향/사용처 | 논문 원본의 동작 | scalar 교체 가능성 |
|---|---|---|
| Draft→target | Chain의 전체 q logits 또는 tree의 parent-q logits | 다른 소비자까지 함께 바꿔야 함 |
| Target→draft, 일반 Policy B | 선택된 위치와 token, tree이면 점수를 함께 pack | 이미 작은 후보 wire이며 전체 e 전송이 아님 |
| Chain h_hat | e(y),q(y)의 비율과 누적곱 | 위치 계산만이면 가능 |
| Residual / e(1-q) root 점수 | 후보 vocabulary에 대한 q(v) | 현재 전체 점수 경로는 scalar 하나로 대체 불가 |
| Proxy-only root + h_hat | e에서 root를 선택, q는 위치 비율에만 사용 | Chain 후보 계산만이면 가능; 검증은 별도 |
| T>0 chain 검증 | q(y)로 수락, 거절 시 `[p-q]+`에서 correction | 전체 q를 없애려면 on-demand 전달 등 프로토콜 변경 필요 |
| Tree의 후속 sibling | 앞 sibling 거절 후 residual 정규화 | 단순 chain scalar 공식을 그대로 적용할 수 없음 |

근거: 원본 `engine/speculator_async.py:335–387`의 response 수신, `engine/verifier.py:983–1024`의 후보 계산·소형 wire 전송, `utils/verify.py:151–178`의 correction, `engine/helpers/p2_tree.py:2401–2425`의 sibling walk. `duet_proxy_on_draft`는 별도 실험 경로이며 B1 제한의 top-M+정규화 상수를 보내는 구성이다. 일반 논문 경로와 섞지 않는다.

T=0의 target argmax 검증에는 q 전체가 필요 없지만, root 정책이 residual/e(1-q)/h_bar라면 그 용도로 여전히 필요할 수 있다. 이 진단은 T>0 중심이며 greedy wire 최적화를 구현한 것은 아니다.

##### C. 실험 방법과 한계

- GPU 6 RTX4090에서 계산, 같은 PCIe switch의 GPU4/5에서 NCCL 전송을 측정했다. 두 GPU의 CUDA P2P 접근은 false였다. CPU affinity는 계산20–23/52–55, 통신16–19/48–51이다. 다른 GPU의 작업은 변경하지 않았다.
- Llama2 계열 V=32000/FP16, Llama3 계열 V=128256/BF16 형태의 dense synthetic logits다. 모델 weight를 로드하거나 prompt 생성 실험을 하지 않았다. 같은 확률 수식/실행 크기의 비용을 분리하려는 실험이다.
- B=1/8, 분포 행 수 K=4/12, T=.7, B1/K4는 T=1도 확인했다. **여기서 K는 점수·전송 대상 위치 수이며 K1/K2 forward round 수와 같다는 뜻이 아니다.**
- 계산은 3개 독립 process 실행, 실행마다 CUDA graph 준비 후 9회 순서 교차 측정×40 replay다. 수치는 실행별 median의 median이며 괄호 범위는 실행별 차이의 min/max다. 통계적 유의성 검정이나 전체 실행 향상률로 표현하지 않는다.
- Chain root 재현은 top17→global15, top-M 정규화, source=e/residual/e(1-q)다. h_hat와 혼합은 수식 정책 자체가 달라 root·AL·hit 동등성을 주장하지 않는다. scalar h_hat 대 dense h_hat의 값과 proxy-only root는 같은 입력에서 일치했다.
- 통신은 FP16 전체 q logits 대 FP32 scalar q(y)를 같은 `dist.send/recv`로 비교한다. 토큰/metadata는 두 방식 공통이라 제외했다. 최초 단방향 검사에는 수신 준비·CPU barrier 시점 잡음이 있어 `transfer_r1/r2.json`을 보조 기록으로 보존했다. 주 표는 payload→8-byte ACK의 **전체 왕복 transaction 시간**이며 단방향 전송 시간으로 부르지 않는다. 실행당 payload별80회×3개 독립 process 실행이다.
- CUDA graph 준비는 모두 측정 전 완료했다. 통신은 논문처럼 graph 밖에서 실행했다. 작은 payload에서는 호출·동기화 지연이 지배해 byte 감소가 latency 감소로 이어지지 않을 수 있다. 여기서 NCCL 시간을 통째로 `h_bar` 비용으로 더하면 안 된다.

##### D. e,q가 이미 있을 때 위치 가중치만의 비용

단위 ms. Fused는 중간 `[B,K,V]` minimum tensor를 쓰지 않고 block별 합으로 줄이는 **benchmark 전용** 구현이다. 추론 기본 경로에 반영하지 않았다.

| V | B | 위치 수 | h_hat | .75 h_hat+.25 h_bar | 추가 비용 [3회 범위] | Fused 혼합 |
|---:|---:|---:|---:|---:|---:|---:|
''']
    for c in result['compute']:
        if c.get('kind') or c['temperature']!=.7:continue
        t=c['timing_us'];d=c['paired_delta_us']['position_dense_mix minus position_dense_hat']
        parts.append(f"| {c['vocab']} | {c['batch']} | {c['positions']} | {t['position_dense_hat']['median']/1000:.4f} | {t['position_dense_mix']['median']/1000:.4f} | +{d['median']/1000:.4f} [{d['min']/1000:.4f}, {d['max']/1000:.4f}] | {t['position_fused_mix']['median']/1000:.4f} |\n")
    parts.append(r'''
추가 비용은 수락률 overlap을 구하고 별도 누적곱·혼합을 수행하는 비용이다. 계산량 O(BKV)가 추가되지만, 작은 B에서는 메모리 통과보다 kernel 호출 비용도 크다. 큰 V/B에서는 임시 tensor와 reduction 비용이 커지고 fused 구현으로 일부 감소한다. 어떤 크기에서 cache/memory 병목이 전환되는지는 profiler로 분리하지 않았으므로 원인별 기여도를 확정하지 않는다.

##### E. Root 후보 계산 전체와 tree

Softmax·위치 점수·source 점수·top-k를 포함하는 chain 후보 구성 시간, 단위 ms. LM head GEMM·target transformer·통신·수락 검증은 제외한다. Scalar arm은 proxy-only source에만 적용했고 q(y)가 이미 준비돼 있다는 조건이다.

| V | B | 위치 수 | Proxy+h_hat | Proxy+혼합 | Proxy+h_hat, q(y) 재사용 | e(1-q)+h_hat | e(1-q)+혼합 |
|---:|---:|---:|---:|---:|---:|---:|---:|
''')
    for c in result['compute']:
        if c.get('kind') or c['temperature']!=.7:continue
        t=c['timing_us'];vals=[t[k]['median']/1000 for k in ('proxy_hat','proxy_mix','proxy_hat_scalar_q','complement_hat','complement_mix')]
        parts.append(f"| {c['vocab']} | {c['batch']} | {c['positions']} | "+' | '.join(f'{x:.4f}' for x in vals)+' |\n')
    parts.append(r'''
q(y) 재사용은 target의 불필요 q softmax를 없앨 여지가 있음을 보여준다. 송신 측에서 이미 확률이 있으면 q(y) gather는 약3µs지만, logits만 남아 있으면 정규화가 필요하다. 이번 PyTorch `logsumexp→선택 logit` 구현은 full softmax 후 gather보다 위치 계산에서 빠르지 않았다. Scalar 결과를 만들었다는 사실만으로 O(V) 읽기나 kernel 비용이 사라지는 것은 아니다.

Tree는 C=3, 노드12의 기존 batch sibling ladder에 `overlap_mix=.25`를 추가해 별도 측정했다. 이는 chain의 h_bar를 그대로 전체 tree에 적용한 실험과 같지 않으며, 각 sibling의 조건부 residual overlap을 계산하는 기존 연구 옵션의 비용이다. e,q는 이미 정규화된 입력이다. Top-k, LM head와 target forward는 포함하지 않는다.

| V | B | 기존 tree ladder | Overlap 혼합 ladder | 추가 ms |
|---:|---:|---:|---:|---:|
''')
    for c in result['compute']:
        if not c.get('kind'):continue
        t=c['timing_us'];d=c['paired_delta_us']['tree_ladder_mix minus tree_ladder_hat']
        parts.append(f"| {c['vocab']} | {c['batch']} | {t['tree_ladder_hat']['median']/1000:.4f} | {t['tree_ladder_mix']['median']/1000:.4f} | +{d['median']/1000:.4f} |\n")
    parts.append(r'''
##### F. Full-q를 scalar로 바꾸는 순수 통신 진단

**Lossless engine을 scalar로 대체한 실험이 아니다.** 전체 q logits payload가 실제로 없어질 수 있다는 가정의 통신 구간 대조다. 단위 ms, 앞서 정의한 payload+ACK transaction. 순수 byte 전송 시간 또는 전체 inference 향상률로 해석하지 않는다.

| V | B | 위치 수 | Full-q bytes → scalar bytes | Full-q transaction | Scalar transaction | 차이 [3회 범위] |
|---:|---:|---:|---|---:|---:|---:|
''')
    for c in result['wire']:
        f=c['payloads']['full_q'];s=c['payloads']['scalar_q'];d=c['full_minus_scalar_us']
        parts.append(f"| {c['vocab']} | {c['batch']} | {c['positions']} | {f['bytes']:,} → {s['bytes']:,} | {f['gpu_us']['median']/1000:.4f} | {s['gpu_us']['median']/1000:.4f} | {d['median']/1000:.4f} [{d['min']/1000:.4f}, {d['max']/1000:.4f}] |\n")
    parts.append(r'''
표의 차이는 각 실행의 paired 차이를 집계한 것이므로 두 median 열의 차이와 같지 않을 수 있다. 특히 B1/V32000은 반복 간 호출 지연 변화가 커 유의미한 통신 개선을 확정하기 어렵다. B8/큰V의 payload 감소 효과는 반복에서 유지됐다. 장비의 P2P/PCIe/NUMA 및 통신과 계산의 겹침이 바뀌면 달라지므로 Blackwell/TP2의 실측 결과로 인용하지 않는다.

아래는 ACK를 추가하지 않은 원래 `dist.recv` 구간의 보조 측정이다. GPU event는 상대 sender가 준비될 때까지 기다린 시간도 포함한다. 호출 전 CPU barrier는 측정 밖이지만 barrier 해제 시점 차이가 영향을 줄 수 있다. 위 transaction 결과와 수치를 섞거나 평균하지 않는다. 단위 ms, 2개 process 실행의 median과 paired 차이 범위.

| V | B | 위치 수 | Full-q 수신 | Scalar 수신 | Full−scalar 범위 |
|---:|---:|---:|---:|---:|---:|
''')
    for c in result['one_way_receive']:
        d=c['full_minus_scalar_us']
        parts.append(f"| {c['vocab']} | {c['batch']} | {c['positions']} | {c['full_us']['median']/1000:.4f} | {c['scalar_us']['median']/1000:.4f} | [{d['min']/1000:.4f}, {d['max']/1000:.4f}] |\n")
    parts.append(r'''

##### G. 적용 판단과 후속 범위

1. 현재 full-q 경로에서 h_hat 대 혼합을 비교할 때는 통신을 동일하게 두고 추가 계산과 cache hit의 교환을 본다. h_bar 때문에 전체 분포 전송을 새로 추가했다고 집계하지 않는다.
2. Scalar 통신은 **별도 시스템 개선 주제**로 의미가 있다. 가장 단순한 후보는 chain+proxy-only root+h_hat다. 그러나 T>0 lossless 검증을 유지하려면 첫 거절 위치의 full q만 사후 요청하는 등 별도 프로토콜이 필요하다. 추가 왕복·draft worker 대기·분포 보존 비용을 포함해 비교해야 한다. 현재 구현하지 않았다.
3. 이상적인 chain 사후 요청이라면 scalar BK개와 거절한 요청당 V개를 보내는 구조로 줄일 수 있다. 하지만 이는 통신량 모델이며 지연 모델이 아니다. Tree는 sibling 거절 중간에도 residual 정보가 필요하므로 더 복잡하다.
4. e(1-q) 또는 residual root 정책을 유지하면 scalar q(y)만으로 현재 전체 후보 점수를 계산할 수 없다. Top-M 근사로 줄이면 후보 정책이 달라지고, exact pruning 방식은 별도 알고리즘 검증이 필요하다.
5. Residual 점수를 이미 만들 때는 `bar_alpha=1-sum([e-q]+)`라는 항등식으로 중간 연산 재사용을 검토할 수 있다. 단, y를 제외하거나 top-M을 자르기 **전**의 전체 residual이어야 한다. 이번에 항등식은 검사했지만 runtime에 결합하지 않았다.
6. 향후 scalar-first lossless 프로토콜을 완성하면 h_bar가 full 분포 조기 전송을 다시 요구하는 비용이 생길 수 있다. 따라서 '현재 추가 통신0'을 모든 미래 구현의 결론으로 일반화하지 않는다. 이러한 전송 최적화는 같은 검증 구조의 SSD에도 적용 가능성을 함께 검토해야 공정한 비교다.

새 component 검사: scalar/dense h_hat 일치, proxy-only scalar/dense root 일치, fused overlap 허용오차 검사, 서로 같은 선택 token 확률로도 overlap/correction이 달라지는 반례를 확인했다. 추가 GPU9조건에는 동일 분포·서로 겹치지 않는 분포·비정렬V를 포함했다. Fused는 부동소수점 reduction 순서가 달라 완전한 bitwise 동등성을 주장하지 않으며 production 채택 상태가 아니다. 기존 runtime77개 파일의 frozen hash가 일치하고 논문 원본도 clean이다.

![위치 점수와 통신 비용](position_cost/COSTS.png)

원시 결과: `position_cost/compute_r1..3.json`, `transfer_r1/r2.json`(단방향 보조), `transfer_ack_r1..3.json`(주 통신 대조), `CHECKS.json`. 재집계는 `position_cost/analyze.py`; 재현 command·source/raw hash와 archive는 `position_cost/MANIFEST.json`, `SOURCE_AND_RESULTS.tar.gz`에 기록한다. 이번 변경은 미커밋이며 push하지 않았다.
''')
    report=HERE.parent/'REPORT.md';marker='<a id="position-weight-cost-review"></a>'
    original=report.read_text().split(marker)[0].rstrip()
    report.write_text(original+'\n\n'+''.join(parts).strip()+'\n')
    plot(result)
    print('Updated',report,'and SUMMARY.json')


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    chosen=[c for c in result['compute'] if not c.get('kind') and c['temperature']==.7 and (c['batch'],c['positions']) in ((1,4),(8,12))]
    labels=[f"V={c['vocab']:,}\nB={c['batch']}, K={c['positions']}" for c in chosen]
    fig,axs=plt.subplots(1,3,figsize=(15,4.5));x=np.arange(len(chosen));w=.24
    for i,(key,label,color) in enumerate([('position_dense_hat','Observed h','#4575b4'),('position_dense_mix','Mixed h','#d73027'),('position_fused_mix','Mixed h, fused','#66a61e')]):
        axs[0].bar(x+(i-1)*w,[c['timing_us'][key]['median']/1000 for c in chosen],w,label=label,color=color)
    axs[0].set_title('Position weights: e and q already available')
    for i,(key,label,color) in enumerate([('proxy_hat','Proxy + observed h','#4575b4'),('proxy_mix','Proxy + mixed h','#d73027'),('proxy_hat_scalar_q','Proxy + scalar q(y) ready','#66a61e')]):
        axs[1].bar(x+(i-1)*w,[c['timing_us'][key]['median']/1000 for c in chosen],w,label=label,color=color)
    axs[1].set_title('Whole root scoring: excludes LM head')
    wire=[next(d for d in result['wire'] if (d['vocab'],d['batch'],d['positions'])==(c['vocab'],c['batch'],c['positions'])) for c in chosen]
    for i,(key,label,color) in enumerate([('full_q','Full q payload + ACK','#d73027'),('scalar_q','Scalar payload + ACK','#4575b4')]):
        axs[2].bar(x+(i-.5)*.3,[c['payloads'][key]['gpu_us']['median']/1000 for c in wire],.3,label=label,color=color)
    axs[2].set_title('Wire-size diagnostic: hypothetical protocol')
    for ax in axs:
        ax.set_xticks(x,labels);ax.set_ylabel('Latency (ms)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('RTX4090 component benchmarks; 3 process repetitions; no inference TPS claim',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'COSTS.png',dpi=180);fig.savefig(HERE/'COSTS.pdf');plt.close(fig)


if __name__=='__main__':main()
