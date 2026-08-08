# DUET hit-time tree rerank gate (2026-08-08)

## 질문

Draft는 넓은 동적 tree를 그대로 생성하되, cache hit 뒤 누적 confidence가 높은
lossless-closed 부분트리만 target에 보내면 target verify와 다음 P1 root 계산을
줄이면서 수락 품질을 유지할 수 있는가?

비교는 생성 예산을 고정했다. P1은 18 node, P2는 8 node를 계속 생성한다.
달라지는 것은 hit 뒤 target에 보내는 상한뿐이다.

## 사후 후보 선별

기존 실경로 topology/target-walk trace를 새 분석기의 `--rerank-caps`로 다시
분석했다.

### P1 trace: 159 actual hits

- 실제 hit tree 크기: 18 node 137건, 6 node 2건, 3 node 20건
- 평균 전송 node: 15.96
- 실제 accepted path: 평균 3.45 node

| verify cap | 평균 전송 node | node 감소 | accepted-node 보존 | full-path 보존 |
|---:|---:|---:|---:|---:|
| 12 | 10.79 | 32.39% | 98.72% | 95.60% |
| 14 | 12.52 | 21.59% | 99.64% | 98.74% |
| 16 | 14.24 | 10.80% | 100.00% | 100.00% |
| 18 | 15.96 | 0% | 100.00% | 100.00% |

### P2 trace: 65 actual hits

- 실제 hit tree 크기: 8 node 54건, 7 node 4건, 3 node 7건
- 평균 전송 node: 7.40

| verify cap | 평균 전송 node | node 감소 | accepted-node 보존 | full-path 보존 |
|---:|---:|---:|---:|---:|
| 7 | 6.57 | 11.23% | 99.11% | 98.46% |
| 8 | 7.40 | 0% | 100.00% | 100.00% |

이 수치는 후보 선별용이다. rejected proposal을 실제로 제거하면 residual RNG와
이후 autoregressive trajectory가 달라지므로 counterfactual AL은 live run으로
따로 판정했다.

## 짧은 실모델 선별

RTX 3090 target TP4 + draft 1GPU, seed 42, 8 prompt, output 192,
profiler OFF에서 후보만 비교했다.

### P1-only

| P1 gen/verify | TPS | tok/step | target verify | draft step | P1 hit | P1 AL |
|---|---:|---:|---:|---:|---:|---:|
| 18/18 | 49.53 | 3.76 | 65.21ms | 70.86ms | 0.570 | 3.63 |
| 18/14 | 60.46 | 4.18 | 55.33ms | 66.72ms | 0.566 | 4.36 |
| 18/12 | 60.11 | 3.82 | 52.05ms | 61.45ms | 0.543 | 3.71 |

12는 더 공격적으로 시간을 줄였지만 14보다 token/step이 낮아 최종 후보에서
제외했다.

### P2-only

| P2 gen/verify | TPS | tok/step | target verify | draft step | P2 hit | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| 8/8 | 63.82 | 3.77 | 50.27ms | 58.43ms | 0.225 | 1.82 |
| 8/7 | 61.82 | 3.65 | 49.11ms | 58.37ms | 0.248 | 1.91 |

P2 7은 verify를 약 1.2ms 줄였지만 현재 낮은 P2 hit 비중에서는 전체 TPS가
나빠졌다. 현재 K2=4/G2=8에서는 8을 유지한다.

## 3-seed 최종 gate

두 phase를 모두 tree `on`으로 두고 다음 두 arm만 비교했다.

- baseline: P1 18/18, P2 8/8
- rerank: P1 18/14, P2 8/8

각 seed는 20 prompt, output 256, profiler OFF다. 기준/후보를 seed 안에서
interleave하고 seed 123은 순서를 뒤집었다.

### seed별 TPS

| seed | baseline | rerank | delta |
|---:|---:|---:|---:|
| 42 | 56.01 | 57.88 | +1.87 (+3.34%) |
| 123 | 52.82 | 58.40 | +5.58 (+10.56%) |
| 2024 | 54.75 | 57.94 | +3.19 (+5.83%) |
| mean | 54.53 | 58.07 | +3.55 (+6.50%) |

### 평균 지표

| metric | baseline | rerank | delta |
|---|---:|---:|---:|
| TPS | 54.53 | 58.07 | +6.50% |
| token/step | 3.957 | 3.920 | -0.93% |
| target full step | 76.06ms | 70.99ms | -5.06ms |
| target verify | 61.75ms | 55.80ms | -5.95ms |
| draft step | 69.81ms | 66.28ms | -3.53ms |
| total cache hit | 0.803 | 0.803 | 0.000 |
| P1 hit | 0.562 | 0.546 | -0.017 |
| P2 hit | 0.240 | 0.258 | +0.018 |
| P1 conditional AL | 3.803 | 3.850 | +0.047 |
| P2 conditional AL | 1.993 | 1.937 | -0.057 |

모든 seed의 TPS 방향은 양수다. 다만 seed 수가 3이므로 논문용 95% 유의성 주장은
하지 않는다. token/step이 평균 0.9% 낮고 P1/P2 hit 구성도 이동했으므로
14-node 설정은 수학적 quality-equivalent 설정이 아니라 **throughput 설정**이다.
전체 cache hit과 조건부 AL은 평균적으로 거의 보존됐지만 P1 hit 자체는 1.7%p
낮았다.

## 판정과 권장 설정

현재 K1=9/K2=4/G1=18/G2=8 workload의 throughput 설정은 다음이다.

```text
--duet_p1_tree_max_nodes 18 --duet_p1_tree_verify_nodes 14
--duet_p2_tree_max_nodes 8  --duet_p2_tree_verify_nodes 8
```

- P1 14: 채택 가능한 throughput 후보. target와 draft critical path를 함께 줄였다.
- P1 12: aggressive ablation. 기본 권장값으로 채택하지 않는다.
- P2 7: 현재 config에서는 기각. 구현은 큰 K2/G2 실험을 위해 유지한다.
- P1 16: 기존 trace에서는 accepted path 100%를 보존한 quality-conservative 후보지만
  live TPS gate를 하지 않았으므로 이번 champion으로 주장하지 않는다.

Config 기본값은 verify cap을 생략하면 generation cap과 같게 두어 기존 동작을
정확히 보존한다. 위 14 값은 명시적으로 throughput 실험을 할 때 사용한다.

## 재현

```bash
cd /home/chokwans99/PSD/ssd

STAGE=screen \
  bash experiments/proxy_async_overlap/tree_sweep/run_tree_rerank_gate_20260808.sh

STAGE=final P1_WINNER=14 P2_WINNER=8 \
  bash experiments/proxy_async_overlap/tree_sweep/run_tree_rerank_gate_20260808.sh
```

- 구현 commit: `3219f71`
- 최종 gate script commit: `a15e52e`
- 테스트: `test_p1_dynamic_tree.py` 24/24,
  `test_args_cleanup_equiv.py` 16/16
