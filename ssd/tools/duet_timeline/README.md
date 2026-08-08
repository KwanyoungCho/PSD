# DUET timeline 도구

Target의 앞/뒤 모델 실행과 early-exit proxy 계산, draft의 P1/P2 실행을 같은
시간축에서 확인하는 도구다. Chain과 tree 모두 같은 명령을 사용한다.

## 프로파일 수집

실험 실행 환경에 다음 값을 추가한다.

```bash
SSD_PROFILE_DUET=1
SSD_PROFILE_DUET_DETAIL=1
SSD_PROFILE_DIR=/absolute/path/to/profile_dir
```

Proxy CUDA graph는 기본으로 켜져 있다. 비교 진단이 필요할 때만
`SSD_CHAIN_PROXY_GRAPH=0` 또는 `SSD_TREE_PROXY_GRAPH=0`으로 끈다.

새 profile에는 다음 구간이 별도로 기록된다.

- `exit_logits`: early-exit hidden state에서 logits를 만드는 시간
- `exit_proxy_launch`: 기본 GPU 작업줄에서 proxy 작업을 예약하는 시간
- `exit_proxy_side`: 별도 GPU 작업줄에서 실제 logits/proxy/send를 수행한 시간
- `chain_proxy_graph_replay`: chain K1/K2 및 miss의 후보 계산 graph
- `tree_proxy_graph_replay`: tree hit의 후보 계산 graph
- `proxy_send_enqueue`: 계산된 proxy 정보를 비동기 통신에 넣는 시간

따라서 긴 막대가 실제 GPU 계산인지, CPU 예약 지연인지, 통신 enqueue인지
timeline에서 분리해서 볼 수 있다.

## 그래프 생성

`ssd` 디렉터리에서 다음처럼 실행한다.

```bash
bash tools/duet_timeline/plot.sh \
  experiments/proxy_async_overlap/tree_sweep/my_run/profile_tree
```

대표 `hit_k1`, `hit_k2`, `miss` step의 PNG가 입력 디렉터리에 생성된다.
특정 step만 보고 싶으면 plotter 인자를 그대로 덧붙인다.

```bash
bash tools/duet_timeline/plot.sh PROFILE_DIR \
  --step-id 120 --out PROFILE_DIR/timeline_step120.png
```

한 상태의 중앙값 한 장만으로 run의 분산을 숨기지 않으려면 p25/p50/p75 대표
step을 함께 만든다.

```bash
python tools/duet_timeline/plot_representatives.py PROFILE_DIR
```

각 `hit_k1`, `hit_k2`, `miss` 상태에서 target full-step duration의
p25/p50/p75를 고르며, draft response marker까지 저장된 step만 사용한다. 결과는
개별 PNG 9장, 선택 근거 `representatives.tsv`, 3×3
`timeline_representatives_contact_sheet.png`다. 행은 cache 상태, 열은 duration
quantile이다. 기존 JSON은 변경하지 않는다.

## Proxy 구간 숫자 요약

```bash
python tools/duet_timeline/summarize_proxy.py PROFILE_DIR
```

cache 상태와 구간별 표본 수, GPU p50/p95, CPU dispatch p50을 출력한다.
구버전 profile의 `proxy_compute_send` 항목도 읽을 수 있지만, 실제 계산과
예약 지연을 구분하려면 이번 변경 이후 새 profile을 수집해야 한다.
