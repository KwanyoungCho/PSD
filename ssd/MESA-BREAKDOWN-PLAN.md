# MESA-SSD Per-Phase Breakdown — Minimal Plan

## 0. 원칙

- **기존 함수 body / signature 손대지 않는다.** 리팩터링 없음, 이름 변경 없음, context manager 없음.
- 각 측정 지점엔 **한 줄씩 두 번** 추가 (start event, end event). 그 외엔 아무것도 안 건드림.
- Default off. `SSD_PROFILE_MESA=1`일 때만 record. Off면 helper 호출 자체가 즉시 `return None`이라 사실상 nop (~10 ns).
- NVTX / nsys / 플롯 복잡도 / baseline 별도 계기 — **전부 빠짐**. 나중에 필요할 때 따로.

---

## 1. 추가 코드 (전부 신규, 기존 코드 수정 없음)

### 1.1 `ssd/engine/helpers/cudagraph_helpers.py` — 30 LoC 이하

파일 하단에 append:

```python
PROFILE_MESA = os.environ.get("SSD_PROFILE_MESA", "0") == "1"
_mesa_events = []   # [(step, label, start_ev, end_ev)]

def mesa_record(step, label):
    """Returns a CUDA event recorded now, or None if profiling off."""
    if not PROFILE_MESA:
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev

def mesa_close(step, label, start_ev):
    """Records the end event and appends to the global list."""
    if start_ev is None:
        return
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    _mesa_events.append((step, label, start_ev, end_ev))

def mesa_dump(tag):
    """End-of-run: one sync, extract elapsed_ms per event, write JSON."""
    if not _mesa_events:
        return
    import json
    torch.cuda.synchronize()
    rows = [{"step": s, "label": l, "ms": a.elapsed_time(b)}
            for s, l, a, b in _mesa_events]
    with open(f"/tmp/mesa_profile_{tag}.json", "w") as f:
        json.dump(rows, f)
    print(f"[mesa_profile] {len(rows)} events -> /tmp/mesa_profile_{tag}.json", flush=True)
    _mesa_events.clear()
```

### 1.2 측정 지점에 2줄씩 추가 (기존 함수 body는 그대로)

| 프로세스 | 라벨 | 파일:함수 | 추가 위치 |
|----------|------|-----------|-----------|
| draft | `glue` | `draft_runner.py::_glue_decode` | 첫 줄에 start, return 직전에 close |
| draft | `phase1_replay` | `cudagraph_helpers.py::run_fi_tree_decode_cudagraph` (layout=draft) | `graph.replay()` 직전/직후 |
| draft | `proxy_wait` | `draft_runner.py` (proxy irecv `work.wait()` 근처) | wait 호출 직전/직후 |
| draft | `phase2_replay` | `cudagraph_helpers.py::run_fi_tree_decode_cudagraph` (layout=proxy) | 위와 동일 (layout 구분) |
| draft | `merge_cache` | `draft_runner.py::_merge_and_populate_cache` | 첫 줄 / return 직전 |
| target | `graph_pre` | `cudagraph_helpers.py::run_mesa_verify_cudagraph` | `graph_pre.replay()` 직전/직후 |
| target | `proxy_compute_send` | `verifier.py::_compute_and_send_proxy` | 첫 줄 / isend 후 |
| target | `graph_post` | `cudagraph_helpers.py::run_mesa_verify_cudagraph` | `graph_post.replay()` 직전/직후 |

각 지점 실제 추가되는 코드:

```python
_ev = mesa_record(step, "glue")
# ... 기존 body 그대로 ...
mesa_close(step, "glue", _ev)
```

총 8 × 2 = **16줄**. `step` 값은 이미 함수에 전달되는 argument 또는 가까이 있는 counter 그대로 씀 (새로 추가하지 않음; 없으면 해당 지점만 skip).

### 1.3 End-of-run dump (각 프로세스 exit 지점)

`llm_engine.py`에서 기존 METRICS print 뒤, 그리고 draft_runner의 loop 종료 뒤에 한 줄씩:

```python
from ssd.engine.helpers.cudagraph_helpers import mesa_dump
mesa_dump("target" if self.rank == 0 else "draft")
```

끝. 기존 로직 건드리지 않음.

### 1.4 간단 플롯 (신규 파일 하나) — 30 LoC 이하

`bench/plot_mesa_breakdown.py`:

```python
import json, sys
import pandas as pd
import matplotlib.pyplot as plt

dfs = []
for tag in ("draft", "target"):
    try:
        rows = json.load(open(f"/tmp/mesa_profile_{tag}.json"))
        for r in rows: r["proc"] = tag
        dfs.append(pd.DataFrame(rows))
    except FileNotFoundError:
        pass
df = pd.concat(dfs, ignore_index=True)
summary = df.groupby(["proc", "label"])["ms"].agg(["mean", "median", "count"])
print(summary)
summary["mean"].unstack("proc").plot.bar(figsize=(10, 5))
plt.ylabel("ms / step"); plt.tight_layout()
plt.savefig("/tmp/mesa_breakdown.png", dpi=120)
print("-> /tmp/mesa_breakdown.png")
```

---

## 2. 변경 요약

| 파일 | 추가 | 수정 |
|------|-----:|-----:|
| `ssd/engine/helpers/cudagraph_helpers.py` | +30 LoC (하단 append) | 0 |
| `ssd/engine/draft_runner.py` | +6 LoC (3 지점 × 2줄) | 0 |
| `ssd/engine/verifier.py` | +2 LoC (1 지점 × 2줄) | 0 |
| `ssd/engine/llm_engine.py` | +2 LoC (dump 호출) | 0 |
| `bench/plot_mesa_breakdown.py` | +30 LoC (신규) | — |

**합계 ~70 LoC, 전부 추가만**. 기존 함수 body 단 한 줄도 변경 없음.

---

## 3. 성능 영향

- `PROFILE_MESA=0` (default): helper 첫 줄에서 `return None`. Python 함수 호출 overhead ~100 ns × 16 호출/step × 수만 step = 수 ms 수준. **측정상 0%**.
- `PROFILE_MESA=1`: CUDA Event record ~1-2 µs × 16/step = ~30 µs/step. Step당 ~40ms vs 30 µs → **<0.1%**.
- End-of-run sync 1회 + JSON write ~수십 ms.

검증: on/off 각각 baseline_f3 300 seqs 돌려 throughput 차이 확인 (예상 ≤1%).

---

## 4. 작업 시간 & 결정할 것

- 구현: **1~1.5 시간**
- 돌려서 JSON 얻기 + plot: 한 MESA run (300 seqs, 기존 ~10분) + plot 스크립트 1초
- 결정 필요: 위 8개 지점 그대로 OK? 빼거나 더할 거 있음?

OK 주시면 착수.
