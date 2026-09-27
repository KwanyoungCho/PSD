"""Complete CPU-only analysis after the serialized inference campaign."""
import hashlib
import json
from pathlib import Path
import time
from campaign import HERE


def main():
    while not (HERE/'tree_validation.json').exists():time.sleep(10)
    from diagnostics import main as diagnose
    from tree_diagnostics import main as tree_diagnose
    from finalize import extensions,confirmation,cost_audit
    from reward_expectation import main as reward
    from checks import main as check
    from make_figs import main as figures
    from build_report import main as report
    diagnose();tree_diagnose();extensions();confirmation();cost_audit();reward();check();figures();report()
    final=json.loads((HERE/'confirmation_summary.json').read_text())
    checkdata=json.loads((HERE/'checks.json').read_text())
    (HERE/'HANDOVER.md').write_text(f'''# DUET calibration 인계 — 2026-09-13

## 완료된 작업

사용자는 기존 calibration의 효과가 좋지 않다고 했고, 새 설계의 feasibility를
수식과 실험으로 확인하라고 요청했다. 기존 추천을 재사용하지 않고 이 디렉터리에
새 prototype을 만들었다. 이전 residual/proxy 설명을 재개할 TODO는
[미완료 목록](../residial_dist/training_free/TODO.md)에 보존했다.

최종 결과는 [REPORT.md](REPORT.md), 짧은 숫자는 [NUMBERS.txt](NUMBERS.txt),
추천은 [recommended.json](recommended.json)에 있다.

- 성공한 inference 실행 {checkdata['complete_runs']}개, 캠페인 내 중복 없는 56 prompts.
- 최초 5 anchors → 동결 → 19개 검증. Cycle MAPE 1.58%, reward MAPE 17.90%.
- 최초 추천 exit40·4/2, 19-grid 최고 exit56·4/2, 최초 regret 4.65%.
- 별도 선택 prompts에서 exit별 4/2 및 경계 확장 4/1, 총 6개를 실제 측정했다.
- 보완 최종 선택: {final['secondary_selected']}.
- Confirmation: 다른 16 prompts × 2 seeds, 256 output tokens, profiler OFF.
- root/fanout/tree/후보 source/극단적 low-K 보조 실험, 7-layer 분포 replay,
  중간 tree verify 형상 및 조건부 기대 AL 통계를 확인했다.
- Tree 평균 보간은 67.89% 오차로 실패했다. Nv8 anchor의 0.69/1.50초 지연은
  후속 반복에서 재현되지 않았고 중앙값은 약 28ms로 유지됐다. 사후 중앙값 보간은
  독립 검증 성공으로 취급하지 않는다. `tree_diagnostics.json`에 근거가 있다.
- 모델/head 학습 없음. Production verifier/sampler와 기존 사용자 수정은 그대로다.

## 재현과 데이터 계약

집계 재생성 명령은 REPORT 마지막 절에 있다. `campaign.py`는 실제 GPU 실행을
담당하며 완료한 디렉터리는 재실행하지 않는다. 새 캠페인은 기존 frozen/results를
덮어쓰지 말고 별도 디렉터리와 plan으로 시작한다.

- `frozen.json`은 최초 validation 전에 동결된 모델/입력 hash를 갖는다.
- `refinement_plan.json`은 최초 3개 validation을 본 뒤 등록한 secondary 설계다.
- `boundary_plan.json`은 19개 validation을 본 뒤, selection 전에 추가한 확장이다.
- `refinement_frozen.json`은 confirmation 전에 고정한 선택이다.
- `candidate_frozen.json`은 calibration으로 고른 정책을 validation scoring 전에 고정했다.
- `environment.json`, `checks.json`, `ARTIFACTS.json`은 환경/무결성/산출물 목록이다.

## 해석할 때 지킬 범위

1. 19-grid 최고나 confirmation 최고를 전역 최적이라고 부르지 않는다.
2. 1.58%는 지정된 chain 검증 영역이다. K1=K2=2는 8.1–13.6% 오차였다.
3. 최초 품질 모델 실패를 보완 추천의 성과로 지우지 않는다. Gap-only와의 작은 차이도 보고한다.
4. Root replay는 M32, P1 dedup 전 이론 coverage다. 실제 wire/cache hit/AL/TPS가 아니다.
5. Candidate 혼합의 실제 TPS 개선 및 Mirror-SD 전체 시스템 우위는 이번에 입증하지 않았다.
6. Tree N_gen과 N_verify는 별도 축이다. Current dynamic의 legacy beta/fanout knob가
   비활성이었던 결과를 다른 selector 또는 candidate-source beta로 일반화하지 않는다.
7. 현재 calibration 결과는 모델쌍/quantization/RTX4090/TP/B1/T.7/문맥 범위에 종속된다.
8. 기대 AL 통계는 accept coin 분산 감소 방법을 확인한 추가 진단이다. 현재 추천을
   이 값으로 재학습하거나 그 품질 오차를 해결했다고 주장하지 않는다.

## 다음 작업

- 사용자가 준비할 새 tree 선택기의 shape/cost/reward adapter를 정한다.
- proxy-wait와 draft-wait 양쪽에서 anchor를 잡고, 조건부 기대 AL을 calibration
  통계에 포함한 다음 독립 정확도/비용을 재검증한다.
- 모델쌍/하드웨어/문맥 bucket을 늘려 정해진 측정 예산에서 regret을 비교한다.
- 기존 수식 설명 TODO는 사용자가 이해했다고 확인한 항목만 완료 처리한다.

초기 tree 128MiB workspace 실패는 로그를 보존하고 256MiB로 재시도했다.
초기 probe는 매 요청의 collector를 저장하지 않아 마지막 요청만 남았다.
`run_probe.py`에서 요청별 파일명과 명시적 flush로 해결하고 전체 데이터를 재수집했다.
부적합한 probe 실행은 `failed_attempts/`에 분리했고 비용도 별도로 기록했다.
GPU 3–7만 사용했다. 다른 사용자의 GPU 0–1 작업에는 관여하지 않았다.
''')
    manifest={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
              and p.name not in ['ARTIFACTS.json','finish.log'] and not p.name.endswith('.log')}
    (HERE/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2))
    print('All requested feasibility artifacts generated.',flush=True)


if __name__=='__main__':main()
