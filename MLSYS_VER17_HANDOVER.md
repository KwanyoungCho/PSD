# MLSys ver1.7 적용 안내

기준은 `MLSys-ver1.6@4c7e43382db6215166265ed1f19d0a5d6bdd72b3`이다.
첫 ver1.7 커밋은 검증이 완료된 P1 후보 제외 수정만 반영한다.
첫 커밋 `5a881cc2`에는 P1 수정만 포함된다. 이어지는 후속 커밋은 아래 세 오류와
동일한 제외 오류가 남아 있던 batched P1 경로를 함께 수정한다.

## 반영한 변경

동적 P1이 일렬로 저장된 다음 token 하나를 제외하던 것을 같은 prefix의 기존
직접 자식 token 전체를 제외하도록 고쳤다. Leaf에서는 아무 token도 제외하지 않는다.
기존 parent 계산을 공유하고 작은 top-k 표에 fused CUDA 필터를 적용한다.
원래 draft q와 reach 점수, WOR 및 residual 검증 수식은 유지한다.
새 공개 파라미터, CPU readback, 통신 payload 증가는 없다.
Scalar 대입의 host→device copy도 제거했다.

CPU/CUDA/graph 회귀 62개 통과. 동일 입력의 eager root 준비 평균은 context 크기에
따라 약 6–19µs 감소했다. 실제 짧은 84턴×3-seed 검사에서 step은
19.122→19.095ms였으나, AL 2.859→2.798, TPS 147.85→144.68이었다.
Hit rate는 87.98→88.67%였다. **성능 향상 릴리스라고 주장하지 않는다.**
전체 Spec-Bench best-3(2,0,10) 재측정은 추가 검토 완료 후 별도로 진행한다.

[측정·검증 보고서](results/mlsys_ver17/p1_child_exclusion/REPORT.md)와
[요약 수치](results/mlsys_ver17/p1_child_exclusion/SUMMARY.json)를 참고한다.
원시 데이터와 과거 결과는 해당 보고서의 원본 경로에 보존했다.

## 다른 서버에서 사용

```bash
git fetch origin MLSys-ver1.7
git worktree add --detach ../PSD-MLSys-ver1.7 origin/MLSys-ver1.7
cd ../PSD-MLSys-ver1.7
python3 -B ssd/tools/verify_dispatch/check_release.py
```

모델·입력·환경·기존 파라미터는 변경하지 않는다. 연구 버전은 branch/commit으로
구분하며 upstream Python 패키지 버전을 바꾸지 않는다. 기존 ver1.6의 실행 경로
조건은 [안내](MLSYS_VER16_HANDOVER.md)를 따른다. 성능 측정 중 소스를 갱신하지 않는다.

기존 SSD CUDA 환경에서 검사를 재현할 수 있다.

```bash
PYTHONPATH=ssd:ssd/tests SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp \
python -B -m unittest -v test_p1_child_exclusion test_p1_dynamic_tree \
  test_tree_live_siblings test_tree_policy_defaults test_args_cleanup_equiv
```

GPU 번호와 Python 경로는 각 서버 환경에 맞춘다.

## 추가 검토 반영

- P1/P2 및 context bucket의 generator가 같은 난수열로 시작하던 것을
  phase/context/lane별 안정적인 seed 파생으로 분리했다. 초기화에서만 계산하며
  warmup 상태 복원 및 CUDA graph의 정상 RNG offset 진행을 유지한다.
- Proxy 점수 wire는 v3이다. uint32 token과 비음수 FP32 score의 원래 비트를
  같은 int64에 보존하고 sign bit를 version marker로 사용한다. 정확한 0과 작은
  양수의 크기를 유지하며, 기존 log 양자화·1e-6 하한은 제거했다. root prior의
  1e-9 log/power floor도 제거했다. 최종 verifier의 q나 수식은 변경하지 않았다.
- Selector는 zero-score padding을 활성 후보에서 제외한다. 후보가 부족하면
  물리 shape를 유지하면서 token/score를 안전한 0으로 채우고 확장 예산을 0으로 둔다.
- B>1 문맥 보호 종료에서 이미 선택한 미완료 요청을 원래 순서대로 running 큐에
  복원한다. 정상 decode의 hot path에는 추가 작업이 없다.
- Batched P1도 기존 직접 자식 전체를 제외한다. 하나의 fused kernel을 공유하며
  ragged length 밖의 padding은 제외 대상에 넣지 않는다.

**Wire v3는 v1.6과 호환되지 않는다. Target/draft 모두 같은 ver1.7 checkout으로
새 프로세스를 시작한다. 실행 중 한쪽만 업데이트하면 안 된다.**

[추가 검토 보고서](results/mlsys_ver17/additional_review/REPORT.md)에 재현과
287개 회귀 검사 및 latency 측정을 기록했다. Uniform toy CUDA sampling의 난수
중복은 재현했으나, 기존 실제 모델 출력의 편향 크기를 측정한 것은 아니다.
논문 best-3 seed는 기존과 같은 2·0·10을 고정하며 새 결과로 다시 고르지 않는다.
