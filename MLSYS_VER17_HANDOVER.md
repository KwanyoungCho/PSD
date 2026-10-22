# MLSys ver1.7 적용 안내

기준은 `MLSys-ver1.6@4c7e43382db6215166265ed1f19d0a5d6bdd72b3`이다.
첫 ver1.7 커밋은 검증이 완료된 P1 후보 제외 수정만 반영한다.
추가로 제기된 RNG 스트림, proxy 점수 전송, scheduler 조기 종료 검토는 진행 중이며,
이 최초 커밋에서 해결됐다고 간주하지 않는다.

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
