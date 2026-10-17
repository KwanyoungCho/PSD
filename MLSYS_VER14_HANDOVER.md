# MLSys ver1.4 인수인계

2026-10-17 KST. 이 checkout의 기준은 **`MLSys-ver1.4`**다. 부모는 `MLSys-ver1.3@5642de044d9fa2d5e80dfaeeccc8f91cfd1be8e1`, runtime 커밋은 **`fa9963e56e45f73bba3aa4eb39990ebca3ec32c1`**다. 저장소는 `https://github.com/KwanyoungCho/PSD.git`이며 현재 작업 폴더는 `/home/chokwans99/PSD-MLSys-ver1.4`다.

기존 ver1.3/실험 브랜치를 유지하고 새 브랜치에서 **검증된 실행 최적화만** 반영했다. 실험 history를 merge하거나 미채택 코드를 default로 켜지 않았다. `ssd/pyproject.toml`의 upstream 패키지 버전과 연구 브랜치 버전은 별개이며 버전 확인에는 Git commit을 쓴다.

## 검토 문서와 변경

먼저 [최종 보고서](results/mlsys_ver14/REPORT.md)를 읽는다. 여기에는 **실제 ver1.3과 최종 ver1.4를 직접 비교**한 두 GPU pair의 전후 시간, 출력/AL/hit 동일성, 조건별 한계와 제외 step 수가 있다. [직접 비교 그림](results/mlsys_ver14/DIRECT_LATENCY.pdf), [원시 요약](results/mlsys_ver14/DIRECT_COMPARISON.json), [소스·실행 audit](results/mlsys_ver14/RELEASE_AUDIT.json)도 함께 확인한다.

채택한 runtime 변경은 아래 7파일이다 (`ssd/ssd/engine/` 아래).

| 파일 | 반영한 내용 |
|---|---|
|draft_runner.py|DS/PS topology·응답 사전 준비, hit 응답 통합, miss 조회 재사용, CPU glue ancestry 입력 준비|
|helpers/response_topology.py|GPU/CPU/Python integer topology snapshot 및 hit root 검증|
|helpers/fused_tree_response.py|B1 사전 tree 응답 선택, 한 번의 descriptor readback, dummy argmax 생략과 RNG 보존, glue/request 배열|
|speculator_async.py|B1 non-EAGLE 요청의 CPU packet 준비 및 일괄 GPU 복사|
|helpers/p2_tree.py|정확한 전체 overwrite 시 zero 생략, partial tail zero 유지|
|model_runner.py|같은 불변 metadata의 한 call 내 검증 재사용|
|verifier.py|검증 결과 공유와 finally 해제, graph에서 쓰지 않는 eager list 준비 생략|

현재 layer/K1/K2/root/width/M/C default와 preset을 바꾸지 않았다. Root 후보·위치 수식, q-path/확장 경쟁, draft q, target 수락/복구 법칙, sampling 온도, scheduler, KV page/복원 의미, wire layout과 send 순서는 그대로다. 새 사용자 옵션도 없다.

B1 tree G=M의 응답을 최적화하며, 사전 packet을 만들 수 없는 경로는 기존 serving을 유지한다. 두 phase 모두 사전 tree packet이 없을 때는 GPU lookup/readback을 추가하기 전에 돌아간다. B>1/EAGLE은 이 tree helper를 사용하지 않는다. Request CPU packet은 B1 non-EAGLE에서 같은 bytes를 생성한다. Chain/greedy smoke와 기존 batch/KV 검사도 통과했다.

미채택된 bucket graph, q-receive overlap, async/token-first response, host-list 실패 후보, packed target, mixed-miss 정책 및 다른 서버의 알고리즘은 채택하지 않았다. 과거 증거는 `results/mlsys_ver14/evidence/`에 보존했지만 runtime에는 이식하지 않았다.

## 다른 서버에서 받기

기존 저장소에서 실행한다. 아래 새 이름이 이미 있으면 다른 worktree/branch 이름을 쓰고 기존 작업을 삭제하거나 reset하지 않는다.

```bash
git fetch origin MLSys-ver1.4
git worktree add -b review/mlsys-ver1.4 ../PSD-MLSys-ver1.4-review origin/MLSys-ver1.4
cd ../PSD-MLSys-ver1.4-review
git log -1 --format='%H %s'
git status --short
git merge-base --is-ancestor fa9963e56e45f73bba3aa4eb39990ebca3ec32c1 HEAD
git diff 5642de044d9fa2d5e80dfaeeccc8f91cfd1be8e1 HEAD --stat -- ssd/ssd
```

Runtime hash 확인은 모델/GPU 없이 실행할 수 있다.

```bash
python - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('results/mlsys_ver14/RUNTIME_MANIFEST.json').read_text())['sha256']
for name,expected in m.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('runtime hash PASS:',len(m),'files')
PY
```

직접 이식해야 한다면 단일 함수만 복사하지 말고 이 7파일의 의존성을 함께 검토한다. 다른 서버의 연구 알고리즘 변경과 충돌하면 root/위치/tree 점수는 그 서버의 명시적인 결정에 따르고, immutable metadata·proposal q·RNG·packet 수명 및 KV 처리 순서를 유지한다. 검토용 파일 목록은 [RUNTIME_CHANGED_FILES.txt](results/mlsys_ver14/RUNTIME_CHANGED_FILES.txt)다. 원본 논문 브랜치로 전부 덮어쓰지 않는다.

## 검사와 재현

최종 runtime 검사 106회는 selected 28, 기존 batch 8, compatibility 70이며 failure/error/skip 0이다. [checks/CHECKS.json](results/mlsys_ver14/checks/CHECKS.json)에 로그와 기준 commit이 있다. 최초 chain fixture 오류2건은 `batch_initial.log`에 남겼고 helper eligibility 순서를 고친 뒤 재검증했다. 과거 숫자를 이번 버전의 새 검사 수로 더하지 않는다.

서버에 맞는 Python 환경과 SSD_HF_CACHE/SSD_DATASET_DIR/SSD_CUDA_ARCH를 준비한 뒤, 예를 들어 다음처럼 선택 회귀를 실행한다. CUDA_VISIBLE_DEVICES에는 확보한 빈 GPU만 지정한다.

```bash
cd ssd/tests
PYTHONPATH=.. python -B -m unittest -v \
  test_tree_proxy_preparation test_response_snapshot test_response_topology \
  test_tree_verify_planless test_tree_topology_gpu test_tree_verify_mask_direct \
  test_tree_accept_port test_fused_tree_response test_bulk_spec_request
```

전체 재현 실행 설정은 `results/mlsys_ver14/reproduce/`에 있다. 모델 경로·GPU/CPU pin·포트는 **orion 로컬 환경용**이므로 다른 서버 경로에 맞춰 바꾼다. Raw는 `release_raw.tar.xz`, 과거 raw는 각 evidence 폴더의 archive다. 추론 없이 직접 비교를 다시 집계하려면:

```bash
cd results/mlsys_ver14
tar -xJf release_raw.tar.xz
python -B analyze_release.py
python -B plot_release.py
```

이번 release 직접 비교는 7B+68m, B1, T0.7의 안정된 짧은 측정이다. 과거 별도 48문항/56턴은 prepack base 비교라는 점을 구분한다. 전체 480문항/560턴·1024출력 검증은 사용자가 취소했던 상태를 유지했으며 이번에 재개하지 않았다. 다른 GPU/모델·B>1의 성능을 확인하려면 해당 서버에서 동일 설정의 ver1.3/ver1.4를 비교한다. 구현 및 기능 지원과 그 환경의 성능 입증을 구분한다.
