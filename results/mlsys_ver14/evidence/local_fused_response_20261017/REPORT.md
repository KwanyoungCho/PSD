# 큰 구간을 통합한 B1 tree 실행 최적화 검토

앞선 약 0.04ms 개선은 작았다. 이번에는 hit 응답 전체와 glue 입력 준비의 반복 동기화·작은 GPU 호출을 묶어서 줄였다. 최종 통합 버전은 12문항 반복 비교에서 step 2.06%, 별도 48문항/56턴에서 2.93% 감소를 관찰했다. 같은 엔진의 무작위 경로 전환에서는 초기 통합 경로의 직접 효과가 약 0.35ms였다. 큰 폭의 모델 가속을 입증한 결과는 아니며, 실행 오버헤드 감소를 확인한 결과다.

## 바꾼 작업

1. **Tree hit 응답 통합**: hit/all/phase/root/길이/문맥을 따로 GPU에서 읽던 경로를 작은 descriptor 한 번의 readback으로 합쳤다. 이전 step의 DS/PS 결과로 metadata·토큰 응답 패킷을 미리 준비한다. Hit에서는 해당 row를 선택하고 실제 proposal parent-q만 gather하여 전송한다. 전송 형식·메시지 순서·q 행 수는 바꾸지 않았다.
2. **불필요한 dummy argmax 제거**: Tree hit가 덮어쓰는 일반 chain 토큰의 argmax를 수행하지 않는다. 기존 uniform q 생성은 RNG 상태를 같게 유지하기 위해 그대로 소비한다. 실제 proposal q나 토큰 선택 정책을 바꾸지 않았다.
3. **Glue 준비 통합**: 이미 CPU에 검증해 둔 topology에서 depth·ancestor/context 행렬을 계산한 다음 배열 단위로 GPU에 복사한다. GPU 노드를 CPU scalar로 반복해서 읽거나 작은 GPU 배열을 노드별로 수정하던 처리를 줄였다. Glue 입력은 응답 전송 후 실제 draft 사용 전에 만든다. KV restore·staging·attention 의미는 유지했다.
4. **Target 요청 준비 통합**: Seq/cache key·문맥 길이·block table·float32 온도 bit를 CPU 배열 하나에 넣고 한 번에 GPU로 복사한다. GPU scalar 쓰기·형변환·cat을 반복하지 않는다. 기존 int64 wire bytes와 같다.
5. **Miss 중복 검사 제거**: 초기 통합 구현은 miss를 판별한 다음 일반 경로에서 다시 검사해 비용을 늘렸다. 최종 버전은 이미 얻은 miss match를 재사용한다. Miss의 JIT draft 길이·샘플링 정책은 그대로다.
6. 이전 검토의 작은 prep 변경(전체 overwrite 시 zero 생략, 불변 metadata 검증 재사용, graph에서 쓰지 않는 eager list 생성 생략)도 포함한다. 이전 효과와 이번 효과를 더하지 않고 최종 코드 전체를 직접 비교했다.

`tolist()`를 모두 없앤 것은 아니다. 필요한 작은 readback을 하나로 합쳤다. GPU→CPU→GPU 왕복과 작은 호출의 수를 줄이는 방향이다. Host/CPU 제어를 전부 GPU로 이식하거나 별도 CPU 통신 프로토콜을 추가한 결과가 아니다.

## 고정 조건

- 로컬 orion GPU 4/5, 6/7(각 PIX pair), RTX 4090. 다른 사용자의 작업 종료 없음.
- LayerSkip-Llama2-7B + JackFram/llama-68m, full BF16, target TP=1 + draft GPU=1, B=1, T=0.7.
- K1=8, K2=5, exit=22(0-based), DS 위치당 root=4, DS continuation 입력=32, PS root/입력=15/15, C=3, G=M=15/15.
- 기존 complement root `e(1-q)`, 위치 overlap 혼합 0.25, 마지막 깊이 q-path 경쟁. 알고리즘·예산·threshold·확률·온도 변경 없음.
- 12문항은 입력 최대 512, 출력 최대 256, 2 passes. 별도 Spec-Bench 48문항/56턴은 입력 무절단, 출력 최대 256, 1 pass. 전체 480문항/560턴·1024토큰 평가가 아니다.
- 양 arm prefix cold 동일. 절대 TPS를 논문의 prefix cache 설정 성능으로 해석하지 않는다. Graph capture는 초기화에서 준비하며 step 측정은 새로운 준비 비용까지 전부 포함한다.

## Profile OFF 전체 step 직접 비교

모든 표는 같은 입력·seed·코드 조건에서 **전체 출력 및 source/hit/valid-k/accepted length 일치**를 먼저 확인했다. 단위 ms. 표의 평균은 양 arm에 공통으로 포함된 step의 평균이다.

| 비교 | P1 hit 이전→이후 | P2 hit 이전→이후 | Miss 이전→이후 | 전체 이전→이후 | 감소 |
|---|---:|---:|---:|---:|---:|
|응답+glue 초기 통합, 3 process 쌍|18.943→18.582|18.936→18.574|20.529→20.676|19.129→18.829|1.57%|
|최종 통합, 12문항×2|18.870→18.447|18.874→18.434|20.396→20.297|19.054→18.662|2.06%|
|최종 통합, 별도 48문항/56턴|19.037→18.427|19.110→18.527|20.684→20.392|19.244→18.681|2.93%|

최종 12문항 비교 감소 CI95=1.88~2.20%, 별도 48문항=2.65~3.21%. 문항 단위 bootstrap이며 MT의 두 turn은 같은 문항 cluster다. Process 전체의 clock/환경 이동을 이 CI가 제거해 주지는 않는다.

같은 엔진/graph/allocator 안에서 초기 통합 경로를 무작위 전환한 진단도 진행했다. UID/pass/source/valid/context128/직전 source를 맞추고 현재·직전 모드를 함께 추정했다. 현재 step 효과는 -0.354ms(CI95 -0.376~-0.333), 직전 효과까지 -0.359ms였다.
이 진단은 초기 통합(v1) 기준이다. 최종 요청 bulk/miss 재사용까지의 동일 엔진 추정은 아니다. 48개 measured request 중 동일 pass 수의 reference가 있는 처음 24개의 출력·event를 정확히 대조했다. 뒤 24개를 앞 pass의 출력으로 대체해 비교하지 않는다. 별도 draft process RNG가 pass 사이에 유지되어 reference 자체도 pass별 출력이 다르다. `CONTROL_PARITY_SCOPE.json`에 범위를 기록했다.

초기 통합의 miss는 같은 엔진에서 약 +0.145ms 비용이 관찰됐다. 이를 숨기지 않고 중복 lookup 재사용을 추가했다. 같은 pair에서 최종 버전은 초기 통합 대비 전체 18.712→18.662ms, miss 20.529→20.297ms였다. 추가 묶음의 순효과는 전체 약 0.050ms이며, hit의 추가 이득은 작다. 이 추가 묶음을 큰 개선으로 주장하지 않는다.

별도 48문항에서는 target call+accept 평균도 17.381→17.297ms로 이동했다. 모델 그래프 자체를 바꾸지 않았고 일부 process 환경 변동/기존 prep 효과가 섞일 수 있다. 따라서 2.93%를 순수 header 최적화만의 효과로 단정하지 않는다. 반복 A/B와 같은 엔진의 약 0.35ms 감소를 주된 인과 근거로 본다.

## 최종 코드의 breakdown

Profile ON, 동일 입력/실행 설정의 직접 관측값. 서로 중첩되는 구간이 있으므로 이 표의 감소분을 합해서 전체 step 감소로 쓰지 않는다.

| 구간 | P1 hit 이전→이후(ms) | P2 hit 이전→이후(ms) |
|---|---:|---:|
|Draft hit 응답 준비|0.757→0.319|0.763→0.321|
|Draft glue|2.099→1.110|2.205→1.106|
|Target response wait|1.131→0.705|1.147→0.701|
|Target request send|0.228→0.134|0.230→0.133|
|다음 DS 응답 사전 준비|0.110→0.268|0.104→0.273|
|다음 PS 응답 사전 준비|0.058→0.130|0.062→0.132|

Target model pre+post는 15.308→15.268ms다. 모델 그래프를 변경하지 않았으며 이 작은 차이를 Transformer 가속 효과로 주장하지 않는다. Glue 감소의 상당 부분은 target과 겹쳐 실행되고 사전 응답 준비 비용도 늘었다. 전체 step 효과가 component 감소 합계보다 작은 이유다.

[비교 그래프](BREAKDOWN.pdf), [PNG](BREAKDOWN.png), [원시 요약](PROFILES.json).

## 기능·재현 검증

- 최종 worktree runtime은 frozen `fused_bulk` snapshot과 전체 Python 파일 hash가 같다. Portable CPU/CUDA 회귀 27개 PASS, skip 0. 별도 실제 miss entry point의 이전/이후 JIT 출력·RNG 회귀 1개 PASS.
- CPU 검사는 packet layout/padding, duplicate cache의 첫 hit 선택, P1/P2 root offset, dummy RNG 상태, ancestry 행렬, block table/float32 온도 wire bytes를 확인했다. 실제 모델 비교는 GPU token 출력과 전체 검증 event를 대조했다.
- 초기 CPU 검사 실행 한 번은 필수 환경변수 누락으로 import 실패했다. `CPU_TESTS.log`에 보존하고 환경을 설정한 재검사 결과를 별도로 남겼다. Runtime 오류로 숨기거나 최종 통과 기록에 섞지 않았다.
- Frozen snapshot에 잘못 추적된 `__pycache__/*.pyc`가 있어 일부 run의 `paper_dirty`에 compiled cache만 나타난다. Python source/benchmark는 매 run 해시로 고정했고 변경 없음. Dirty 원문을 지우거나 clean으로 바꾸지 않았다. 최종 `fused_bulk` snapshot은 compiled cache tracking을 제거해 clean이다.

| 실행 | measured 요청 | long step 제외 | cap/clip 제외 | 최대 외부 포함 process/GPU |
|---|---:|---:|---:|---:|
|fused_p0_smoke|3|0|3|1|
|base_fused_p0_a|24|0|20|1|
|fused_p0_b|24|0|20|1|
|fused_p0_c|24|0|20|1|
|base_fused_p0_d|24|0|20|1|
|base_fused_p1_a|24|0|20|1|
|fused_p1_b|24|0|20|1|
|mixed_fused_control|48|0|37|1|
|heldout_fused|56|0|36|1|
|heldout_base_fused|56|0|36|1|
|fused_profile|12|0|9|1|
|baseline_fused_profile|12|0|9|1|
|fused_bulk_reference|24|0|20|1|
|fused_bulk_candidate|24|0|20|1|
|heldout_fused_bulk|56|0|36|1|
|fused_bulk_profile|12|0|9|1|

비정상 long step은 기존 사용자 승인 2× 기준, cap/clip 및 부분 batch 제외 정책을 유지한다. 제외 시 시간과 토큰을 같이 제외한다. Profile ON에는 long step 필터를 적용하지 않는다. Raw를 보존했고 원시 시간·token 재집계와 독립 wall observer 일치를 확인했다.

별도 48문항의 AL=2.602068, hit=89.5346%는 양 arm 동일하다. 포함 step=4061, 실제 반환 토큰=10659, 포함 step의 반환 토큰=10567이다. 문항 평균 AL이나 논문 full dataset 결과로 대체하지 않는다.

## 현재 반영과 인수인계

`/home/chokwans99/PSD-local-followup-20261017`, branch `exp/duet-local-followup-20261017`에 최종 통합 코드와 검사를 반영했다. Production `MLSys-ver1.3`과 이전 response-async 브랜치의 runtime은 그대로다. 사용자 지시대로 push하지 않았고, 공유 브랜치의 commit도 만들지 않았다. `/data/.../code`의 source-only commit은 실험 provenance용 local snapshot이다.
전체 baseline은 기존 prepack `1dcb10ba`다. 이전 eslab18의 1.43%나 앞선 0.04ms와 이번 비율을 합산하지 않는다. Runtime 변경은 기존 prep 3파일 + draft_runner/speculator_async + 새 helper의 6파일이다. 연구 점수/정책을 추가하지 않았다.

`COMPARISON.json`의 `adopted_candidate`는 상속된 분석 helper의 기계적 조건(최소 2 process 쌍)이다. 단일 비교인 `final_12`/`heldout_final`에서는 false이며, 둘을 묶은 `selected_joint`에서 true다. 이 필드만으로 인과관계나 새로운 환경의 채택을 보증하지 않는다. 최종 선택은 별도 입력·반복 비교·동일 엔진 진단·기능 검사에 근거했다.

검토 순서는 `COMPARISON.json`(parity/시간) → `MIXED_ANALYSIS.json`(동일 엔진) → `PROFILES.json`(component) → `VALIDATION.json`/`SELECTED_TESTS.log` → `FUSED_BULK_RUNTIME.patch`(최종 전체 patch)다. `FUSED_RUNTIME.patch`는 초기 통합, `MIXED_FUSED_RUNTIME.patch`는 진단용으로 production 채택 대상이 아니다.
실험 도구와 계획은 `reproduce/`, raw run은 `RAW_RESULTS.tar.xz`, source별 hash는 `SOURCE_MANIFEST.json`이다. Local runner의 GPU/CPU pin/모델 경로/포트는 서버에 맞춰 바꾼다. Raw와 provenance를 덮어쓰지 않는다.

현재 결과가 뒷받침하는 것은 이 7B/68m, B1, T=0.7, G=M 조건의 실행 비용 감소다. B>1·다른 모델·다른 tree budget·greedy 통합 성능과 1024토큰/full benchmark는 이번에 실행하지 않았다. 더 큰 가속을 주장하려면 target model forward 자체 또는 다른 병목에 대한 별도 변경과 측정이 필요하다.
