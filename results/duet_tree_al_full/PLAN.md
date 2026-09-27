# Full Spec-Bench: root 정책 고정, tree AL 평가

2026-09-21 사용자 지시: root 후보 선정의 목적은 cache hit, tree 구성의 목적은
AL 향상이다. TPS는 이번 tree 정책 선택의 목적함수로 쓰지 않는다.

## 고정할 것과 바꿀 것

기존 q-path tree와 calibration reach tree를 비교한다. Root 후보 수식/source,
exit56, K1/K2=4/2, root 수, C=3, P1/P2 node cap=8/6을 동일하게 유지한다.
각 phase에서 G=M이다. 기존 scalar calibration을 재학습하거나 이 데이터에서
계수를 다시 고르지 않는다. 같은 예산에서 tree 점수만 바꾼다.

실행 코드로 확인한 예산 표기 보충: `tree_root_count=15`는 P2 설정이고 P1은
context당 root 3개다. P1 context bucket 3/7/9의 round 폭은 각각
(9,9,9,9), (21,15,15,15), (27,15,15,15)다. 두 정책이 같은 예산 규칙을
사용한다는 뜻이며, 서로 다른 생성 경로에서 실현된 총 root/node 수가 정확히
같다는 뜻은 아니다. 동결된 실행 설정과 주 지표는 바꾸지 않았다.

Root 집합을 C, 실제 root 사건 질량을 pi라고 하면

\[
H(C)=\sum_{r\in C}\pi_r,\qquad
AL_{\mathrm{hit}}=\frac{\sum_{r\in C}\pi_r E[A(T_r)]}{H(C)}.
\]

Root 정책과 C를 고정하면 분모는 tree 확장 최적화에서 상수다. 따라서 root
prior를 tree 배분에 쓰는 것은 hit 후보를 새로 고르는 것과 다르며, 적중할
가능성이 큰 root 아래 AL에 예산을 배분하는 역할이다. 실제 독립 생성에서는
context와 hit mix가 달라질 수 있으므로 P1/P2별 조건부 AL을 함께 평가한다.

## 데이터와 생성 계약

- [공식 Spec-Bench](https://github.com/hemingkx/Spec-Bench)의 480문항 전부,
  6개 과제 각각 80문항. 총 560턴이다. 로컬 원본과 공식 파일의 문항·turn 내용
  일치를 확인하고 원본과 hash를 보존했다.
- 이번 1차 확대는 이전 score 실험과 같은 **출력 상한 128**이다. Full은 문항
  범위 전체라는 뜻이며, 논문의 1,024-token cap 실험을 완료했다는 뜻이 아니다.
- EOS를 무시하지 않는다. 자연 종료 이후 token으로 AL을 부풀리지 않는다.
  이 점은 이전 128-token 강제 생성 smoke와 다르므로 절대 수치를 합치지 않는다.
- 두 번째 turn은 첫 turn의 실제 입력·응답을 포함한다. 과거 준비 파일처럼
  두 번째 질문만 독립 요청으로 취급하지 않는다. Base checkpoint의 raw token
  형식을 사용하며 공식 Vicuna chat-template 실행의 복제라고 주장하지 않는다.
- 입력 삭제/절단 없음. Native 2048 context에 모든 입력·출력 cap·tree reserve가
  들어가는지 사전에 검사했다. 최대 보수적 필요 길이는 plan에 기록한다.
- Seed 1,42,123. 각 seed에서 양 정책 전체를 실행한다. 총 3,360턴이다.
- 별도 neural head 학습 없음. 기존 calibration 8 prompts와 exact text overlap=0.
  기존 연구에서 사용했던 benchmark이므로 globally unseen data라고 부르지 않는다.

## AL의 정의와 신뢰도

Primary AL은 **hit tree에서 수락된 descendant 수**다. 이미 확정된 correction
root와 recovery/bonus는 제외한다. EOS나 max length로 잘린 마지막 verification
step을 제외한 결과를 주 지표로 하고, 모든 step을 포함한 민감도도 함께 낸다.

먼저 각 과제의 hit-conditional AL을 계산하고 6개 과제를 동일 가중한다.
P1/P2별 AL, phase를 50:50으로 고정한 보조 AL, accepted-length histogram과
tail 확률, question macro 평균을 함께 보고한다. Hit가 없는 문항의 conditional
AL은 0으로 임의 대체하지 않는다. 전체 문항 수에는 포함하고 missing 개수를 공개한다.

Bootstrap 단위는 original question이며 두 turn과 세 seed를 같은 cluster에
묶는다. 각 과제 안에서 paired 5,000회 재표집한다. 과제별·seed별 변화도 공개한다.
정책별 incomplete run 또는 누락 UID가 하나라도 있으면 최종 추론을 거부한다.

TPS는 진단용으로 저장하지만 AL 우위를 TPS 기준으로 기각하거나 선택하지 않는다.
AL 우위가 확인되어도 고정 depth/node 예산 밖의 전역 최적 tree를 찾았다고
주장하지 않는다. 이번은 앞서 제안한 score의 full-corpus 검증이다.

## 실행과 재현

`plan.json`은 결과 확인 전에 동결한다. `launch.py --smoke`는 별도 파일에
모든 task와 최장 입력을 점검한다. 본 실행은 `launch.py`, 중간 누락 검사는
`analyze.py --partial`, 완료 후 분석은 `analyze.py`다.

분포 저장/추가 verifier observer는 없다. Draft의 KV 메모리 비율만 실험 process
hook으로 줄여 전체 page bucket의 graph를 준비한다. 계산 내용과 root 정책은
그대로다. 완료 run은 checksum 검사 후 건너뛰고, 실패한 partial run은 자동
덮어쓰지 않는다. 결과 보고서 작성 시 실행 완료 수를 명시한다.
