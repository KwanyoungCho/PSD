# 22 — T6-dynamic: 고정 크기 GPU arena rollout (설계)

2026-08-04. 근거: 21번 §4.7 분해 (+23.5ms/step 중 ~86%가 host 재조립)
+ 리뷰3 판정 (20번) — budget-static은 현 raw_q-적응 정책의 drop-in이
아니므로 (실형상 반례: q 회전 시 40중 18 재배치), **의미를 보존하는
GPU화**가 1차 트랙. budget-static은 별도 정책 arm (P2AL 보존 검증
조건부).

## 목표

P2AL +18% (재판정 §5-re)를 그대로 두고 시간만 회수한다.
- core-입증 회수: P2 창 내부 idle 9.2ms (0.27 대비)
- stretch: build→merge 전체 Δ22ms 중 GPU화 가능분 (구현 후 측정)
- 명시적 비목표: plan-once (#20 붕괴 재도전 금지), fixed topology

## 설계

### A. Arena (고정 크기 GPU 상주 상태)

capacity = R + F·W·C 고정. 전 필드 GPU int64/float32 텐서:
`active/tok/parent_idx/parent_cell/root/depth/sib/logpri/raw_q/cell`
+ `n_nodes` GPU 스칼라 (cumsum으로 갱신 — readback 금지).
tip_idx[R]·remaining[R]·reserve[R]도 GPU.

### B. per-forward 루프 (python K2회 유지, 내부 전부 텐서 op)

1) **select**: elig = (state==0)&(depth==f, level) → tip 의무 lane
   (#27 유지: remaining>0 root의 tip 먼저) + 잔여 lane stable
   argsort(logpri) top-W. sel [W] GPU (pad=-1).
2) **fanout**: backbone 규칙 (tip 1 + rescue priority-라운드) —
   W·R≤10이라 dense [W,R] 마스크 연산으로 전개 (순차 의존은 C≤3
   라운드 unroll).
3) **mask**: 조상 bitset [W, F·W] GPU 증분 갱신 (자식 = 부모 bitset
   | 자기 셀 비트) → uint8 packbits 동형 비트연산 → captured
   `_custom_mask_buf`에 **in-place copy** (GPU→GPU; 리뷰3-6 —
   포인터 교체 불가). prefix/글루 열은 요청 시점 상수라 사전 조립.
4) **forward**: 기존 per-forward FlashInfer plan **유지** (parity
   먼저 — plan-ahead는 후속 독립 A/B). input_ids/rope는 arena
   gather.
5) **sample+insert**: tree_sample_wor GPU 그대로 → 자식 삽입을
   cumsum(fan) offset scatter로 (raw_q≤0 배제 #38은 mask로).
   logpri = gather(parent)+log(raw_q). **readback 0회.**

### C. 종료부 (view/wire)

- build_root_views 대체: [R,Nv,V] parent-q 물질화 금지 (리뷰3-12) —
  `cell_logits[F·W, V]` + `parent_cell[R,Nv]` 참조 유지, 서빙 시
  hit root만 gather.
- pack_tree_ints GPU화 → wire 버퍼 직행. 필요 sync는 응답 경계
  1회 이하.

### D. target측 병행 (critical path = max(draft, target) — 리뷰3-5)

- graph_pre Δ+7.4ms: 트리 verify CG의 mask copy/컨텍스트 교체 구간
  프로파일 후 축소 (별도 이슈로 계측부터).
- Policy-B 잔여 6.2ms: 사다리 커널 융합 또는 CG화.

### E. 검증 게이트 (순서 고정)

1. stub-forward 동일-시드 **topology 동등성**: arena vs 현 CPU
   rollout (교체 전 회귀 기준). RNG 소비 순서 보존 확인.
2. C=1 체인-퇴화 byte-parity (기존 게이트 재실행).
3. alloc 불변 (requested/allocated/generated — #39 3값).
4. E2E 스모크 → 동일-시드 A/B (P2AL 보존 ±노이즈 이내 필수).
5. 성능 판정: component 합산 금지 — 3연속 구간 wall
   (① target proxy gate ② P2 first-prep→last-replay ③ gate→merge).

## correctness 부채 (arena 전 처리 — 리뷰3-10)

- [x] WOR support (#38), 양단 temp 게이트 (#37), assert 경화 (#40)
- [ ] wire epoch 상수 1 → seq 재진입 카운터 (staging seq 가드 #35와
  통합)
- [ ] SHM read-ACK vs GPU-완료 ACK: B=1·순차·단일 stream 가정에서만
  안전 — arena가 stream을 늘리면 15번 §ACK 계약대로 승격 필요
- [ ] W10-top6 chain 대조군 knob (시간축 엄밀 분리용)
