# B1 PS preparation 비용 검증

- 기준: MLSys-ver1.1 0916871, frozen checkout `/data/chokwans99/duet-ps-prep-20261015/baseline`.
- 실제 모델: LayerSkip Llama2 7B + AMD 135M, BF16, B1, T0.7, exit20, K1/K2=6/3, M1/M2=12/8, C3, DS width26, PS width15/root10, frontier.
- RTX4090 GPU4/5 동일 pair에서 전후 역순 반복. CPU NUMA1. 기존 그림의 3090/68M과 절대시간 비교하지 않음.
- native CUDA/CPU event 기록으로 phase2_build, dispatch gap, p2_prepare 2개, p2_graph_replay, merge, PS 전체 시간 비교. 부모/자식 event 중복 합산 금지.
- 입력/출력/AL/hit 및 런타임 source hash 보존. warmup/capture는 성능 표본에서 제외. cold prefix 양 arm 동일.
- 모델 검사는 전체 데이터 성능 평가와 구분한 실행 비용 실험. 동일 SpecBench 24턴, 출력96, 입력512, warmup4. 이후 48턴 재검증.
- 채택 조건: 동작/검증 등가, 준비 시간 개선, PS 전체로 비용 전가하지 않음.
