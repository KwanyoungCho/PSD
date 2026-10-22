# ver1.7 추가 오류 검토

기준 `4c7e43382db6215166265ed1f19d0a5d6bdd72b3`와 최초 ver1.7 P1 수정
`5a881cc2`를 바탕으로 검토했다. 기존 원시 결과와 런타임 checkout은 보존했다.

## 판정과 반영

1. **RNG: 동일 난수열 시작 재현.** 기존 constructor는 모든 executor에 동일
   초기 seed를 사용했다. CUDA production WOR sampler의 uniform 2-token 실험에서도
   두 generator의 결과가 전부 같았다. Phase/context/lane를 포함한 안정적 64-bit
   seed 파생으로 분리했다. CUDA graph 등록·warmup 상태 복원·반복 replay 시 offset
   진행을 검사했다. 실제 LLM 출력의 편향 크기나 일반 무손실성의 통계적 증명으로
   toy 검사를 확대 해석하지 않는다. 새 seed 구분은 초기화 때만 적용한다.
2. **점수 전송: 오류 재현.** 0, 1e-12, 1e-9, 1e-8이 모두 약1e-6으로 복원됐고
   0 root가 NV=8의 예산을 받았다. 동일 int64 payload 안의 여유 비트를 써서
   비음수 FP32 score를 그대로 전송하도록 수정했다. 0·subnormal·작은 양수·큰
   vocabulary token ID를 bitwise roundtrip 검사했다. Selector는 zero padding을
   제외하며 executor의 child budget은0이다. Root prior의 별도1e-9 floor도
   제거했다. 기존 raw-q/ordered-WOR/residual 수식은 변경하지 않았다.
3. **Scheduler: 요청 누락 재현.** A를 선택한 다음 B가 문맥 보호로 종료하면
   A가 실행 큐에서 사라졌다. 조기 반환 전에 선택한 요청들을 순서대로 복원한다.
   연속 보호 종료·마지막 요청·AR/SD 경로와 자원 소유를 검사했다.
4. **관련 B>1 P1 경로:** 통합 회귀에서 batched root selector에도 옛날의 다음
   token 하나 제외가 남은 것을 발견했다. B1과 같은 직접 자식 제외 규칙으로
   통일하고 fused kernel을 배치 전체에 한 번 실행한다. Ragged topology를 바꿔
   CUDA graph를 재생해도 CPU reference와 일치함을 검사했다.

## 검증

`CHECKS_RELEASE.log`: 20개 suite, 287개 CPU/CUDA/graph 검사 통과.
`REPRODUCTION.json`: 기존 scheduler와 production sampler의 재현 수치.
`OVERHEAD.json`: 기존 wire 산술을 AST로 그대로 읽어 새 codec+selector와 비교했다.
개발 중 실패 로그도 보존했으며 `CHECKS_RELEASE.log`가 최종 통합 검사다.

RNG 공식 동작 설명은 [NVIDIA CUDA Graph RNG 안내](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/handling-dynamic-patterns.html)를 참고했다.
본 판단의 직접 근거는 현재 설치된 PyTorch/CUDA에서 실행한 재현과 회귀 검사다.

## 추가 overhead

RTX4090, CPU affinity 고정, eager codec+selector 16블록×100호출 교대 A/B.
모델 실행 시간이나 전체 step 시간이 아니다.

| Batch / wire 개수 | 이전 µs | 수정 µs |
|---|---:|---:|
| 1 / 120 | 377.97 | 312.46 |
| 1 / 256 | 489.04 | 407.64 |
| 16 / 120 | 480.14 | 396.09 |

이 구간의 CUDA kernel 수는 B1 60→49, B16 61→50이다. 기존 version 검사에
쓰던 DtoH1회는 그대로이며 추가 전송은 없다. NCCL byte 수·호출 수는 그대로다.
RNG 분리는 replay 연산을 추가하지 않는다. Scheduler 변경은 보호 종료에만 실행한다.

별도 batched P1 준비 검사(BF16, V32000, P24, U4, 12블록×80회):

| Batch | eager 이전→수정 µs | graph GPU 이전→수정 µs |
|---|---:|---:|
| 1 | 628.43→598.52 | 117.74→97.90 |
| 16 | 444.89→407.52 | 378.86→359.74 |

입력/shape가 같은 전후 비교다. 각 형상의 GPU clock/온도가 서로 달라질 수 있어
행 사이 B1/B16 절대값을 scaling 측정으로 사용하지 않는다. 원시 분산은
`BATCH_ROOT_OVERHEAD.json`에 있다. 이 작은 구간의 감소가 전체 TPS 향상을
보장하지는 않으며, 후보·난수열이 바뀐 전체 성능은 별도 재측정한다.

## 성능 재측정 조건

기존 best-3 2·0·10을 고정하고 원래 실행에 저장된560턴의 input token ID·순서를
그대로 사용한다. 최근 다른 진단의 재토큰화된 입력을 섞지 않는다.
BF16/B1/exit21/K1=10/K2=5/P1 W56·위치당4·G=M23/P2 R24·W28·G=M15/C3,
Ttarget=Tdraft0.7/top-p1/문맥2048/출력1024/자연EOS/기존문맥보호/첫3턴 제외.
로컬 PIX(1,2)/(4,5)/(6,7)에서 seed당 같은 GPU쌍으로 v1.6과 수정판을 비교한다.
기존 v1.4 논문 결과도 별도 열로 보고한다. Profile은 끈다.

문항 평균·P1/P2/Miss 비율/AL/step/TPS, 전체 지표와 제외 건수를 보고한다.
공통 유효 문항 비교와 원래 논문 표 방식의 available-case·필터 없는 수치를
구분한다. 전후 총6회가 완료되고 모두 검증을 통과했다.

[Full best-3 비교](../best3_20261023/REPORT.md): 공통367문항·422턴에서
step19.2494→19.2585ms(+0.05%), TPS171.58→169.47(-1.23%),
AL3.325→3.286, hit92.07→92.28%다. 직접 수정한 작은 구간의 비용 감소를
전체 속도 향상으로 해석하지 않는다. 이 세 seed에서 전체 TPS 개선은 없었다.
기존 논문 표 방식의 필터 없는 실행별 문항 평균은168.58→167.27 TPS다.
