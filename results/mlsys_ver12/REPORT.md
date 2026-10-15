# MLSys ver1.2 반영 결과

2026-10-15 KST. 기준 `MLSys-ver1.1@0916871`, 채택 runtime `9390e21`.
이 문서는 적용 상태의 정본이다. 개별 실험의 당시 소스·조건·보고서는 그대로 보존했다.

## 반영 판정

**효과가 검증된 PS 준비 최적화만 채택했다.** 후보 선택·위치 확률·tree 확장 수학과
파라미터를 유지한 채 같은 작업의 실행 비용을 줄였다.

| 구간 | 이전 ms | 채택 후 ms | 변화 |
|---|---:|---:|---:|
| PS 전체: proxy 수신 완료→cache merge 완료 | 5.269 | 4.928 | **−6.48%** |
| PS 준비 네 구간 합 | 1.947 | 1.607 | **−17.47%** |
| PS root/layout | 0.877 | 0.771 | −12.00% |
| Slot·상태 검사·dispatch 간격 | 0.638 | 0.417 | −34.65% |
| PS 준비 A | 0.193 | 0.202 | +4.65% |
| PS 준비 B | 0.239 | 0.216 | −9.45% |
| PS graph: forward+tree 선택 | 2.861 | 2.856 | −0.19% |
| Target forward | 15.522 | 15.514 | −0.05% |
| DS graph | 6.591 | 6.581 | −0.16% |

전체와 부분행은 중첩되므로 표의 모든 행을 합산하지 않는다. 준비 A가 늘어난 것도
포함하여 PS 전체 순감소로 판단했다. 작은 graph/target 변화는 가속 효과로 주장하지 않는다.
**PS 차이95% 구간은 −0.390~−0.290ms**다.

측정은 Llama2-7B+AMD135M, RTX4090, B1, BF16,T0.7, exit20,K1/K2=6/3,
G=M=DS12/PS8,C3,DS 입력26,PS 입력15/root10,frontier의 고정 설정이다.
24턴×3쌍의2,391개 대응 step, 통계 기여22문항이다. 출력72/72, source·AL·hit 동일.
별도 profile OFF48/48 출력도 동일하며 graph 준비 영향 step0이다.
SpecBench full AL 평가 또는 모델별 최적 파라미터 탐색으로 해석하지 않는다.

실험 조건·반복·telemetry·CI·정확한 수치는
[채택 실험 보고서](../mlsys_ver11/ps_preparation_20261015/REPORT.md),
[구간별 전후 그림](../mlsys_ver11/ps_preparation_20261015/BREAKDOWN_COMPARISON.pdf),
[같은 UID/step 전후 native 그림](../mlsys_ver11/ps_preparation_20261015/native/NATIVE_BEFORE_AFTER.pdf)에 있다.

## 코드 변경과 남은 비용

1. **Root 매핑 재사용:** 같은 `repeat_interleave` 계산을 다시 하지 않고 `fan_idx_hit`를 재사용한다.
   남은 연산에는 이미 아는 `output_size`를 제공한다.
2. **Tree 조상 마스크 GPU 유지:** GPU→CPU NumPy→GPU 왕복 대신 GPU에서 직접 executor에 복사한다.
   Eager/fallback·진단에서 필요한 NumPy 계약은 해당 경로에서 유지한다.
3. **여섯 metadata 검사 통합:** 여러 작은 Torch kernel을 Triton integer kernel 하나로 묶는다.
   같은 실패 조건·우선순위·root/padding 처리를 유지하고 **CPU 결과 readback 한 번은 남는다.**

두 번째로 시도한 fan-out readback 지연·host 문맥 길이 재사용은 **미채택**이다.
PS4.965→4.948ms의 차이95% 구간−0.050~+0.028ms가0을 포함했다.
[그 후보 보고서](../mlsys_ver11/ps_device_20261015/REPORT.md)의386개 검사와 patch는
현재 runtime에 반영한 내용이 아니다. 현재91개 runtime Python 파일의 해시가
앞서382개 검사를 통과한 채택 소스와 동일하다.

남은 동기화·metadata 복사·slot 준비·merge는 존재한다. 모든 CPU–GPU 왕복을 없앤 것이 아니다.
새 모델/70B/B>1에서 동일한 이득을 단정하지 않으며, 다른 서버 알고리즘은 병합하지 않았다.

## 상세 timeline과 연속 두 step

**[모델별 timeline 보고서](model_timelines/REPORT.md)**에서 확인한다. 기존 여러 형식의
모델별 그림·HTML189개를 삭제하고, 모델 조합당 PDF 하나와 [전체17쪽 PDF](model_timelines/ALL_MODELS.pdf)로 통일했다.
5개 모델 조합이며 CodeLlama의 두 설정은 같은 PDF 안에서 구분한다.

각 페이지는 같은 요청의 연속 두 step이다. 빨간 점선은 다음 target 요청 시작,
초록 원은 PS cache 완성, 삼각형은 draft가 후처리를 끝내고 다음 명령을 받을 준비가 된 시점이다.
Ready margin은 관측된 차이이고, 전부 추가 forward 예산으로 사용할 수 있다는 보장은 없다.
다음 target response wait 전체를 이전 PS 지연으로 해석하지 않는다.

흰 점선은 DS/PS forward 횟수이며 실측 개별 forward 경계가 아니다. 준비·변환·merge는
기존 상세 색/무늬로 유지했다. 미계측 구간을 GPU idle이라고 단정하지 않는다.
원시 시각을 이동하거나 최소 막대 폭으로 늘리지 않았다.

자료는 기존 profile을 다시 그린 것이며 새로운 inference를 수행하지 않았다.
3090/ver1.0 계열의 모델별 측정과4090/1.2 채택 runtime의 최적화 검증 설정을 구분했다.
다른 서버에서는 [도구 사용법](../../ssd/tools/duet_timeline/README.md)에 따라 portable JSON으로 재생성한다.
[삭제 목록](model_timelines/CLEANUP.json)에 이전 파일과 해시를 기록했으며 raw 데이터는 보존했다.

## 검증과 재현성

- Runtime: 기존382개/40suites, 실패·오류·skip0의 소스와91개 파일 동일. 이번 릴리스에서 새 GPU 실행을 했다고 주장하지 않는다.
- Timeline/분석: **31개 CPU 검사 통과**. 요청 경계 혼동, 누락된 다음 step, 잘못된 command 연결,
  마감 초과의 음수 값, cache/수신 준비 구분, 원시 이벤트 보존 및 PDF 생성을 검사한다.
- 최초 전달의 두-step16쪽의 PDF 페이지 수, raw window hash, 정확한 deadline 산식, forward 구간 보존을 확인한다.
- 새 최적화 예시는 portable JSON만으로 원본 경로 없는 재생도 확인했다.
- [최초 전달 검사](checks/CHECKS.json)는 당시 기록이며 현재 그림은 [모델별 모음 검사](model_timelines/checks/CHECKS.json)를 따른다.

다른 서버는 각 PNG 옆 JSON으로 다시 그릴 수 있다. 기존 원시 profile은 수정하지 않았다.
채택/미채택 실험의 `RAW_RESULTS.tar.gz`와 SHA manifest도 함께 전달한다.
Runtime·tool 변경은 별도 커밋이며, 최종 변경 검토 및 fetch 명령은
[인수인계](../../MLSYS_VER12_HANDOVER.md)를 따른다.
