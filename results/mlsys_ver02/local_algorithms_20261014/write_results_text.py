"""Write the completed primary result section from frozen aggregate evidence."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'RESULTS.json').read_text());assert r['status']=='complete'
lines=['14개 primary arm이 모두 정상 종료했고 각 arm은 480문항 × 2 pass, 960개 출력을 기록했다. 외부 GPU process가 관측된 원래 Llama3 두 실행은 제외하고 정상 재실행을 선택했다. 전체 표·신뢰구간은 [TABLES.md](TABLES.md), 숫자는 [RESULTS.json](RESULTS.json)에 있다.','','| 모델 | 변경 | Batch | AL | Cache hit (%) | TPS (보조) |','|---|---|---:|---:|---:|---:|']
for model,title in [('l2','Llama2-7B / AMD-135M'),('l3','Llama3-8B / Qwama-0.5B')]:
 for name,label in [('base','기존 ver0.2'),('root_only','Root만'),('tree_only','Tree만'),('combined','Root+tree')]:
  s=r['models'][model][name]['summary'];lines.append(f"| {title} | {label} | 8 | {s['al']:.4f} | {s['hit']*100:.3f} | {s['tps']:.2f} |")
lines+=['','### 판정','','1. **Tree의 P1 AL 개선 근거는 있다.** 기존 root를 유지하고 tree만 바꾸면 P1-hit AL은 Llama2 2.641→2.775, Llama3 2.837→2.999다. 차이의 문항 bootstrap 95% CI는 각각 `[+0.052,+0.218]`, `[+0.090,+0.235]`다. 총 AL은 각각 +0.058, +0.048이며 Llama2의 총 AL CI는 0을 포함한다. Llama3의 총 AL CI 하한은 +0.0003으로 작으므로 강한 보편적 우위라고 과장하지 않는다.',
'2. **Tree는 같은 cap을 더 활용했다.** P1 hit tree의 실제 node 평균은 Llama2 7.272→7.748, Llama3 7.175→7.680, 최대 깊이는 3.563→3.744 / 3.505→3.689다. 따라서 이번 실험은 “동일 실제 node 수에서 점수만 더 정확해졌다”는 증명이 아니다. Reach/gain/frontier 묶음이 남은 예산을 더 쓰며 구조를 바꾼 효과이고, 각 tree 구성 요소의 기여도는 별도 ablation이 필요하다.',
'3. **Root 변경의 cache hit 우위는 확인되지 않았다.** 기존 대비 root-only hit 차이는 Llama2 +0.834%p, Llama3 +0.272%p이며 두 CI 모두 0을 포함한다. Full 정규화만 추가하거나 `.25 hbar`를 추가한 단계에서도 보편적 개선은 확인되지 않았다. 특히 위치 혼합의 점추정은 두 모델에서 hit·AL 모두 낮아졌다. `.25`를 범용 최적값으로 주장하지 않는다.',
'4. **e(1-q)가 proxy-only를 이겼다고 쓸 수 없다.** 위치식·full 정규화를 맞춘 비교에서 hit 차이는 Llama2 −0.006%p `[−1.582,+1.516]`, Llama3 +0.775%p `[−0.695,+2.373]`다. 두 모델 모두 확정적인 우위가 없고 AL 점추정은 낮았다. 이것은 Mirror-SD 전체 시스템 비교가 아니라 DUET root 토큰식의 비교다.',
'5. **Root+tree 전체 조합은 모델 의존적이다.** 기존 대비 Llama2 AL은 +0.0987 `[+0.0295,+0.1695]`로 증가했다. Llama3 AL 차이 +0.0192는 CI가 0을 포함하고 hit는 −1.936%p `[−3.876,−0.161]`로 낮아졌다. 따라서 이 조합을 모든 모델에서 검증된 개선 기본값이라고 제시하지 않는다.',
'6. **이번 branch에는 요청한 연구 조합을 실제로 연결했다.** 구현 완료와 최종 논문 정책 채택은 구분한다. 코드의 운영 연구 preset은 complement/full/.25/reach-gain-frontier 한 조합이다. 최종 논문 채택의 근거는 현재 tree의 AL 방향이 더 강하고, root·위치 조합은 추가 검증이 필요하다. 다른 서버 결과를 합칠 때 이전 정책과 이 조합을 재비교할 수 있도록 2×2/분리 비교와 원시 자료를 함께 남긴다.',
'','### 그림과 해석','','- [2×2 비교 PNG](comparison.png) / [PDF](comparison.pdf): AL·hit·TPS.','- [분리 비교 CI PNG](ablations.png) / [PDF](ablations.pdf): 정규화, 위치, 토큰식, tree 추가의 변화.','- P1/P2 사용 비중과 조건부 AL의 대칭 분해는 RESULTS의 `al_accounting`에 있다. 이는 총 AL 차이를 정확히 나누는 집계 항등식이며, 같은 문맥에서 원인을 개별 조작한 인과 추정은 아니다.','- 실제 요청의 전체 문항 평균도 TABLES에 있다. 주 표는 B8/cap/clipping 제외 step 평균이고, 보조 표는 각 문항의 tail까지 포함하므로 가중치와 값이 다를 수 있다.','']
p=HERE/'REPORT.md';s=p.read_text()
start=s.index('## 3. 결과')+len('## 3. 결과')
end=s.index('**TPS 해석 주의:**',start)
s=s[:start]+'\n\n'+'\n'.join(lines)+'\n\n'+s[end:]
p.write_text(s)
