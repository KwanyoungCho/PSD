#!/usr/bin/env python3
"""One current two-step timeline PDF per model pair, plus one combined PDF.

Rebuilds from portable, hash-pinned raw windows. This does not run inference or
relabel historical measurements as the current runtime's performance.
"""
import argparse
import html
import json
import os
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

from plot_consecutive_steps import render
from plot_model_breakdowns import load, require, sha, write_csv, STATES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=args.manifest.resolve();out=args.output.resolve()
    spec=load(manifest);require(spec['schema']=='duet-model-timelines-v1','Unknown model manifest')
    out.mkdir(parents=True,exist_ok=True)
    combined=PdfWriter();models=[];all_metrics=[];cards=[];report_rows=[]
    for model in spec['models']:
        ident=model['id'];require(Path(ident).name==ident,'Invalid model directory')
        dest=out/ident;images=dest/'images';images.mkdir(parents=True,exist_ok=True)
        panels=[];links=[];settings=[]
        with PdfPages(dest/'TIMELINE.pdf') as pdf:
            for i,entry in enumerate(model['panels'],1):
                source=manifest.parent/entry['window']
                require(sha(source)==entry['sha256'],'Portable raw window hash mismatch')
                window=load(source)
                require(window['metadata']['state']==entry['source'],'Wrong source label')
                label=STATES[entry['source']]
                name=f'{i:02d}_'+{1:'p1_hit',2:'p2_hit',0:'miss'}[entry['source']]+'.png'
                panel=render(window,pdf,images/name)
                panel.update(window=os.path.relpath(source,dest),window_sha256=entry['sha256'],
                             page=i,setting=entry['setting'],image=f'images/{name}')
                panels.append(panel);all_metrics.extend(dict(m,model=ident,setting=entry['setting']) for m in panel['metrics'])
                links.append(f'<details><summary>{i}. {html.escape(entry["setting"])} — {label}</summary>'
                             f'<a href="{ident}/images/{name}"><img loading="lazy" src="{ident}/images/{name}"></a></details>')
                description=window['metadata']['description']
                if description not in settings:settings.append(description)
        require(len(PdfReader(dest/'TIMELINE.pdf').pages)==len(panels),'Page count mismatch')
        combined.append(str(dest/'TIMELINE.pdf'),outline_item=model['name'])
        audit=dict(model=ident,name=model['name'],note=model['note'],pages=len(panels),panels=panels,
                   settings=settings,source_window_hashes_verified=True,clock_shift_ns=0)
        (dest/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
        models.append(dict(id=ident,name=model['name'],pages=len(panels),pdf=f'{ident}/TIMELINE.pdf',
                           pdf_sha256=sha(dest/'TIMELINE.pdf'),audit_sha256=sha(dest/'AUDIT.json')))
        report_rows.append(f"| {model['name']} | {len(panels)} | [PDF]({ident}/TIMELINE.pdf) | {model['note']} |")
        cards.append(f'<section id="{ident}"><h2>{html.escape(model["name"])}</h2><p>{html.escape(model["note"])}</p>'
                     f'<p><a href="{ident}/TIMELINE.pdf">이 모델 PDF ({len(panels)}쪽)</a></p>'+''.join(links)+'</section>')
        print(ident,len(panels),'pages',flush=True)
    combined.write(out/'ALL_MODELS.pdf');combined.close()
    pages=sum(m['pages'] for m in models)
    require(len(PdfReader(out/'ALL_MODELS.pdf').pages)==pages,'Combined page count mismatch')
    write_csv(out/'DEADLINES.csv',all_metrics)
    (out/'INDEX.json').write_text(json.dumps(dict(status='complete',model_count=len(models),pages=pages,
        models=models,manifest_sha256=sha(manifest),builder_sha256=sha(Path(__file__)),
        renderer_sha256=sha(Path(__file__).with_name('plot_consecutive_steps.py')),
        inference_run=False,all_models_pdf_sha256=sha(out/'ALL_MODELS.pdf')),indent=2,ensure_ascii=False)+'\n')
    (out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET 모델별 timeline</title>
<style>body{font:16px system-ui;max-width:1900px;margin:30px auto;padding:20px;background:#f6f8fa;color:#243443}section{background:white;padding:24px;margin:20px 0}img{width:100%}p{line-height:1.7}summary{cursor:pointer;padding:12px;background:#edf2f7;margin:6px 0}a{color:#1565a2}</style>
<h1>모델 조합별 timeline — 이 폴더만 확인</h1><p><a href="ALL_MODELS.pdf">전체 PDF</a> · <a href="REPORT.md">설정·해석</a> · <a href="DEADLINES.csv">시간 수치</a></p>
<p>모델마다 PDF 하나입니다. 각 페이지는 같은 요청의 연속 두 step이며 P1 hit / P2 hit / miss를 구분합니다.
CodeLlama의 두 설정은 같은 모델 PDF 안에서 구분합니다. 이전 여러 형식의 그림은 삭제했습니다.</p>
<p>빨간 점선: 다음 target 요청 시작(deadline 기준). 초록 원: PS cache 완성. 삼각형: draft 수신 준비.
흰 점선은 forward 횟수 표시이며 실측 개별 forward 경계가 아닙니다. 양수 여유 시간이 모두 추가 계산 예산이라는 뜻은 아닙니다.</p>
<p>기존 raw 기록을 다시 그렸습니다. 3090/ver1.0 계열 측정과 4090/1.2 채택 최적화 측정은 환경이 다르며, 새로운 inference 결과가 아닙니다.</p>'''+''.join(cards)+'</html>')
    (out/'REPORT.md').write_text('''# 모델 조합별 timeline — 현재 확인 경로

이 폴더가 모델별 timeline의 단일 확인 경로다. 이전 ALIGNED/NATIVE/WAITS 및 별도 예시 PDF는 삭제했다.
원시 profile·수치·이벤트 JSON은 보존했으며 새로운 inference를 실행한 것은 아니다.

- [전체 PDF](ALL_MODELS.pdf), [브라우저에서 모델별 보기](index.html).
- 모델당 PDF 하나, 페이지마다 **같은 요청의 연속 두 step**. P1 hit, P2 hit, miss 순서다.
- CodeLlama 두 설정은 같은 PDF에 묶었으며, 12/6에서 유효한 miss 표본이 없어 그 페이지는 없다.
- 그래프마다 측정 버전, 파라미터, 요청·step을 표시한다. 일부 모델의 과거 trace를1.2 측정으로 취급하지 않는다.

| 모델 조합 | 쪽수 | 그림 | 측정 구분 |
|---|---:|---|---|
'''+ '\n'.join(report_rows)+'''

## 표시 읽는 법

- 빨간 점선: 다음 step의 `target_send_request` 시작. Draft 준비의 관측된 deadline 기준이다.
- 초록 원: PS cache merge 완료. 삼각형: 후처리까지 끝나고 다음 명령 수신을 시작한 시각.
- Ready margin = 다음 target send 시작 − draft 수신 준비. 음수는 준비가 늦었다는 관찰이다.
- 다음 response wait에는 cache 조회·통신·miss draft도 포함되므로 이전 PS의 지연으로 모두 귀속하지 않는다.
- 흰 점선은 DS/PS forward 횟수 안내다. Graph 내부 개별 forward 경계를 측정한 것이 아니다.
- 사선 빈 구간은 미계측 elapsed이며 GPU idle의 증거가 아니다. 부모·자식 구간을 합산하지 않는다.
- 시간축 이동과 최소 막대 폭 없이 원시 시각을 사용한다. GPU 간 시각 정렬 오차가 있고,
  관측된 여유를 전부 추가 계산에 사용할 수 있다는 보장은 없다.

## 측정 조건과 대표 선택

3090 모델별 자료는 설정별4턴의 ver1.0 계열 profile이다. 같은 source에서 후속 두 anchor가
있는 step의 target 구간 길이 중앙값을 선택했다. CodeLlama는 별도 code 입력이다.
7B+135M/4090은1.2 채택 소스의24턴 PS 최적화 비교 기록에서 기존 A/B 대표를 사용했다.
이는 파라미터 최적화나 전체 데이터셋 성능 평가가 아니다. 서로 다른 환경의 절대 시간을
직접 비교하여 모델/버전의 우열을 판단하지 않는다.

`SOURCES.json`과 `data/`의 window JSON은 새 그림의 재현 입력이다. 원본 interval과
source hash를 포함한다. `INDEX.json`, 모델별 `AUDIT.json`, `DEADLINES.csv`에서
페이지별 source·step·정확한 시각을 확인할 수 있다.

저장소 루트에서 재생성:

```bash
python ssd/tools/duet_timeline/build_model_timelines.py \\
  --manifest results/mlsys_ver12/model_timelines/SOURCES.json \\
  --output results/mlsys_ver12/model_timelines
```

필요 패키지는 numpy, matplotlib, pypdf다. GPU·모델·이전 서버의 파일 경로는 필요 없다.
이전 그림은 Git 커밋 `1f87514`에서 복구할 수 있다. 현재 확인에는 이 폴더만 사용한다.
''')


if __name__=='__main__':main()
