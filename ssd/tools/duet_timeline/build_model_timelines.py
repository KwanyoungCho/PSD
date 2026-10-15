#!/usr/bin/env python3
"""One current two-step timeline PDF per model pair, plus one combined PDF.

Rebuilds from portable, hash-pinned raw windows. This does not run inference or
relabel historical measurements as the current runtime's performance.
"""
import argparse
import html
import json
import os
import shutil
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

from plot_consecutive_steps import render
from plot_model_breakdowns import load, require, sha, write_csv, STATES
from phase_averages import (read_samples, summarize, table_cells, HEADERS,
                            DEFINITIONS, overlap_summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);manifest=args.manifest.resolve();out=args.output.resolve()
    spec=load(manifest);require(spec['schema']=='duet-model-timelines-v1','Unknown model manifest')
    out.mkdir(parents=True,exist_ok=True)
    # A redraw into a new directory must remain a complete portable bundle,
    # not just figures linked back to the exporting server's directory.
    if out != manifest.parent:
        inputs=[spec['phase_samples']]+[dict(path=p['window'],sha256=p['sha256'])
                    for m in spec['models'] for p in m['panels']]
        for entry in inputs:
            rel=Path(entry['path'])
            require(not rel.is_absolute() and '..' not in rel.parts,'Bundle input paths must stay within the bundle')
            src=manifest.parent/rel;dest=out/rel
            require(sha(src)==entry['sha256'],'Portable input hash mismatch')
            if dest.exists():
                require(sha(dest)==entry['sha256'],'Refusing to replace a different raw input in output directory')
            else:
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        dest=out/'SOURCES.json'
        if dest.exists():require(sha(dest)==sha(manifest),'Use a new output directory for a different source manifest')
        else:shutil.copyfile(manifest,dest)
        manifest=dest
    sample_path=manifest.parent/spec['phase_samples']['path']
    require(sha(sample_path)==spec['phase_samples']['sha256'],'Phase samples hash mismatch')
    populations=read_samples(sample_path)
    summaries={}
    for item in populations['sets']:
        key=(item['model'],item['setting'])
        require(key not in summaries,'Duplicate setting')
        summaries[key]=dict(model=item['model'],setting=item['setting'],measurement=item['measurement'],
                            k1=item['k1'],k2=item['k2'],rows=summarize(item['samples']),
                            proxy_overlap=overlap_summary(item['samples']))
    write_csv(out/'PHASE_AVERAGES.csv',[dict(model=s['model'],setting=s['setting'],**r)
                                      for s in summaries.values() for r in s['rows']])
    (out/'PHASE_AVERAGES.json').write_text(json.dumps(dict(definitions=DEFINITIONS,sets=list(summaries.values()),
        samples_sha256=sha(sample_path)),indent=2,ensure_ascii=False)+'\n')
    combined=PdfWriter();models=[];all_metrics=[];cards=[];report_rows=[]
    mean_report=[]; overlap_report=[]
    for model in spec['models']:
        ident=model['id'];require(Path(ident).name==ident,'Invalid model directory')
        dest=out/ident;images=dest/'images';images.mkdir(parents=True,exist_ok=True)
        panels=[];links=[];settings=[];mean_cards=[];seen_settings=set()
        with PdfPages(dest/'TIMELINE.pdf') as pdf:
            for i,entry in enumerate(model['panels'],1):
                source=manifest.parent/entry['window']
                require(sha(source)==entry['sha256'],'Portable raw window hash mismatch')
                window=load(source)
                require(window['metadata']['state']==entry['source'],'Wrong source label')
                label=STATES[entry['source']]
                name=f'{i:02d}_'+{1:'p1_hit',2:'p2_hit',0:'miss'}[entry['source']]+'.png'
                summary=summaries[(ident,entry['setting'])]
                if entry['setting'] not in seen_settings:
                    seen_settings.add(entry['setting'])
                    cells=table_cells(summary['rows'])
                    th=''.join('<th>'+html.escape(v)+'</th>' for v in HEADERS)
                    trs=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in cells)
                    overlap=summary['proxy_overlap']
                    overlap_note=(f'Proxy 계산 ↔ target 후반 forward 중첩: {overlap["overlapping_steps"]}/{overlap["steps"]} step, '
                                  f'전체 step 평균 중첩 {overlap["mean_overlap_ms"]:.3f}ms. 전송 완료 시간은 미계측.')
                    mean_cards.append(f'<h3>{html.escape(entry["setting"])} — K1/K2 {summary["k1"]}/{summary["k2"]}</h3><p>{html.escape(summary["measurement"])}</p><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table><p>{overlap_note}</p>')
                    overlap_report.append(f'- {model["name"]} / {entry["setting"]}: {overlap_note}')
                    mean_report.append('\n### '+model['name']+' — '+entry['setting']+'\n\n'+
                        summary['measurement']+'\n\n'+
                        '| '+' | '.join(HEADERS)+' |\n|'+ '|'.join(['---']*len(HEADERS))+'|\n'+
                        '\n'.join('| '+' | '.join(r)+' |' for r in cells)+'\n')
                panel=render(window,pdf,images/name,phase_summary=summary)
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
                     f'<p><a href="{ident}/TIMELINE.pdf">이 모델 PDF ({len(panels)}쪽)</a></p>'+''.join(mean_cards)+''.join(links)+'</section>')
        print(ident,len(panels),'pages',flush=True)
    combined.write(out/'ALL_MODELS.pdf');combined.close()
    pages=sum(m['pages'] for m in models)
    require(len(PdfReader(out/'ALL_MODELS.pdf').pages)==pages,'Combined page count mismatch')
    write_csv(out/'DEADLINES.csv',all_metrics)
    (out/'INDEX.json').write_text(json.dumps(dict(status='complete',model_count=len(models),pages=pages,
        models=models,manifest_sha256=sha(manifest),builder_sha256=sha(Path(__file__)),
        renderer_sha256=sha(Path(__file__).with_name('plot_consecutive_steps.py')),
        phase_averages_tool_sha256=sha(Path(__file__).with_name('phase_averages.py')),
        phase_samples_sha256=sha(sample_path),phase_averages_sha256=sha(out/'PHASE_AVERAGES.json'),
        inference_run=False,all_models_pdf_sha256=sha(out/'ALL_MODELS.pdf')),indent=2,ensure_ascii=False)+'\n')
    (out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET 모델별 timeline</title>
<style>body{font:16px system-ui;max-width:1900px;margin:30px auto;padding:20px;background:#f6f8fa;color:#243443}section{background:white;padding:24px;margin:20px 0}img{width:100%}p{line-height:1.7}summary{cursor:pointer;padding:12px;background:#edf2f7;margin:6px 0}a{color:#1565a2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;border:1px solid #dce2e8;text-align:center}th{background:#e1eaf0}tbody tr:last-child{background:#edf4ec;font-weight:bold}</style>
<h1>모델 조합별 timeline — 이 폴더만 확인</h1><p><a href="ALL_MODELS.pdf">전체 PDF</a> · <a href="REPORT.md">설정·해석</a> · <a href="PHASE_AVERAGES.csv">경우별·전체 평균 CSV</a> · <a href="DEADLINES.csv">대표 step 원시 시각</a></p>
<p>모델마다 PDF 하나입니다. 각 페이지는 같은 요청의 연속 두 step이며 P1 hit / P2 hit / miss를 구분합니다.
같은 모델의 여러 설정은 같은 PDF 안에서 구분합니다.</p>
<p>빨간 점선: 다음 target 요청 시작(deadline 기준). 초록 원: PS cache 완성. 삼각형: draft 수신 준비.
흰 점선은 forward 횟수 표시이며 실측 개별 forward 경계가 아닙니다. 양수 여유 시간이 모두 추가 계산 예산이라는 뜻은 아닙니다.</p>
<p><b>표는 P1 hit·P2 hit·Miss·Overall(전체 평균)의 네 행입니다.</b> 그림 위의 두 step은 실제 예시이고, 표는 해당 설정의 유효 step 전체 평균입니다.
Step: 현재 target 요청 시작→다음 요청 시작. DS/PS total: 준비·정리 포함 phase 시간.
DS spare*: proxy 전송 준비 완료−DS 완료(수신 도착 시점은 아님). PS spare: 다음 target 요청 시작−draft 수신 준비.
양수는 여유, 음수는 초과입니다. Graph/K1·K2는 graph 시간을 forward 횟수로 나눈 값이며 sampling·tree 선택을 포함합니다.</p>
<p>분류는 <b>현재 step에서 사용한 cache</b> 기준입니다. 전체 평균은 실제 step 수로 가중합니다. 표본이 없으면 0 대신 N/A입니다.</p>
<p><b>Proxy 계산과 target 후반 forward의 중첩 여부는 설정별로 기록에서 계산합니다.</b> exit_proxy_side라는 이름만으로 별도 stream 실행을 뜻하지 않습니다.
전송은 비동기 enqueue만 기록되어 실제 통신 완료·중첩 시간은 알 수 없습니다. Target와 별도 draft GPU의 병렬 실행과는 별개입니다.</p>
<p>기존 raw 기록의 후처리입니다. 측정 버전·환경은 모델별 설명을 확인하세요. 다른 서버/설정의 절대 시간을 직접 비교하여 개선으로 해석하지 않습니다.</p>'''+''.join(cards)+'</html>')
    (out/'REPORT.md').write_text('''# 모델 조합별 timeline — 현재 확인 경로

이 폴더는 입력 manifest에 지정한 모델별 timeline 모음이다.
원시 시각·평균 재현 입력을 함께 보존하며 새로운 inference를 실행한 것은 아니다.

- [전체 PDF](ALL_MODELS.pdf), [브라우저에서 모델별 보기](index.html).
- 모델당 PDF 하나, 페이지마다 **같은 요청의 연속 두 step**. P1 hit, P2 hit, miss 순서다.
- 같은 모델의 여러 설정은 같은 PDF에 묶는다. 유효한 연속 step 표본이 없는 source는 그림을 만들지 않는다.
- 그래프마다 측정 설명, 설정, 요청·step을 표시한다. 도구 버전과 실제 측정 runtime 버전을 구분한다.

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

## 경우별·전체 평균

그래프 하단의 절대 시각 표를 **P1 hit / P2 hit / Miss / Overall** 평균 표로 교체했다.
그림의 두 step은 실제 예시이며, 표는 같은 모델·설정의 유효 step 전체를 사용한다.
분류는 현재 step에서 사용한 cache source다. 그 step의 DS/PS는 다음 결과를 대비하는 생성이다.
전체 평균은 실제 step 수로 가중하며, 표본이 없는 경우는 N/A로 표시한다.

- Step: 현재 `target_send_request.start`부터 같은 요청의 다음 `target_send_request.start`까지.
  요청/응답·검증·후처리·다음 요청까지의 간격을 포함한다. `graph_pre` 간격이나 AL이 아니다.
- DS total: root 준비 시작부터 `proxy_wait.start`까지. Glue와 proxy 대기는 제외한다.
- DS spare*: target의 `proxy_send_enqueue.end` − DS 완료. 실제 수신 도착 시점은 계측되지
  않았으므로 **전송 준비 시각 기준**을 사용한다. 과거 보고서의 transport를 더한 추정 gap1과 다르다.
  `proxy_wait.end`로 계산하면 DS가 늦어 이미 도착한 proxy를 기다리게 한 경우를 숨기므로 사용하지 않는다.
- PS total: `proxy_wait.end`부터 `merge_cache.end`까지.
- PS spare: 다음 target 요청 시작 − 다음 draft 명령 수신 준비. Cache 완료 후 정리 시간까지 고려한다.
- DS/PS graph ÷ K1/K2: forward round당 평균. Graph 안의 sampling·tree 선택을 포함하며,
  개별 transformer forward를 분리 측정한 수치가 아니다. Source에 따라 입력 width도 다르다.
- Spare는 음수를 유지한다. 추가 forward에 전부 사용할 수 있는 확정 예산이나 인과적 stall 측정값은 아니다.

새 profile exporter는 warmup·요청의 처음 두 step·마지막 step·cap/clip step을 제외한다.
기존 자료의 필터는 함께 보존된 `eligibility`를 따른다. 지연시간이 길다는 이유로 제외하지 않는다.
입력 수·반복·실측 환경은 설정별 측정 설명을 따른다. 반복을 서로 다른 입력으로 세지 않는다.

`PHASE_AVERAGES.csv/json`은 아래 평균과 일치한다. `data/PHASE_SAMPLES.json.gz`에는
모든 대상 step의 원시 시각·현재/다음 source·K1/K2와 입력 hash를 보존했다.
다른 서버에서도 이 파일로 평균을 다시 계산한다.
'''+''.join(mean_report)+'''

## Proxy와 target 후반 forward의 중첩 여부

`exit_proxy_side`와 `graph_post`의 원시 시간 구간이 실제로 겹치는지 계산했다.
`PHASE_AVERAGES.json`의 설정별 `proxy_overlap`에 전체 대상 step의 중첩 건수와 평균을 저장했다.
`exit_proxy_side`는 별도 stream을 사용하지 않는 분기에도 붙는 이벤트 이름이다.
별도 stream 사용은 runtime의 `SSD_PROXY_STREAM=1` 또는 exit replica 경로에서 결정된다.
전송 막대 `proxy_send_enqueue`는 비동기 전송 요청 구간이며 실제 통신 완료를 계측하지 않는다.
따라서 통신이 겹치는지·얼마나 겹치는지를 이 막대만으로 판단하지 않는다.
이는 별도 draft GPU의 DS/PS가 target verification과 병렬 실행되는 것과 다른 문제다.
'''+ '\n\n'+ '\n'.join(overlap_report)+'''

## 측정 조건과 대표 선택

측정 버전·모델·입력 범위는 위 설정별 설명을 확인한다. 각 window의 `metadata.selection`에
대표 선택 근거가 있다. 새 exporter는 같은 source에서 후속 두 anchor가 있는 유효 step 중
현재 target 요청 시작→다음 요청 시작 시간의 중앙값에 해당하는 실제 step을 선택한다.
표는 대표 step만이 아닌 모든 유효 step의 평균이다. 이 도구는 파라미터 최적화나
전체 데이터셋 평가를 수행하지 않는다. 서로 다른 환경의 절대 시간을 직접 비교하지 않는다.

`SOURCES.json`과 `data/`의 window JSON 및 PHASE_SAMPLES.json.gz는 새 그림의 재현 입력이다. 원본 interval과
source hash를 포함한다. `INDEX.json`, 모델별 `AUDIT.json`, `DEADLINES.csv`에서
페이지별 source·step·정확한 시각을 확인할 수 있다.

저장소 루트에서 재생성:

```bash
python /path/to/repo/ssd/tools/duet_timeline/build_model_timelines.py \\
  --manifest /path/to/this_bundle/SOURCES.json \\
  --output /path/to/redrawn_bundle
```

필요 패키지는 numpy, matplotlib, pypdf다. GPU·모델·이전 서버의 파일 경로는 필요 없다.
새 실험에서 만드는 명령과 지원 입력 형식은 도구의 README를 참고한다.
''')


if __name__=='__main__':main()
