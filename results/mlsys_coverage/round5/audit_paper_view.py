"""Reconcile atlas coverage, additive timings, metrics, links and PDF pages."""
import csv
import hashlib
from html.parser import HTMLParser
import math
import platform
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import matplotlib
import numpy
import pypdf

from paper_view_data import HERE, OUT, dump, read


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]

    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.links.append(value)


def main():
    data=read(OUT/'DATA.json');complete=read(OUT/'COMPLETE.json')
    assert complete['complete'] and complete['gpu_jobs_launched']==0
    assert len(data['rows'])==487 and len(data['passes'])==691
    assert len({r['name'] for r in data['rows']})==487
    status_checks=0
    for source,expected in data['source_sha256'].items():
        assert hashlib.sha256((HERE/source).read_bytes()).hexdigest()==expected,source
    for r in data['rows']:
        w,m=r['wall'],r['last']
        assert math.isclose(w['tps'],m['boundary_excluded_step_tps'],rel_tol=1e-12)
        assert math.isclose(sum(g['time_share'] for g in w['groups']),1,rel_tol=1e-12)
        assert sum(g['steps'] for g in w['groups'])==w['included_steps']
        for ext in ('png','pdf'):
            assert (OUT/'figs'/f"{r['name']}.{ext}").stat().st_size>1000
        assert len(pypdf.PdfReader(OUT/'figs'/f"{r['name']}.pdf").pages)==1
        if r['detail']:
            assert r['detail']['max_target_stage_overlap_ms']<1e-7
            for g in r['detail']['groups']:
                value=sum(c['mean_ms'] for c in g['components'] if c['lane']=='Target')
                assert math.isclose(value,g['mean_target_ms'],rel_tol=1e-8)
                assert g['n']==len(g['step_ids'])==sum(g['query_shapes'].values())
                assert g['representative']['step_id'] in g['step_ids']
                status_checks+=1
    counts={}
    for name,n in [('ALL_EXPERIMENTS.csv',487),('ALL_PASSES.csv',691),('ALL_PARAMETERS.csv',487),('FULL480.csv',70),('SUMMARY.csv',28)]:
        with (OUT/name).open(encoding='utf-8-sig') as f:count=len(list(csv.DictReader(f)))
        assert count==n,(name,count,n)
        counts[name]=count
    main_pages=len(pypdf.PdfReader(OUT/'ALL_BREAKDOWNS.pdf').pages)
    final_pages=len(pypdf.PdfReader(OUT/'FINAL_BREAKDOWNS.pdf').pages)
    assert main_pages==487 and final_pages==17
    links=0
    for p in list(OUT.rglob('*.html'))+list(OUT.glob('*.md')):
        text=p.read_text()
        if p.suffix=='.html':
            parser=Links();parser.feed(text);targets=parser.links
        else:targets=re.findall(r'\]\(([^)]+)\)',text)
        for target in targets:
            url=urlsplit(target)
            if url.scheme or not url.path:continue
            q=(p.parent/unquote(url.path)).resolve()
            assert q.is_file(),(p,target)
            links+=1
    report=dict(passed=True,jobs=487,passes=691,profiles=337,wall_only=150,
                status_component_sum_checks=status_checks,wall_tps_reconciliations=487,
                csv_counts=counts,all_pdf_pages=main_pages,final_pdf_pages=final_pages,
                html_pages=len(list(OUT.rglob('*.html'))),local_links_checked=links,
                dependencies=dict(python=platform.python_version(),numpy=numpy.__version__,
                                  matplotlib=matplotlib.__version__,pypdf=pypdf.__version__),
                source_sha256=data['source_sha256'])
    dump(OUT/'AUDIT.json',report)
    print(report)


if __name__=='__main__':main()
