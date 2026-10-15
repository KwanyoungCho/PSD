"""Same-input before/after native timelines using the established renderer."""
import json,sys
from pathlib import Path
from pypdf import PdfWriter
from analyze import read_arm,load_profile,group_events
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'ssd/bench'),str(ROOT/'ssd/tools/duet_timeline')]
import plot_duet_aligned_timeline as native
from plot_model_native_events import DETAIL_STYLES
native.COLORS.update(DETAIL_STYLES)
native.COLORS.update({'PS dispatch: slots / validity':('#eeeeee','////'),
                     'PS prepare A: root metadata':('#fdae6b','//'),
                     'PS prepare B: graph inputs':('#fd8d3c','//')})


def main():
    raw=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    _,rows,_=read_arm(raw/'before_graph1');pdf=PdfWriter();selected=[]
    for source in (1,2,0):
        subset=sorted([r for r in rows if r['source']==source],key=lambda r:r['ps_total'])
        if not subset:continue
        row=subset[len(subset)//2];key=(row['epoch'],row['step_id'])
        for name in ('before_graph1','after_graph1'):
            d,_=group_events(load_profile(raw/name,'draft'),'draft_recv_request')
            t,_=group_events(load_profile(raw/name,'target_rank0'),'target_send_request')
            target=[e for es in t[key].values() for e in es]
            draft=[e for es in d[key].values() for e in es]
            # Previous PS may overlap this target wait. Preserve those raw spans.
            wait=t[key]['target_spec_wait'][0]
            for es in d.get((key[0],key[1]-1),{}).values():
                draft.extend(e for e in es if min(e['wall_end_ns'],wait['wall_end_ns'])>max(e['wall_start_ns'],wait['wall_start_ns']))
            preps=sorted(d[key]['p2_prepare'],key=lambda e:e['cpu_dispatch_start_ns'])
            aliases={id(e):'PS prepare '+('A: root metadata' if i==0 else 'B: graph inputs') for i,e in enumerate(preps)}
            draft=[dict(e,label=aliases[id(e)]) if id(e) in aliases else e for e in draft]
            build=d[key]['phase2_build'][0]
            draft.append(dict(label='PS dispatch: slots / validity',step_id=key[1],parent_label=None,
                              wall_start_ns=build['wall_end_ns'],wall_end_ns=preps[0]['wall_start_ns']))
            order=lambda e:(-(e['wall_end_ns']-e['wall_start_ns']),e['wall_start_ns'],e['label'])
            stem=f'{name}_source{source}_req{key[0]}_step{key[1]}'
            label = 'Previous PS optimization' if name.startswith('before') else 'Deferred host metadata'
            suffix=f'{label}: Llama2 7B + AMD135M, B1 BF16 T0.7; source={source}, UID={row["uid"]}\nSame request + step; K1/K2=6/3, exit20, C3, M1/M2=12/8\nRaw timestamps; dispatch gap derived from endpoints; dotted forward counts are NOT measured boundaries'
            for ext in ('pdf','png'):
                native.plot_aligned_step(sorted(target,key=order),sorted(draft,key=order),key[1],out/(stem+'.'+ext),
                    title_suffix=suffix,draft_forward_counts={'p1_graph_replay':6,'p2_graph_replay':3})
            pdf.append(str(out/(stem+'.pdf')))
            selected.append(dict(arm=name,source=source,epoch=key[0],step_id=key[1],uid=row['uid'],file=stem+'.pdf'))
    pdf.write(out/'NATIVE_BEFORE_AFTER.pdf');pdf.close()
    (out/'NATIVE_SELECTIONS.json').write_text(json.dumps(selected,indent=2))

if __name__=='__main__':main()
