"""Native chain replay timing and inclusive accepted-length survival."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_consecutive_timeline import fixture,event
from phase_averages import sample,metrics,summarize
from plot_consecutive_steps import make_window,render
from plot_model_wait_timelines import command_index
from duet_profile_analysis import ProfileAlignmentError
from matplotlib.backends.backend_pdf import PdfPages

path=ROOT/'results/mlsys_ver12/chain40_k10_20261016/analyze.py'
spec=importlib.util.spec_from_file_location('chain_reporting_analysis',path)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def chain_fixture():
    groups,commands=fixture(3)
    for sid in range(1,4):
        base=(sid-1)*20
        d=groups['draft'][(0,sid)]
        d[:]=[e for e in d if e['label'] not in ('p1_graph_replay','p2_graph_replay','p2_prepare')]
        d += [event('phase1_build',sid,base+3.1,base+9.5),event('proxy_wait',sid,base+9.5,base+10)]
        for i in range(10):
            for label,offset,stride in [('phase1_replay',4,.4),('phase2_replay',14,.2)]:
                e=event(label,sid,base+offset+i*stride,base+offset+i*stride+.1)
                # Exact integer-nanosecond fixture, avoiding binary-float
                # truncation when testing a known100-us replay duration.
                e['wall_end_ns']=e['wall_start_ns']+100000
                e['cpu_dispatch_end_ns']=e['cpu_dispatch_start_ns']+100000
                d.append(e)
        groups['target'][(0,sid)] += [event('proxy_send_enqueue',sid,base+9.2,base+9.3),
            event('exit_proxy_side',sid,base+9,base+9.3),event('graph_post',sid,base+9.4,base+18)]
    return groups,commands


class ChainReporting(unittest.TestCase):
    def row(self,g,c):
        return sample(g,command_index(c),dict(epoch=0,step_id=1,uid='a',state=1,next_state=2,
            accepted_len=3,emitted_len=3,context_tokens=42,verify_width=10),10,10,'chain')

    def test_forward_sum_excludes_between_round_work(self):
        g,c=chain_fixture();r=self.row(g,c)
        self.assertAlmostEqual(r['ds_forward_once_ms'],.1,places=6)
        self.assertAlmostEqual(r['ps_forward_once_ms'],.1,places=6)
        self.assertAlmostEqual(metrics(r)['ds_round_ms'],.37,places=6)
        self.assertAlmostEqual(r['ds_compute_window_ms'],.8,places=6)
        self.assertAlmostEqual(r['ps_compute_window_ms'],2.5,places=6)
        self.assertAlmostEqual(summarize([r])[-1]['ds_forward_once_ms'],.1,places=6)

    def test_missing_round_is_rejected(self):
        g,c=chain_fixture();d=g['draft'][(0,1)];d.remove(next(e for e in d if e['label']=='phase1_replay'))
        with self.assertRaises(ProfileAlignmentError):self.row(g,c)

    def test_native_chain_picture_uses_actual_replays(self):
        g,c=chain_fixture();r=self.row(g,c)
        w=make_window(g,c,0,1,title='Native chain',metadata={},k1=10,k2=10)
        self.assertEqual(w['forward_counts'],{})
        with tempfile.TemporaryDirectory() as d:
            with PdfPages(Path(d)/'test.pdf') as pdf:
                audit=render(w,pdf,Path(d)/'test.png',phase_summary=dict(rows=summarize([r])))
        self.assertEqual(len(audit['forward_dividers']),4)
        self.assertTrue(all(r['measured_boundaries'] and r['count']==10 for r in audit['forward_dividers']))

    def test_survival_is_inclusive_and_conditional(self):
        rows=[dict(uid=str(i%2),state=s,accepted_len=v) for s in (1,2,0) for i,v in enumerate((1,2,4))]
        for r in analysis.survival(rows,draws=100):
            p=[x['probability'] for x in r['points']]
            self.assertEqual(p[:5],[1,2/3,1/3,1/3,0])
            self.assertAlmostEqual(sum(p),r['mean_al'])
            self.assertEqual(r['steps'],3)


if __name__=='__main__':unittest.main()
