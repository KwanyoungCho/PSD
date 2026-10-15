"""Deadline association and portable raw two-step plotting, without a GPU."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'/'duet_timeline'))
from plot_consecutive_steps import make_window, display_rows, render
from duet_profile_analysis import ProfileAlignmentError
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader
from phase_averages import metrics, summarize, table_cells, overlap_summary
from export_profile_timelines import export, normalize_outcomes
from build_model_timelines import main as build_bundle


def event(label,sid,start,end,parent=None):
    return dict(label=label,step_id=sid,parent_label=parent,
                wall_start_ns=int(start*1e6),wall_end_ns=int(end*1e6),
                cpu_dispatch_start_ns=int(start*1e6),cpu_dispatch_end_ns=int(end*1e6))


def fixture(count=3):
    groups={'target':{},'draft':{}};commands=[]
    for sid in range(1,count+1):
        base=(sid-1)*20
        groups['target'][(0,sid)]=[
            event('target_spec_wait',sid,base,base+4),
            event('target_send_request',sid,base+.5,base+1,'target_spec_wait'),
            event('target_recv_response_wait',sid,base+1,base+4,'target_spec_wait'),
            event('target_postprocess',sid,base+18,base+19)]
        groups['draft'][(0,sid)]=[
            event('draft_recv_request',sid,base+2,base+3),
            event('p1_graph_replay',sid,base+4,base+9),
            event('phase2_build',sid,base+10,base+11),
            event('p2_prepare',sid,base+12,base+13),
            event('p2_prepare',sid,base+13,base+14),
            event('p2_graph_replay',sid,base+14,base+17),
            event('merge_cache',sid,base+17,base+18)]
        commands.append(event('draft_recv_cmd',sid,base+19,base+21))
    return groups,commands


def build(groups,commands):
    return make_window(groups,commands,0,1,title='Test',metadata={},k1=6,k2=3)


class ConsecutiveTimeline(unittest.TestCase):
    def test_cache_completion_is_not_receive_readiness(self):
        groups,commands=fixture();original=deepcopy((groups,commands))
        window=build(groups,commands);tr=window['transitions'][0]
        self.assertEqual(tr['cache_margin_ms'],2.5)
        self.assertEqual(tr['ready_margin_ms'],1.5)
        self.assertEqual(tr['post_cache_cleanup_ms'],1)
        self.assertEqual(tr['next_response_wait_ms'],3)
        self.assertGreater(tr['next_response_wait_ms'],tr['ready_margin_ms'])
        self.assertEqual((groups,commands),original)

    def test_two_steps_retained_and_third_only_supplies_boundary(self):
        groups,commands=fixture();window=build(groups,commands)
        self.assertEqual({e['step_id'] for e in window['target_events']},{1,2})
        self.assertEqual(window['transitions'][1]['next_step_id'],3)
        self.assertEqual(window['transitions'][1]['deadline_ns'],40500000)

    def test_missing_third_or_cross_request_anchor_fails(self):
        for role in ('draft','target'):
            groups,commands=fixture();groups[role][(1,3)]=groups[role].pop((0,3))
            with self.assertRaises(ProfileAlignmentError):build(groups,commands)

    def test_commands_associated_by_time_not_stale_step_id(self):
        groups,commands=fixture()
        commands[0]['step_id']=999
        commands.append(event('draft_recv_cmd',1,-30,-20))
        self.assertEqual(build(groups,commands)['transitions'][0]['command']['step_id'],999)

    def test_missing_or_ambiguous_command_is_rejected(self):
        groups,commands=fixture()
        for bad in (commands[1:],commands+[deepcopy(commands[0])]):
            with self.assertRaises(ProfileAlignmentError):build(groups,bad)

    def test_late_ready_margin_remains_negative(self):
        groups,commands=fixture()
        commands[0]['wall_start_ns']=20800000
        window=build(groups,commands)
        self.assertAlmostEqual(window['transitions'][0]['ready_margin_ms'],-.3)
        self.assertEqual(window['transitions'][0]['cache_margin_ms'],2.5)

    def test_overlapping_tail_is_not_silently_clock_shifted(self):
        groups,commands=fixture();commands[0]['wall_start_ns']=17500000
        with self.assertRaises(ProfileAlignmentError):build(groups,commands)

    def test_display_aliases_do_not_mutate_raw_and_skip_early_prepare(self):
        groups,commands=fixture()
        groups['draft'][(0,1)].append(event('p2_prepare',1,3,3.5))
        window=build(groups,commands);before=deepcopy(window)
        _,display,gaps=display_rows(window)
        self.assertEqual(sum(e['label']=='PS prepare A: roots / visibility' for e in display),2)
        self.assertEqual(sum(e['label']=='p2_prepare' for e in display),1)
        self.assertEqual(window,before)
        self.assertTrue(all(e['derived'] for e in gaps))

    def test_portable_render_and_forward_count_annotations(self):
        groups,commands=fixture();window=build(groups,commands);before=deepcopy(window)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with PdfPages(path/'out.pdf') as pdf:
                audit=render(window,pdf,path/'out.png')
            self.assertEqual(len(PdfReader(path/'out.pdf').pages),1)
            self.assertGreater((path/'out.png').stat().st_size,1000)
        self.assertEqual(window,before)
        self.assertEqual(len(audit['forward_dividers']),4)
        self.assertFalse(any(e['measured_boundaries'] for e in audit['forward_dividers']))
        self.assertEqual(audit['minimum_bar_width_ms'],0)
        self.assertEqual(audit['clock_shift_ns'],0)


class PhaseAverages(unittest.TestCase):
    def sample(self, sid=1, state=1, step_ms=20):
        return dict(run='example',epoch=0,step_id=sid,state=state,k1=2,k2=1,
                    step_start_ns=0,next_step_start_ns=int(step_ms*1e6),
                    ds_start_ns=2000000,ds_end_ns=8000000,proxy_send_ns=7000000,
                    ds_graph_start_ns=3000000,ds_graph_end_ns=7000000,
                    ps_start_ns=9000000,ps_end_ns=16000000,draft_ready_ns=21000000,
                    ps_graph_start_ns=10000000,ps_graph_end_ns=14000000,
                    proxy_start_ns=6000000,proxy_end_ns=7000000,
                    target_post_start_ns=7000000,target_post_end_ns=19000000)

    def test_overall_uses_step_weights_and_empty_source_is_not_zero(self):
        samples=[self.sample(1,1,20),self.sample(2,1,22),self.sample(3,2,30)]
        rows=summarize(samples)
        self.assertEqual(rows[0]['step_ms'],21)
        self.assertEqual(rows[3]['step_ms'],24)
        self.assertEqual(rows[2]['n'],0)
        self.assertIsNone(rows[2]['step_ms'])
        self.assertEqual(table_cells(rows)[2][2],'N/A')

    def test_negative_spare_and_graph_per_round_preserved(self):
        m=metrics(self.sample())
        self.assertEqual(m['ds_margin_ms'],-1)
        self.assertEqual(m['ps_margin_ms'],-1)
        self.assertEqual(m['ds_round_ms'],2)
        self.assertEqual(m['ps_round_ms'],4)
        self.assertEqual(m['ds_total_ms'],6)

    def test_duplicate_step_rejected_but_separate_repeats_allowed(self):
        a=self.sample();b=deepcopy(a)
        with self.assertRaises(ProfileAlignmentError):summarize([a,b])
        b['run']='repeat2'
        self.assertEqual(summarize([a,b])[-1]['n'],2)

    def test_overlap_comes_from_intervals_not_label(self):
        a=self.sample()
        self.assertEqual(overlap_summary([a])['overlapping_steps'],0)
        a['proxy_end_ns']=8000000
        self.assertEqual(overlap_summary([a])['mean_overlap_ms'],1)

    def test_render_shows_all_population_means_without_changing_windows(self):
        groups,commands=fixture();window=build(groups,commands);before=deepcopy(window)
        summary=dict(rows=summarize([self.sample()]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with PdfPages(path/'out.pdf') as pdf:
                audit=render(window,pdf,path/'out.png',phase_summary=summary)
            content=PdfReader(path/'out.pdf').pages[0].extract_text()
            self.assertIn('Overall',content)
            self.assertIn('DS graph/K1',content)
            self.assertIn('N/A',content)
        self.assertEqual(window,before)
        self.assertEqual(audit['phase_summary'],summary)


class PortableExport(unittest.TestCase):
    def outcomes(self):
        return dict(schema='duet-timeline-outcomes-v1',status='complete',batch_size=1,k1=6,k2=3,
                    requests=[dict(epoch=0,uid='question0',warmup=False,steps=[
                        dict(step_id=s,source=1 if s==3 else 2,cap=s==4,clipped=False) for s in range(1,7)])])

    def test_rejects_ambiguous_sources_and_batched_outcomes(self):
        for modify in (lambda x:x.update(batch_size=2),
                       lambda x:x['requests'][0]['steps'][0].update(source=3),
                       lambda x:x['requests'][0]['steps'][0].pop('cap')):
            data=self.outcomes();modify(data)
            with self.assertRaises((ProfileAlignmentError,KeyError)):normalize_outcomes(data)

    def test_existing_benchmark_outcomes_adapter(self):
        data=self.outcomes();r=data['requests'][0]
        run=dict(status='complete',args=dict(batch=1,k1=6,k2=3),warm_groups=[],groups=[dict(
            phase='measured_0',n=1,request_records=[dict(paper_uid='question0')],steps=[
                dict(cap=s['cap'],clipped=s['clipped'],events=[dict(batch_size=1,step_id=s['step_id'],source=s['source'])])
                for s in r['steps']])])
        self.assertEqual(normalize_outcomes(run),data)

    def test_export_redraw_without_original_server_files(self):
        groups,commands=fixture(6)
        for sid in range(1,7):
            base=(sid-1)*20
            groups['draft'][(0,sid)]+=[event('p1_root_build',sid,base+3.1,base+3.5),
                                      event('proxy_wait',sid,base+9.5,base+10)]
            groups['target'][(0,sid)]+=[event('exit_proxy_side',sid,base+9,base+9.3),
                                       event('proxy_send_enqueue',sid,base+9.25,base+9.3),
                                       event('graph_post',sid,base+9.4,base+18)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'original';original.mkdir();profiles=original/'profiles';profiles.mkdir()
            for role,prefix in [('target','target_rank0'),('draft','draft')]:
                events=[e for es in groups[role].values() for e in es]+(commands if role=='draft' else [])
                (profiles/f'duet_profile_{prefix}_test.json').write_text(json.dumps(events))
            (original/'outcomes.json').write_text(json.dumps(self.outcomes()))
            spec=dict(schema='duet-timeline-inputs-v1',datasets=[dict(model_id='test_model',model_name='Test Model',
                setting='test',measurement='Synthetic B1 fixture',runs=[dict(outcomes='outcomes.json',profile_dir='profiles')])])
            spec_path=original/'inputs.json';spec_path.write_text(json.dumps(spec))
            output=root/'exported';export(spec_path,output)
            # All means need only the compact sample export. The picture needs
            # only the selected window; no lazy read of absolute source paths.
            relocated=root/'another_server';shutil.copytree(output,relocated);shutil.rmtree(original);shutil.rmtree(output)
            build_bundle(['--manifest',str(relocated/'SOURCES.json'),'--output',str(root/'redrawn')])
            result=json.loads((root/'redrawn/PHASE_AVERAGES.json').read_text())['sets'][0]
            self.assertEqual([r['n'] for r in result['rows']],[1,1,0,2])
            self.assertEqual(result['rows'][-1]['step_ms'],20)
            self.assertEqual(result['proxy_overlap']['overlapping_steps'],0)
            index=json.loads((root/'redrawn/INDEX.json').read_text())
            self.assertEqual(index['pages'],1)
            self.assertTrue((root/'redrawn/SOURCES.json').is_file())
            self.assertTrue((root/'redrawn/data/PHASE_SAMPLES.json.gz').is_file())
            audit=json.loads((root/'redrawn/test_model/AUDIT.json').read_text())
            self.assertEqual(audit['panels'][0]['window'],'../data/dataset0_source1.json')
            content=(root/'redrawn/REPORT.md').read_text()
            self.assertIn('Synthetic B1 fixture',content)
            self.assertNotIn('RTX3090',content)


if __name__=='__main__':unittest.main()
