"""Deadline association and portable raw two-step plotting, without a GPU."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'/'duet_timeline'))
from plot_consecutive_steps import make_window, display_rows, render
from duet_profile_analysis import ProfileAlignmentError
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader


def event(label,sid,start,end,parent=None):
    return dict(label=label,step_id=sid,parent_label=parent,
                wall_start_ns=int(start*1e6),wall_end_ns=int(end*1e6),
                cpu_dispatch_start_ns=int(start*1e6),cpu_dispatch_end_ns=int(end*1e6))


def fixture():
    groups={'target':{},'draft':{}};commands=[]
    for sid in (1,2,3):
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


if __name__=='__main__':unittest.main()
