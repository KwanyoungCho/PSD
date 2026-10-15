"""Offline checks for raw detailed timelines; no model or GPU required."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'/'duet_timeline'))
from plot_model_breakdowns import group_events, union_ms, representatives
from duet_profile_analysis import ProfileAlignmentError
from plot_duet_aligned_timeline import draw_phase_lanes
from matplotlib import pyplot as plt
from plot_model_native_events import native_rows
from plot_model_wait_timelines import command_index, diagnose


def event(label, sid, start, end, parent=None):
    return dict(label=label,step_id=sid,parent_label=parent,
                wall_start_ns=start*10**6,wall_end_ns=end*10**6,
                cpu_dispatch_start_ns=start*10**6,cpu_dispatch_end_ns=end*10**6)


class DetailedTimelines(unittest.TestCase):
    def test_parent_before_anchor_at_one_step_warmup_reset(self):
        # Both adjacent requests have step 1: the containing parent must win.
        events=[event('target_spec_wait',1,0,10),
                event('target_send_request',1,1,2,'target_spec_wait'),
                event('target_spec_wait',1,100,110),
                event('target_send_request',1,101,102,'target_spec_wait')]
        grouped=group_events(events,'target_send_request')
        self.assertEqual([e['wall_start_ns'] for e in grouped[(1,1)]],[100*10**6,101*10**6])
        reverse=group_events(list(reversed(events)),'target_send_request')
        self.assertEqual({k:sorted(e['wall_start_ns'] for e in v) for k,v in grouped.items()},
                         {k:sorted(e['wall_start_ns'] for e in v) for k,v in reverse.items()})

    def test_repeated_shared_helpers_are_retained(self):
        events=[event('draft_recv_request',1,0,1),event('p2_prepare',1,2,3),
                event('p2_prepare',1,5,6),event('draft_recv_request',2,10,11)]
        grouped=group_events(events,'draft_recv_request')
        self.assertEqual(len(grouped[(0,1)]),3)
        self.assertEqual(len(grouped[(0,2)]),1)

    def test_sparse_labels_do_not_define_requests(self):
        events=[event('draft_recv_request',1,0,1),event('draft_recv_request',2,10,11),
                event('tree',2,12,13),event('draft_recv_request',3,20,21),
                event('draft_recv_request',1,30,31),event('tree',1,32,33)]
        grouped=group_events(events,'draft_recv_request')
        self.assertEqual(len(grouped[(0,2)]),2)
        self.assertEqual(len(grouped[(1,1)]),2)

    def test_missing_step_and_unanchored_label_rejected(self):
        for events in ([event('draft_recv_request',1,0,1),event('draft_recv_request',3,10,11)],
                       [event('draft_recv_request',1,0,1),event('helper',2,2,3)]):
            with self.assertRaises(ProfileAlignmentError):group_events(events,'draft_recv_request')

    def test_nested_and_overlapping_spans_union(self):
        events=[event('x',1,0,10),event('x',1,2,3),event('x',1,9,12),event('x',1,20,21)]
        self.assertEqual(union_ms(events),13)
        self.assertEqual(union_ms(events,'cpu_dispatch_start_ns','cpu_dispatch_end_ns'),13)
        self.assertEqual(union_ms([]),0)
        bad=deepcopy(events);bad[0]['wall_end_ns']=-1
        with self.assertRaises(ProfileAlignmentError):union_ms(bad)

    def test_representative_is_actual_step_and_deduplicates_small_sample(self):
        rows=[dict(state=1,epoch=1,step_id=i,target_step_span=float(i)) for i in range(1,10)]
        rows += [dict(state=0,epoch=1,step_id=10+i,target_step_span=20.+i) for i in range(2)]
        reps=representatives(rows)
        self.assertEqual([r['step_id'] for s,q,r in reps if s==1],[3,5,7])
        self.assertEqual(len([r for s,q,r in reps if s==0]),2)
        self.assertTrue(all(s!=2 for s,q,r in reps))
        self.assertTrue(all(any(r is source for source in rows) for s,q,r in reps))

    def test_compact_lanes_preserve_short_width_and_origin(self):
        stages=[dict(start_ns=1_000_000,end_ns=1_001_000,lane='proxy',
                     text='Proxy',color='#24966b')]
        original=deepcopy(stages);fig,ax=plt.subplots()
        try:
            records=draw_phase_lanes(ax,stages,origin_ns=500_000,xmax_ms=10)
            self.assertAlmostEqual(ax.patches[0].get_width(),.001)
            self.assertAlmostEqual(ax.patches[0].get_x(),.5)
            self.assertAlmostEqual(records[0]['duration_ms'],.001)
            self.assertEqual(stages,original)
        finally:plt.close(fig)

    def test_compact_lanes_reject_graph_outside_phase(self):
        stages=[dict(start_ns=1_000_000,end_ns=2_000_000,lane='draft',
                     text='DS',color='#8dc7ec',graphs=[(0,1_500_000)])]
        fig,ax=plt.subplots()
        try:
            with self.assertRaises(ValueError):
                draw_phase_lanes(ax,stages,origin_ns=0,xmax_ms=10)
        finally:plt.close(fig)

    def test_native_events_keeps_raw_intervals_and_overlapping_previous_step(self):
        target=[event('target_spec_wait',3,100,110),event('graph_pre',3,112,130)]
        current=[event('p1_prepare',3,108,109),event('p1_graph_replay',3,109,119)]
        previous=[event('p2_graph_replay',2,90,104),event('merge_cache',2,104,106),
                  event('p1_graph_replay',2,60,70)]
        arm=dict(grouped=dict(target={(1,3):target},draft={(1,3):current,(1,2):previous}))
        original=deepcopy(arm)
        t,d=native_rows(arm,dict(epoch=1,step_id=3))
        self.assertEqual(set(map(id,t)),set(map(id,target)))
        self.assertEqual(set(map(id,d)),set(map(id,current+previous[:2])))
        self.assertEqual(arm,original)

    def test_native_events_cannot_cross_request_boundary(self):
        arm=dict(grouped=dict(target={(2,1):[event('target_spec_wait',1,100,110)]},
            draft={(2,1):[event('p1_prepare',1,112,113)],
                   (1,1):[event('p1_prepare',1,100,120)]}))
        _,d=native_rows(arm,dict(epoch=2,step_id=1))
        self.assertEqual(d,arm['grouped']['draft'][(2,1)])

    def test_command_wait_associates_by_cpu_time_not_repeated_step_id(self):
        draft=[event('phase2_build',3,1,2),event('p2_prepare',3,3,4),
               event('merge_cache',3,7,8)]
        next_request=event('draft_recv_request',4,12,13)
        arm=dict(key='fixture',grouped=dict(draft={(2,3):draft,(2,4):[next_request]}))
        row=dict(epoch=2,step_id=3,uid='x',state=1,D2=7.,ps_graph=2.,ps_other=5.)
        # Commands inherit previous step ID; a different epoch repeats that ID.
        raw=[event('draft_recv_cmd',3,9,11),event('draft_recv_cmd',3,-10,-5)]
        original=deepcopy((arm,row,raw))
        diag=diagnose(arm,row,command_index(raw));m=diag['metrics']
        self.assertIs(diag['command'],raw[0])
        self.assertEqual(m['command_wait_ms'],2)
        self.assertEqual(m['post_ps_unprofiled_ms'],1)
        self.assertEqual(m['ps_end_to_next_request_ms'],4)
        self.assertEqual(m['ps_setup_gap_ms'],1)
        self.assertEqual((arm,row,raw),original)

    def test_command_wait_refuses_next_request_epoch_and_missing_wait(self):
        draft=[event('phase2_build',3,1,2),event('p2_prepare',3,3,4),
               event('merge_cache',3,7,8)]
        arm=dict(key='fixture',grouped=dict(draft={(2,3):draft,
                 (3,4):[event('draft_recv_request',4,12,13)]}))
        row=dict(epoch=2,step_id=3,uid='x',state=1,D2=7.,ps_graph=2.,ps_other=5.)
        with self.assertRaises(ProfileAlignmentError):diagnose(arm,row,command_index([]))
        arm['grouped']['draft'][(2,4)]=arm['grouped']['draft'].pop((3,4))
        with self.assertRaises(ProfileAlignmentError):diagnose(arm,row,command_index([]))

    def test_command_wait_rejects_overlapping_timestamps(self):
        draft=[event('phase2_build',3,1,2),event('p2_prepare',3,3,4),
               event('merge_cache',3,7,8)]
        arm=dict(key='fixture',grouped=dict(draft={(2,3):draft,
                 (2,4):[event('draft_recv_request',4,12,13)]}))
        row=dict(epoch=2,step_id=3,uid='x',state=1,D2=7.,ps_graph=2.,ps_other=5.)
        cmd=event('draft_recv_cmd',3,9,11);cmd['wall_start_ns']=7*10**6
        with self.assertRaises(ProfileAlignmentError):diagnose(arm,row,command_index([cmd]))


if __name__=='__main__':unittest.main()
