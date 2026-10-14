"""Portable offline checks for the current-server timing alignment repair."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from duet_profile_analysis import index_events,parse_events,ProfileAlignmentError


def fixture():
    target=[];draft=[]
    def event(label,sid,base,start,end):
        return dict(label=label,step_id=sid,cpu_dispatch_start_ns=base+start*10**6,
                    wall_start_ns=base+start*10**6,wall_end_ns=base+end*10**6)
    for epoch in range(2):
        for sid in range(1,5):
            base=10**15+epoch*10**10+sid*400*10**6
            for label,start,end in (('target_send_request',0,1),('graph_pre',1,10),
                                   ('proxy_send_enqueue',15,16),('graph_post',16,60),
                                   ('final_logits',60,61)):
                target.append(event(label,sid,base,start,end))
            for label,start,end in (('draft_recv_request',0,2),('p1_root_build',3,5),
                                   ('p1_graph_replay',5,12),('proxy_wait',13,20),
                                   ('merge_cache',30,40)):
                draft.append(event(label,sid,base,start,end))
    records=[dict(uid='fixture',prompt_tokens=20,metrics=dict(phase_events=[
        dict(step_id=sid,source=1,accepted_len=2) for sid in range(1,5)]))]
    return draft,target,records,dict(k1=8,k2=4)


class ProfileAlignment(unittest.TestCase):
    def test_sparse_tree_and_long_sample_not_censored(self):
        rows,diag=parse_events(*fixture())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['cycle'],400.)
        self.assertEqual(diag['coverage'],1.)
        self.assertEqual(diag['dropped'],{})
        self.assertEqual(diag['p1_span_labels'],{'p1_root_build':1})
        self.assertEqual(rows[0]['D1'],10.)
        self.assertEqual(rows[0]['D2'],20.)
        self.assertEqual(rows[0]['gap2'],360.)
        self.assertEqual(rows[0]['gap2_verify'],21.)

    def test_nested_emission_order_does_not_define_epoch(self):
        d,t,r,c=fixture()
        a,_=parse_events(d,t,r,c);b,_=parse_events(list(reversed(d)),list(reversed(t)),r,c)
        self.assertEqual(a,b)

    def test_missing_duplicate_negative_and_count_errors(self):
        for kind in ('anchor','span','duplicate','negative','count'):
            d,t,r,c=fixture()
            if kind=='anchor': d.pop(0)
            if kind=='span':d=[x for x in d if x['label']!='proxy_wait']
            if kind=='duplicate':d.append(deepcopy(d[1]))
            if kind=='negative':d[1]['wall_end_ns']=d[1]['wall_start_ns']-1
            if kind=='count':r[0]['metrics']['phase_events'].pop()
            with self.subTest(kind=kind),self.assertRaises(ProfileAlignmentError):parse_events(d,t,r,c)

    def test_empty_outcome_request_does_not_shift_profile_epoch(self):
        d,t,r,c=fixture();r.insert(0,dict(uid='empty',metrics=dict(phase_events=[])))
        rows,_=parse_events(d,t,r,c)
        self.assertEqual(rows[0]['prompt'],1)


if __name__=='__main__':unittest.main()
