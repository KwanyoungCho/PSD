import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from mlsys_job_status import queue_state,validation_record


class JobStatus(unittest.TestCase):
    def test_last_job_name_is_not_liveness(self):
        base=dict(planned=31,validated=13,active_job='b2_workspace')
        self.assertEqual(queue_state(**base),'needs_attention')
        self.assertEqual(queue_state(**base,process_alive=True,heartbeat_age=1),'running')
        for age in (None,121,float('nan'),-1):
            self.assertEqual(queue_state(**base,process_alive=True,heartbeat_age=age),'needs_attention')

    def test_explicit_states_and_validation_scope(self):
        base=dict(planned=31,validated=13)
        for flag,state in [('stop_requested','stopped'),('waiting','waiting'),
                           ('queue_exhausted','drained'),('failed','needs_attention')]:
            self.assertEqual(queue_state(**base,**{flag:True}),state)
        self.assertEqual(queue_state(planned=31,validated=31),'complete')
        self.assertNotEqual(queue_state(planned=0,validated=0),'complete')
        r=validation_record('diagnostic',True,evidence='adapted.log')
        self.assertEqual(r['level'],'diagnostic')
        with self.assertRaises(ValueError):validation_record('all_pass',True)
