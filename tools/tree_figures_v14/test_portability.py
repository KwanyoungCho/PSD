"""CPU-only publication tests; GPU inference is never launched."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reproduce as r


class Portability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.work = self.base / 'work'
        self.rows = [dict(uid=f'{q}_t{t}',question_id=q,prompt='fixture',input_ids=[1,2])
                     for q in range(81,561) for t in range(2 if q<=160 else 1)]
        self.inputs = self.base / 'external_inputs.json'
        r.write(self.inputs, self.rows)
        self.fake_spec = dict(full_records=560,questions=480,smoke_records=9,
             full_semantic_sha256=r.input_digest(self.rows),
             smoke_semantic_sha256=r.input_digest(self.rows[:9]))
        for name in ('target','draft'):
            r.write(self.base/name/'config.json', {})
        self.env = patch.dict(os.environ); self.env.start()
        self.engine = patch.object(r,'engine_check'); self.engine.start()
        self.spec = patch.object(r,'input_spec',return_value=self.fake_spec); self.spec.start()

    def tearDown(self):
        self.spec.stop();self.engine.stop();self.env.stop();self.temp.cleanup()

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            r.init(argparse.Namespace(checkout=str(self.base/'engine'),
                target=str(self.base/'target'),draft=str(self.base/'draft'),
                inputs_full=str(self.inputs),work=str(self.work),cuda_arch='8.9',
                pairs=['0,1'],cpu_sets=None))
        return r.context(self.work)

    def test_bundle_integrity_and_no_results_or_dataset_payload(self):
        r.bundle_check()
        self.assertFalse((r.BUNDLE/'payload/INPUTS_FULL.json').exists())
        self.assertFalse((r.BUNDLE/'reference').exists())
        self.assertFalse(list(r.BUNDLE.rglob('*.pdf')))

    def test_external_input_mutation_and_order_are_rejected(self):
        rows = self.rows.copy(); rows[0],rows[2] = rows[2],rows[0]
        with self.assertRaisesRegex(ValueError,'External inputs differ'):
            r.validate_inputs(rows)
        rows = json.loads(json.dumps(self.rows));rows[0]['input_ids']=[3]
        with self.assertRaisesRegex(ValueError,'External inputs differ'):
            r.validate_inputs(rows)

    def test_init_preserves_frozen_code_and_derives_smoke_from_full(self):
        work, config = self.prepare()
        self.assertEqual(config['seeds'],[42,123,2026])
        self.assertEqual(json.loads((work/'raw/INPUTS_SMOKE.json').read_text()),self.rows[:9])
        for path in (r.BUNDLE/'payload').rglob('*.py'):
            self.assertEqual(path.read_bytes(),(work/'raw'/path.relative_to(r.BUNDLE/'payload')).read_bytes())
        for path in (work/'report/tools').glob('*.py'):
            compile(path.read_text(),str(path),'exec')
        self.assertNotIn('/data/chokwans99/',(work/'report/tools/analyze_render.py').read_text())

    def test_configuration_mutation_is_rejected(self):
        self.prepare()
        config=json.loads((self.work/'CONFIG.json').read_text());config['pairs']=[[2,3]]
        r.write(self.work/'CONFIG.json',config)
        with self.assertRaisesRegex(RuntimeError,'configuration changed'):
            r.context(self.work)

    def test_prepared_dataset_mutation_is_rejected(self):
        self.prepare()
        rows=self.rows.copy();rows.reverse();r.write(self.work/'raw/INPUTS_FULL.json',rows)
        with self.assertRaisesRegex(ValueError,'External inputs differ'):
            r.context(self.work)

    def test_full_and_smoke_contracts(self):
        work,config=self.prepare()
        value=r.job(work,config,123,'observed',True)
        self.assertEqual((value['config']['k1'],value['config']['k2']),(10,5))
        self.assertEqual(value['config']['output_tokens'],1024)
        self.assertEqual(value['figure_stride'],8)
        small=r.job(work,config,42,'small',False,True)
        self.assertEqual(small['config']['output_tokens'],96)
        self.assertEqual(small['figure_stride'],1)
        self.assertFalse(small['shapes'])

    def test_runner_uses_receiving_server_paths_without_GPU(self):
        work,config=self.prepare();runner=r.load_runner(work,config)
        value=r.job(work,config,42,'mock_only',True)
        class FakeProcess:
            pid=99999
            returncode=0
            def poll(self):return 0
        with patch.object(runner,'state',return_value=dict(gpu=['0, U0','1, U1'],processes=[])), \
             patch.object(runner.subprocess,'Popen',return_value=FakeProcess()) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            result=r.launch(runner,value,0)
        argv=popen.call_args.args[0];env=popen.call_args.kwargs['env']
        self.assertEqual(argv[argv.index('--target')+1],config['target'])
        self.assertEqual(argv[argv.index('--draft')+1],config['draft'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
        self.assertEqual(env['REVIEW_SHAPES'],'1')
        self.assertEqual(env['FIGURE_STRIDE'],'8')
        self.assertTrue(result['source_unchanged'])

    def test_parity_rejects_changed_AL_even_if_tokens_match(self):
        self.prepare()
        baseline=dict(paper_commit=r.COMMIT,paper_dirty='',status='complete',prompt_ids=[[1]],
            warm_groups=[],groups=[dict(outputs=[[2]],steps=[dict(events=[dict(source=1,accepted_len=2)])])])
        for tag in ('native','observed'):
            r.write(self.work/'raw/runs'/tag/'run.json',baseline)
        r.parity(self.work,'native','observed',1)
        baseline['groups'][0]['steps'][0]['events'][0]['accepted_len']=3
        r.write(self.work/'raw/runs/observed/run.json',baseline)
        with self.assertRaisesRegex(RuntimeError,'Observer changed'):
            r.parity(self.work,'native','observed',1)

    def test_real_external_input_contract_when_provided(self):
        path=os.getenv('PORT_TEST_INPUTS')
        if not path:self.skipTest('Set PORT_TEST_INPUTS for the original dataset integration check')
        self.spec.stop()
        rows=json.loads(Path(path).read_text())
        self.assertEqual(r.validate_inputs(rows),rows[:9])


if __name__=='__main__':unittest.main(verbosity=2)
