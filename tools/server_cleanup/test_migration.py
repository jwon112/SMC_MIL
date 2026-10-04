"""Synthetic migration tests; real /proc access is exercised only on the server."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import manage

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.plan,self.patches=manage.read_package()
        for name in self.patches:
            target=self.root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((REPO/name).read_bytes())
        (self.root/'train_smc_event_stain_mil.py').write_text('# active event training marker')
        self.selected=[next(i for i in self.plan if 'acr_high_grade_l0_' in i['source'] and i['group']==group) for group in ('early_cv','weak_linkage')]
        for item in self.selected:
            folder=self.root/item['source']
            folder.mkdir(parents=True)
            (folder/'summary.csv').write_text('cv_val_auc,cv_val_acc\n0.75,0.75\n0.8,0.8\n0.9,0.9\n')
            for fold in range(3):
                data={f'{fold}-negative':{'label':0,'prob':[0.8,0.2]},f'{fold}-positive':{'label':1,'prob':[0.1,0.9]}}
                (folder/f'split_{fold}_results.pkl').write_bytes(pickle.dumps(data))
                (folder/f's_{fold}_checkpoint.pt').write_bytes(b'fixture checkpoint')
        active=self.root/'results/smc_event_patch_control_20261004_exclude25'
        active.mkdir()
        (active/'train.log').write_text('ongoing')
        # A nested external summary must also survive discovery after archival.
        ext=self.root/self.selected[0]['source']/'external'
        ext.mkdir()
        (ext/'external_summary.csv').write_text('experiment,task,cohort,auroc\n'+Path(self.selected[0]['source']).name+',acr_high,wsi_he,0.8\n')

    def action(self,action):
        with patch.object(sys,'argv',['manage.py','--project',str(self.root),'--action',action]),patch.object(manage,'check_processes'),contextlib.redirect_stdout(io.StringIO()) as out:
            manage.main()
        return out.getvalue()

    def collect_metrics(self):
        code='''import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
def load(name,file):
 s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('collect','tools/collect_all_smc_results.py')
internal=c.collect_internal(Path('results'),0.5)
external=c.collect_external(Path('results'))[0]
for row in internal+external: row.pop('source_path',None)
a=load('ensemble','tools/analyze_smc_prediction_ensembles.py')
oof=a.read_internal(Path('results'),3,{1}).to_dict('records')
v=load('external','tools/gse290577/evaluate_all_internal_experiments.py')
args=SimpleNamespace(results_root=Path('results'),include_task=None,include_variant=None,include_seed=None,worker='all')
discovered=[(x['experiment'],x['checkpoint_count']) for x in v.discover(args)]
print(json.dumps({'internal':internal,'external':external,'oof':oof,'external_discovery':discovered},sort_keys=True))
'''
        run=subprocess.run([sys.executable,'-c',code],cwd=self.root,text=True,capture_output=True,check=True)
        return json.loads(run.stdout)

    def test_metrics_discovery_apply_idempotence_and_rollback(self):
        before=self.collect_metrics()
        self.action('plan')
        self.assertFalse((self.root/manage.REGISTRY).exists())
        self.action('apply')
        self.assertEqual(self.collect_metrics(),before)
        self.action('apply')
        for item in self.selected:
            self.assertFalse((self.root/item['source']).exists())
            self.assertTrue((self.root/item['target']).exists())
        self.assertEqual((self.root/'results/smc_event_patch_control_20261004_exclude25/train.log').read_text(),'ongoing')
        self.action('rollback')
        self.assertEqual(self.collect_metrics(),before)
        self.assertFalse((self.root/manage.REGISTRY).exists())
        for name,h in self.patches.items():
            if h['before'] is not None:
                self.assertEqual(manage.code_digest(self.root/name),h['before'])

    def test_changed_server_code_blocks_all_moves(self):
        (self.root/'compare_experiments.py').write_text('# server changed')
        with self.assertRaisesRegex(ValueError,'differs'):
            self.action('apply')
        self.assertTrue(all((self.root/i['source']).exists() for i in self.selected))

    def test_failure_after_move_restores_everything(self):
        with patch.object(manage,'verify',side_effect=ValueError('simulated verification failure')):
            with self.assertRaisesRegex(ValueError,'simulated'):
                self.action('apply')
        self.assertTrue(all((self.root/i['source']).exists() for i in self.selected))
        self.assertFalse((self.root/manage.REGISTRY).exists())
        self.assertEqual(json.loads((self.root/manage.STATE_NAME/'transaction.json').read_text(encoding='utf-8'))['status'],'rolled_back')

    def test_rollback_collision_keeps_both_directories(self):
        self.action('apply')
        item=self.selected[0]
        (self.root/item['source']).mkdir()
        with self.assertRaisesRegex(ValueError,'collision'):
            self.action('rollback')
        self.assertTrue((self.root/item['source']).exists())
        self.assertTrue((self.root/item['target']).exists())

    def test_changed_archived_data_blocks_rollback(self):
        self.action('apply')
        (self.root/self.selected[0]['target']/'summary.csv').write_text('changed after migration')
        with self.assertRaisesRegex(ValueError,'changed'):
            self.action('rollback')
        self.assertFalse((self.root/self.selected[0]['source']).exists())

if __name__=='__main__':
    unittest.main()
