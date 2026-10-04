"""Validate source-only commits and notebook output removal using a local remote."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('publish_sources',Path(__file__).resolve().parents[1]/'publish_server_sources.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)

class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.root=self.base/'checkout';self.root.mkdir()
        self.remote=self.base/'remote.git'
        subprocess.run(['git','init','--bare','--initial-branch=main',str(self.remote)],check=True,capture_output=True)
        def git(*args): return publisher.git(self.root,*args).stdout.decode().strip()
        self.git=git
        git('init','-b','main');git('config','user.name','Test');git('config','user.email','test@example.invalid');git('config','core.autocrlf','false')
        (self.root/'.gitignore').write_text('.server_source_sync/\ndata/\nresults/\n')
        (self.root/'train.py').write_text('# baseline\n')
        git('add','.gitignore','train.py');git('commit','-m','baseline');git('remote','add','origin',str(self.remote));git('push','-u','origin','main')
        (self.root/'new_research.py').write_text('print("research")\n')
        (self.root/'experiment_config.json').write_text('{"epochs": 50}\n')
        (self.root/'dataset_dump.json').write_text('{"patient_rows": ["SECRET_PATIENT"]}')
        (self.root/'secret.py').write_text('access_token = "hf_'+'A'*30+'"\n')
        (self.root/'data').mkdir();(self.root/'data/feature.pt').write_bytes(b'feature')
        (self.root/'marker.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'widgets':{'x':'OUTPUT_DATA'}},'cells':[{'cell_type':'code','source':['print(1)'],'metadata':{},'execution_count':1,'outputs':[{'text':'OUTPUT_DATA'}]}]}))

    def plan(self):
        with contextlib.redirect_stdout(io.StringIO()): return publisher.make_plan(self.root)

    def test_publish_only_selected_source_and_preserve_notebook_original(self):
        original=(self.root/'marker.ipynb').read_bytes()
        plan=self.plan();selected={x['path'] for x in plan['selected']}
        self.assertEqual(selected,{'new_research.py','experiment_config.json','marker.ipynb'})
        with contextlib.redirect_stdout(io.StringIO()): commit=publisher.publish(self.root)
        self.assertEqual(self.git('rev-parse','origin/main'),commit)
        published=self.git('ls-tree','-r','--name-only','HEAD')
        self.assertNotIn('secret.py',published);self.assertNotIn('dataset_dump',published);self.assertNotIn('feature.pt',published)
        self.assertNotIn('OUTPUT_DATA',self.git('show','HEAD:marker.ipynb'))
        backups=list((self.root/publisher.STATE/'notebook_backups').rglob('marker.ipynb'))
        self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),original)
        self.assertIn('SECRET_PATIENT',(self.root/'dataset_dump.json').read_text())

    def test_file_changed_since_plan_blocks_commit(self):
        self.plan();head=self.git('rev-parse','HEAD')
        (self.root/'new_research.py').write_text('# edit after plan\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            publisher.publish(self.root)
        self.assertEqual(self.git('rev-parse','HEAD'),head)
        self.assertEqual(self.git('diff','--cached','--name-only'),'')

    def test_failed_push_retries_same_commit_and_preserves_later_edits(self):
        self.plan()
        real_git=publisher.git
        def fail_push(root,*args,**kwargs):
            if args[:1]==('push',):
                return subprocess.CompletedProcess(args,1,b'',b'simulated authentication failure')
            return real_git(root,*args,**kwargs)
        with patch.object(publisher,'git',side_effect=fail_push), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'Commit preserved locally'):
                publisher.publish(self.root)
        commit=self.git('rev-parse','HEAD')
        self.assertNotEqual(self.git('rev-parse','origin/main'),commit)
        (self.root/'new_research.py').write_text('# later local work\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(publisher.publish(self.root),commit)
            self.assertEqual(publisher.publish(self.root),commit)
        self.assertEqual(self.git('rev-parse','HEAD'),commit)
        self.assertEqual(self.git('rev-parse','origin/main'),commit)
        self.assertEqual((self.root/'new_research.py').read_text(),'# later local work\n')
        self.assertEqual(len(list((self.root/publisher.STATE/'notebook_backups').rglob('marker.ipynb'))),1)

    def test_retry_rejects_unrelated_commit_or_changed_plan(self):
        self.plan()
        with contextlib.redirect_stdout(io.StringIO()): publisher.publish(self.root)
        path=self.root/publisher.STATE/'plan.json'
        original=path.read_text()
        plan=json.loads(original)
        plan['selected'][0]['publish_sha256']='0'*64
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'differs from reviewed plan'):
            publisher.publish(self.root)
        path.write_text(original)
        self.git('commit','--allow-empty','-m','unrelated work')
        remote_before=self.git('rev-parse','origin/main')
        with self.assertRaisesRegex(ValueError,'HEAD changed'):
            publisher.publish(self.root)
        self.assertEqual(self.git('rev-parse','origin/main'),remote_before)

    def test_existing_staged_work_is_not_committed(self):
        self.git('add','new_research.py')
        with self.assertRaisesRegex(ValueError,'staged'):
            self.plan()

    def test_known_pathomics_configs_require_schema_and_eta_requires_shebang(self):
        cfg={'cohort_csv':'data/cohort.csv','stain_csv':'data/stain.csv','baseline_dir':'data/features','output':'results/pathomics','sources':{}}
        p=self.root/'pathomics_all_wsi.json'
        p.write_text(json.dumps(cfg))
        self.assertEqual(publisher.inspect_file(self.root,p.name)[0]['kind'],'configuration')
        cfg['rows']=[{'patient_id':'PRIVATE'}]
        p.write_text(json.dumps(cfg))
        self.assertIsNone(publisher.inspect_file(self.root,p.name)[0])
        eta=self.root/'eta.txt'
        eta.write_text('#!/usr/bin/env bash\nps -ef\n')
        self.assertEqual(publisher.inspect_file(self.root,eta.name)[0]['kind'],'code')
        eta.write_text('last run ended at 12:00')
        self.assertIsNone(publisher.inspect_file(self.root,eta.name)[0])

if __name__=='__main__': unittest.main()
