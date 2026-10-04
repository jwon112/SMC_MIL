"""Verify audit coverage and that label rows/notebook outputs are not exported."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import audit_workspace as audit

class AuditTests(unittest.TestCase):
    def test_dependencies_labels_and_export_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary).resolve()
            root=base/'project';root.mkdir()
            def git(*args):
                subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True)
            git('init');git('config','user.email','test@example.invalid');git('config','user.name','Test')
            (root/'train_smc_event_stain_mil.py').write_text('# active\n')
            git('add','train_smc_event_stain_mil.py');git('commit','-m','fixture')
            (root/'legacy.py').write_text('# server-only research\n')
            (root/'utils').mkdir()
            (root/'utils/use_legacy.py').write_text('from legacy import research\n')
            (root/'notes.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','source':['import legacy'],'outputs':[{'text':'SECRET_NOTEBOOK_OUTPUT'}]}]}))
            result=root/'archive/experiments/early_cv/old_run'
            result.mkdir(parents=True)
            (result/'run_config.json').write_text(json.dumps({'input':'/labels/derived/wsi_curation_v1/labels.csv'}))
            (root/'data/features').mkdir(parents=True)
            (root/'data/features/feature.pt').write_bytes(b'NO_FEATURE_EXPORT')
            (root/'old.log').write_text('NO_LOG_EXPORT')
            labels=base/'labels'
            for name in ('wsi_curation_v1','wsi_curation_v2'):
                folder=labels/'derived'/name;folder.mkdir(parents=True)
                (folder/'labels.csv').write_text('slide_id,case_id,stain_group\nSECRET_SLIDE,SECRET_PATIENT,HE\nSECRET_SLIDE,SECRET_PATIENT,HE\n')
            output=base/'audit_output'
            with patch.object(audit,'process_refs',return_value=([],[])),contextlib.redirect_stdout(io.StringIO()):
                report=audit.audit(root,labels,output)
            legacy=next(x for x in report['root_items'] if x['name']=='legacy.py')
            self.assertFalse(legacy['tracked_by_git'])
            self.assertGreaterEqual(legacy['reference_count'],2)
            table=report['label_files'][0]['table']
            self.assertEqual(table['rows'],2)
            self.assertEqual(table['repeated_nonempty_ids']['slide_id'],1)
            self.assertEqual(len(report['identical_label_files']),1)
            version=next(x for x in report['label_directories'] if x['name']=='wsi_curation_v1')
            self.assertGreater(version['reference_count'],0)
            self.assertTrue(any('archive/experiments' in x['source'] for x in version['references']))
            with zipfile.ZipFile(output/'workspace_audit.zip') as z:
                payload=b'\n'.join(z.read(n) for n in z.namelist())
                for secret in (b'SECRET_SLIDE',b'SECRET_PATIENT',b'SECRET_NOTEBOOK_OUTPUT',b'NO_FEATURE_EXPORT',b'NO_LOG_EXPORT'):
                    self.assertNotIn(secret,payload)
            self.assertTrue((root/'legacy.py').exists())
            self.assertTrue((labels/'derived/wsi_curation_v1/labels.csv').exists())

if __name__=='__main__':
    unittest.main()
