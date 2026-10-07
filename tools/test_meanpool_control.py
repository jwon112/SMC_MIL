"""Verify deterministic pooling, fit-only scaling, matched evaluation and safe resume."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'tools'))
from run_smc_meanpool_control import CAP, DIMENSION, embeddings, fit_candidates, pool_slide
from utils.threshold_control import patient_folds, stop_partition


class MeanpoolTests(unittest.TestCase):
    def test_pooling_matches_fixed_evaluation_sampling(self):
        x = torch.arange((CAP+13)*DIMENSION,dtype=torch.float32).reshape(CAP+13,DIMENSION)
        expected = x[torch.linspace(0,len(x)-1,CAP).long()].mean(0)
        torch.testing.assert_close(pool_slide(x),expected)
        with self.assertRaises(ValueError):
            pool_slide(torch.empty(0,DIMENSION))
        with self.assertRaises(ValueError):
            pool_slide(torch.full((1,DIMENSION),float('nan')))

    def test_scaler_uses_fit_only_and_exact_ties_prefer_stronger_regularization(self):
        fit = np.zeros((10,3)); labels = np.array([0,1]*5)
        scaler,model,c,candidates = fit_candidates(fit,labels,np.full((4,3),10000.),np.array([0,1,0,1]))
        np.testing.assert_array_equal(scaler.mean_,[0,0,0])
        self.assertEqual(c,.001)
        self.assertEqual(model.class_weight,None)
        self.assertEqual(len(candidates),4)

    def test_equal_slide_weight_not_patch_count_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); bags=root/'pt_files';bags.mkdir()
            torch.save(torch.full((1,DIMENSION),10.),bags/'a.pt')
            torch.save(torch.zeros(9,DIMENSION),bags/'b.pt')
            events=pd.DataFrame({'event_id':['e']})
            slides=pd.DataFrame({'event_id':['e','e'],'slide_id':['a','b']})
            result=embeddings(events,slides,{'feature_dir':str(root)},root)
            np.testing.assert_array_equal(result,np.full((1,DIMENSION),5.))

    def test_end_to_end_resume_and_changed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); manifest=root/'manifest';manifest.mkdir()
            reference=root/'reference';reference.mkdir()
            bags=root/'features'/'pt_files';bags.mkdir(parents=True)
            events=pd.DataFrame({'event_id':[f'e{i:02}' for i in range(30)],
                                 'case_id':[f'p{i:02}' for i in range(30)],'label':[1]*10+[0]*20})
            slides=pd.DataFrame({'event_id':events.event_id,'slide_id':events.event_id,'stain_group':'HE'})
            events.to_csv(manifest/'events.csv',index=False,lineterminator='\n')
            slides.to_csv(manifest/'event_slides.csv',index=False,lineterminator='\n')
            torch.manual_seed(9)
            inventory={}
            for row in events.itertuples():
                path=bags/f'{row.event_id}.pt'
                torch.save(torch.randn(3,DIMENSION)+float(row.label),path)
                stat=path.stat();inventory[row.event_id]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            conditions=['balanced_sampler','natural_ce','natural_weighted_ce']
            predictions=[]
            for fold,(train,test) in enumerate(patient_folds(events,5,1)):
                fit,stop=stop_partition(train,1010+fold)
                folder=reference/'seed1'/f'fold_{fold}';folder.mkdir(parents=True)
                pd.concat([f.assign(role=role) for role,f in [('fit',fit),('stop',stop),('test',test)]]).to_csv(folder/'partitions.csv',index=False)
                predictions.append(test.assign(probability=np.where(test.label.eq(1),.7,.1),fold=fold))
            for condition in conditions:
                pd.concat(predictions).to_csv(reference/'seed1'/f'{condition}_oof_predictions.csv',index=False)
            protocol={'seeds':[1],'conditions':conditions,'feature_dir':str(bags.parent),'feature_inventory':inventory,
                      'input_sha256':{f'/manifest/{p.name}':hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest.glob('*.csv')}}
            (reference/'protocol.json').write_text(json.dumps(protocol))
            out=root/'out'
            command=[sys.executable,str(PROJECT/'tools/run_smc_meanpool_control.py'),
                     '--manifest-dir',str(manifest),'--reference-root',str(reference),'--output-dir',str(out)]
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            def run(action):
                return subprocess.run(command+['--action',action],capture_output=True,text=True,env=env)
            for action in ['audit','run','summarize']:
                result=run(action)
                self.assertEqual(result.returncode,0,result.stderr)
            predictions_path=out/'seed1/meanpool_logreg_oof_predictions.csv'
            before=predictions_path.read_bytes()
            frame=pd.read_csv(predictions_path)
            self.assertEqual(len(frame),30)
            self.assertEqual(frame.event_id.nunique(),30)
            result=run('run')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('[CACHE]',result.stdout)
            self.assertEqual(result.stdout.count('[SKIP]'),5)
            self.assertEqual(predictions_path.read_bytes(),before)
            selected=json.loads((out/'seed1/fold_0/selection.json').read_text())
            self.assertTrue(selected['no_refit'])
            with (out/'event_embeddings.npz').open('ab') as handle:
                handle.write(b'corruption')
            result=run('run')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Embedding cache changed',result.stderr)
            (manifest/'event_slides.csv').write_text('changed')
            result=run('audit')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Training manifest changed',result.stderr)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
