"""Check batch-one gradients, sample exposure and paired end-to-end training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.imbalance_control import balanced_weights, make_sampler, training_loss
from utils.threshold_control import patient_folds, stop_partition, assert_disjoint


class ImbalanceTests(unittest.TestCase):
    def test_weight_changes_positive_gradient_at_batch_one(self):
        w = balanced_weights([0]*99+[1]); label = torch.tensor([1])
        plain = torch.tensor([[.2, -.3]], requires_grad=True)
        weighted = plain.detach().clone().requires_grad_()
        default = plain.detach().clone().requires_grad_()
        training_loss(plain, label, 'natural_ce', w).backward()
        training_loss(weighted, label, 'natural_weighted_ce', w).backward()
        F.cross_entropy(default, label, weight=w).backward()
        self.assertTrue(torch.allclose(weighted.grad, plain.grad*w[1]))
        self.assertTrue(torch.allclose(default.grad, plain.grad))
        self.assertEqual(w[1].item(), 50)

    def test_sampler_exposure_and_missing_class(self):
        labels = [0]*990+[1]*10
        for condition in ['natural_ce', 'natural_weighted_ce']:
            indices = list(make_sampler(labels, condition, torch.Generator().manual_seed(1)))
            self.assertEqual(len(set(indices)), 1000)
            self.assertEqual(sum(labels[i] for i in indices), 10)
        indices = list(make_sampler(labels, 'balanced_sampler', torch.Generator().manual_seed(1)))
        self.assertEqual(len(indices), 1000)
        self.assertTrue(350 < sum(labels[i] for i in indices) < 650)
        with self.assertRaises(ValueError):
            balanced_weights([0, 0])

    def test_stop_patient_isolation(self):
        events = pd.DataFrame(dict(event_id=[f'e{i}' for i in range(30)],
                                   case_id=[f'p{i}' for i in range(30)], label=[1]*10+[0]*20))
        train, test = patient_folds(events, 5, 1)[0]
        fit, stop = stop_partition(train, 1010)
        assert_disjoint(fit, stop, test)
        self.assertEqual(set(fit.event_id) | set(stop.event_id), set(train.event_id))
        self.assertTrue(fit.label.nunique() == stop.label.nunique() == 2)

    def test_full_cpu_run_resume_and_paired_initializations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); task = root/'manifest'/'acr_high'; task.mkdir(parents=True)
            features = root/'features'/'pt_files'; features.mkdir(parents=True)
            events = pd.DataFrame(dict(event_id=[f'e{i}' for i in range(30)],
                                       case_id=[f'p{i}' for i in range(30)], label=[1]*10+[0]*20))
            events.to_csv(task/'events.csv', index=False)
            pd.DataFrame(dict(event_id=events.event_id, slide_id=events.event_id, stain_group='HE')).to_csv(task/'event_slides.csv', index=False)
            torch.manual_seed(7)
            for event in events.event_id:
                torch.save(torch.randn(2, 1536), features/f'{event}.pt')
            splits = task/'splits'/'seed1'; splits.mkdir(parents=True)
            for fold, (train, test) in enumerate(patient_folds(events, 5, 1)):
                pd.DataFrame({'train': pd.Series(train.event_id.to_list()), 'val': pd.Series(test.event_id.to_list())}).to_csv(splits/f'splits_{fold}.csv', index=False)
            command = [sys.executable, str(PROJECT/'tools/run_smc_imbalance_control.py'),
                       '--manifest-root', str(root/'manifest'), '--feature-dir', str(features.parent),
                       '--results-root', str(root/'results'), '--device', 'cpu', '--seeds', '1',
                       '--max-epochs', '1', '--min-epochs', '1', '--patience', '1']
            env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            for action in ['audit', 'run', 'run', 'summarize']:
                result = subprocess.run(command+['--action', action], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            metrics = pd.read_csv(root/'results'/'per_seed_metrics.csv')
            self.assertEqual(len(metrics), 3)
            self.assertTrue(metrics.events.eq(30).all())
            for folder in (root/'results'/'seed1').glob('fold_*'):
                hashes = []
                for condition in ['balanced_sampler', 'natural_ce', 'natural_weighted_ce']:
                    config = json.loads((folder/condition/'fit_config.json').read_text())
                    hashes.append(config['initialization_sha256'])
                    log = pd.read_csv(folder/condition/'epochs.csv')
                    self.assertEqual(int(log.draws.iloc[0]), config['fit_events'])
                    if condition != 'balanced_sampler':
                        self.assertEqual(int(log.positive_draws.iloc[0]), config['fit_positive'])
                        self.assertEqual(int(log.unique_events.iloc[0]), config['fit_events'])
                self.assertEqual(len(set(hashes)), 1)
            failed = subprocess.run(command+['--action', 'run', '--patience', '2'], env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Inputs/code/settings changed', failed.stderr)


if __name__ == '__main__':
    unittest.main()
