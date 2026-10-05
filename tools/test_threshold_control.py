"""Verify nested patient isolation, threshold selection and full CPU workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.threshold_control import assert_disjoint, choose_threshold, patient_folds
from tools.run_smc_threshold_control import nested_plan


class ThresholdControlTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame([dict(event_id=f'e{i}', case_id=f'p{i}', label=int(i < 10)) for i in range(30)])

    def test_patient_isolation_and_outer_labels_do_not_change_inner_plan(self):
        events = self.events()
        events = pd.concat([events, pd.DataFrame([dict(event_id='mixed', case_id='p0', label=0)])], ignore_index=True)
        train, test = patient_folds(events, 5, 1)[0]
        parts, _ = nested_plan(train, test, 1, 0)
        changed = test.copy(); changed['label'] = 1-changed.label
        other, _ = nested_plan(train, changed, 1, 0)
        for (fit, stop, calibration), alt in zip(parts, other):
            assert_disjoint(fit, stop, calibration, test)
            self.assertTrue(all(a.equals(b) for a, b in zip((fit, stop, calibration), alt)))
        self.assertEqual(set(pd.concat([x[2] for x in parts]).event_id), set(train.event_id))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            assert_disjoint(events.iloc[:1], events.iloc[:1])

    def test_f1_f2_and_tie_policy(self):
        frame = pd.DataFrame(dict(event_id=['a', 'b', 'c', 'd'], case_id=['a', 'b', 'c', 'd'],
                                  label=[1, 0, 0, 1], probability=[.9, .8, .7, .2]))
        self.assertEqual(choose_threshold(frame, 1), .9)
        self.assertEqual(choose_threshold(frame, 2), .2)

    def test_insufficient_positive_patients_blocks(self):
        events = self.events(); events['label'] = 0; events.loc[0, 'label'] = 1
        with self.assertRaisesRegex(ValueError, 'Too few'):
            patient_folds(events, 3, 1)

    def test_full_cpu_run_resume_and_summary(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); task = root/'manifest'/'acr_high'; task.mkdir(parents=True)
            features = root/'features'/'pt_files'; features.mkdir(parents=True)
            events = self.events(); events.to_csv(task/'events.csv', index=False)
            slides = pd.DataFrame(dict(event_id=events.event_id, slide_id=events.event_id, stain_group='HE'))
            slides.to_csv(task/'event_slides.csv', index=False)
            torch.manual_seed(7)
            for slide in slides.slide_id:
                torch.save(torch.randn(2, 1536), features/f'{slide}.pt')
            split = task/'splits'/'seed1'; split.mkdir(parents=True)
            for fold, (train, test) in enumerate(patient_folds(events, 5, 1)):
                pd.DataFrame({'train': pd.Series(train.event_id.to_list()), 'val': pd.Series(test.event_id.to_list())}).to_csv(split/f'splits_{fold}.csv', index=False)
            command = [sys.executable, str(PROJECT/'tools/run_smc_threshold_control.py'),
                       '--manifest-root', str(root/'manifest'), '--feature-dir', str(features.parent),
                       '--results-root', str(root/'results'), '--device', 'cpu', '--seeds', '1',
                       '--max-epochs', '1', '--min-epochs', '1', '--patience', '1']
            env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            for action in ['audit', 'run', 'run', 'summarize']:
                result = subprocess.run(command+['--action', action], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                if action == 'run':
                    self.assertTrue('[OK]' in result.stdout or '[SKIP]' in result.stdout)
            metrics = pd.read_csv(root/'results'/'per_seed_metrics.csv')
            self.assertEqual(len(metrics), 3)
            self.assertEqual(metrics.auroc.nunique(), 1)
            self.assertEqual(metrics.pr_auc.nunique(), 1)
            self.assertTrue(metrics.events.eq(30).all())
            self.assertTrue(metrics.positive_events.eq(10).all())
            oof = pd.read_csv(root/'results'/'seed1'/'oof_predictions.csv')
            self.assertEqual(oof.event_id.nunique(), 30)
            self.assertTrue(oof.prediction_fixed_0p5.eq((oof.probability >= .5).astype(int)).all())
            protocol = json.loads((root/'results'/'protocol.json').read_text())
            self.assertEqual(protocol['primary'], 'inner_f2')
            changed = subprocess.run(command+['--action', 'run', '--patience', '2'], env=env, capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('Protocol or inputs changed', changed.stderr)


if __name__ == '__main__':
    unittest.main()
