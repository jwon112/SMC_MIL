"""Check stain invariance and a real paired CPU fit on synthetic feature bags."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

import pandas as pd
import torch

from recovered_stain_control import assert_draws_match, fit_agnostic, make_model
from run_smc_imbalance_control import fit_and_evaluate
from models.stain_aware_event_mil import StainAwareEventMIL
from train_smc_event_stain_mil import seed_all
from utils.imbalance_control import model_hash
from prepare_smc_research_stains import digest


class RecoveredStainControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def aware_config(self):
        seed_all(1)
        model = StainAwareEventMIL(1536, 128, .25)
        return {'initialization_sha256': model_hash(model)}, model

    def test_agnostic_logits_ignore_stain_labels_with_shared_initial_encoder(self):
        config, aware = self.aware_config()
        model, shared_hash = make_model(1, 0, config, 'cpu')
        self.assertEqual(shared_hash, model_hash(aware.patch_encoder))
        self.assertEqual(model.branch_groups, ('all',))
        self.assertTrue(model.include_presence_masks)
        self.assertLess(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in aware.parameters()))
        model.eval()
        bags = [torch.randn(3, 1536), torch.randn(4, 1536)]
        with torch.no_grad():
            original, _ = model([('HE', bags[0]), ('IHC', bags[1])])
            relabeled, _ = model([('other', bags[0]), ('not-a-stain', bags[1])])
        torch.testing.assert_close(original, relabeled, rtol=0, atol=0)

    def test_mismatched_reference_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'saved aware initialization'):
            make_model(1, 0, {'initialization_sha256': 'wrong'}, 'cpu')

    def test_exposure_change_is_rejected(self):
        actual = pd.DataFrame([dict(epoch=1, step=0, event_id='toy-a', label=1)])
        changed = actual.copy(); changed.loc[0, 'event_id'] = 'toy-b'
        with self.assertRaises(AssertionError):
            assert_draws_match(actual, changed)

    def test_real_cpu_fits_keep_event_exposure_and_held_out_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root/'features'; (features/'pt_files').mkdir(parents=True)
            events = pd.DataFrame([dict(event_id=f'toy-event-{i}', case_id=f'toy-case-{i}', label=i % 2) for i in range(8)])
            slides = pd.DataFrame([dict(event_id=r.event_id, slide_id=f'toy-slide-{i}', stain_group=['HE', 'IHC', 'other'][i % 3]) for i, r in enumerate(events.itertuples())])
            for row in slides.itertuples():
                torch.save(torch.randn(4, 1536), features/'pt_files'/f'{row.slide_id}.pt')
            args = SimpleNamespace(feature_dir=features, device='cpu', max_epochs=2, min_epochs=1, patience=2)
            fit, stop, test = events.iloc[:4], events.iloc[4:6], events.iloc[6:]
            aware = root/'aware'; aware.mkdir()
            agnostic = root/'agnostic'; agnostic.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                fit_and_evaluate(fit, stop, test, slides, args, 1, 0, 'balanced_sampler', aware)
                fit_agnostic(fit, stop, test, slides, args, 1, 0, aware, agnostic)
            prediction = pd.read_csv(agnostic/'outer_predictions.csv')
            self.assertEqual(set(prediction.event_id), set(test.event_id))
            self.assertTrue(prediction.probability.between(0, 1).all())
            self.assertEqual(assert_draws_match(pd.read_csv(agnostic/'training_draws.csv'),
                                               pd.read_csv(aware/'training_draws.csv')), 2)
            selection = json.loads((agnostic/'selection.json').read_text())
            self.assertEqual(selection['identical_event_draw_epochs'], 2)

    def test_command_audit_accepts_frozen_reference_and_rejects_changed_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); prepared = root/'prepared'; aware = root/'aware'
            features = root/'features'/'pt_files'; features.mkdir(parents=True)
            events = pd.DataFrame([dict(event_id=f'toy-event-{i:04}', case_id=f'toy-case-{i:04}',
                                       label=int(i < 14 or i >= 575)) for i in range(578)])
            slides = pd.DataFrame([dict(event_id=events.iloc[i % 578].event_id,
                                       slide_id=f'toy-slide-{i:04}', stain_group='HE') for i in range(1269)])
            files = {}
            def save(relative, frame):
                path = prepared/relative; path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False); files[relative] = digest(path)
            save('restored7/acr_high/events.csv', events)
            save('restored7/acr_high/event_slides.csv', slides)
            save('baseline575/acr_high/events.csv', events.iloc[:575])
            for seed in [1, 11, 21, 31, 41]:
                predictions = []
                for fold in range(5):
                    part = events.copy()
                    part['role'] = ['test' if i % 5 == fold else 'stop' if i % 5 == (fold+1) % 5 else 'fit' for i in range(578)]
                    save(f'partitions/seed{seed}/fold_{fold}/partitions.csv', part)
                    folder = aware/'restored7'/f'seed{seed}'/f'fold_{fold}'; folder.mkdir(parents=True)
                    config = dict(seed=seed, fold=fold, condition='balanced_sampler', mask='observed',
                                  patch_cap=2048, fit_events=int(part.role.eq('fit').sum()),
                                  fit_positive=int(part.loc[part.role.eq('fit'), 'label'].sum()))
                    (folder/'fit_config.json').write_text(json.dumps(config))
                    (folder/'selection.json').write_text('{}')
                    (folder/'training_draws.csv').write_text('epoch,step,event_id,label\n')
                    prediction = part.loc[part.role.eq('test'), ['event_id', 'case_id', 'label']].copy()
                    prediction['probability'] = .4; prediction['fold'] = fold
                    prediction.to_csv(folder/'outer_predictions.csv', index=False); predictions.append(prediction)
                pd.concat(predictions, ignore_index=True).to_csv(aware/'restored7'/f'seed{seed}'/'oof_predictions.csv', index=False)
            inventory = {}
            for sid in slides.slide_id:
                file = features/f'{sid}.pt'; file.touch(); stat = file.stat()
                inventory[sid] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            dependency = root/'frozen_source.txt'; dependency.write_text('original')
            manifest = dict(seeds=[1, 11, 21, 31, 41], output_sha256_lf=files)
            (prepared/'manifest.json').write_text(json.dumps(manifest))
            protocol = dict(cohorts=['restored7'], condition='balanced_sampler', threshold=.5,
                            prepared_outputs_sha256_lf=files, seeds=manifest['seeds'],
                            source_sha256_lf={str(dependency): digest(dependency)},
                            feature_dir=str(features.parent), feature_inventory=inventory)
            (aware/'protocol.json').write_text(json.dumps(protocol))
            before = {str(p): digest(p) for p in aware.rglob('*') if p.is_file()}
            command = [sys.executable, str(Path(__file__).with_name('run_smc_recovered_stain_control.py')),
                       '--action', 'audit', '--manifest-root', str(prepared), '--aware-root', str(aware),
                       '--results-root', str(root/'control')]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue(json.loads(first.stdout)['ready_to_train'])
            self.assertEqual(before, {str(p): digest(p) for p in aware.rglob('*') if p.is_file()})
            dependency.write_text('changed')
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Aware reference dependency changed', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
