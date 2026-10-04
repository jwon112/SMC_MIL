"""Exercise result selection and staging with an isolated Git repository."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish_server_results as results


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'core.autocrlf', 'false')
        ignore = Path(__file__).resolve().parents[2] / '.gitignore'
        (self.root / '.gitignore').write_bytes(ignore.read_bytes())
        self.git('add', '.gitignore')
        self.git('commit', '-m', 'baseline')
        self.files = {
            'results/current/metrics.csv': b'auroc\n0.9\n',
            'results/current/config.json': b'{"epochs":50}\n',
            'results/current/train.log': b'epoch 1\n',
            'results_old/old/summary.txt': b'old result\n',
            'results_old2/old/plot.svg': b'<svg/>\n',
            'archive/experiments/early_cv/run/metrics.csv': b'auc\n0.8\n',
            'results/current/fold_0.pt': b'weights',
            'results/current/features.npy': b'features',
            'results/current/bundle.zip': b'archive',
            'archive/installers/package/readme.txt': b'installer',
        }
        for name, content in self.files.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)

    def git(self, *args):
        return results.git(self.root, *args).stdout.decode().strip()

    def plan(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return results.plan(self.root, **kwargs)

    def test_result_scope_and_stage_without_commit_or_deletion(self):
        head = self.git('rev-parse', 'HEAD')
        plan = self.plan()
        self.assertEqual(len(plan['selected']), 6)
        self.assertEqual(len(plan['held']), 3)
        with contextlib.redirect_stdout(io.StringIO()):
            results.stage(self.root)
        self.assertEqual(set(self.git('diff', '--cached', '--name-only').splitlines()),
                         {x['path'] for x in plan['selected']})
        self.assertEqual(self.git('rev-parse', 'HEAD'), head)
        for name, content in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), content)

    def test_changed_training_log_blocks_before_staging(self):
        self.plan()
        (self.root / 'results/current/train.log').write_bytes(b'epoch 2\n')
        with self.assertRaisesRegex(ValueError, 'changed since plan'):
            results.stage(self.root)
        self.assertEqual(self.git('diff', '--cached', '--name-only'), '')

    def test_size_limits(self):
        p = self.plan(max_file_mib=1 / results.MIB)
        self.assertEqual(p['selected'], [])
        self.plan(max_total_mib=1 / results.MIB)
        with self.assertRaisesRegex(ValueError, 'total exceeds'):
            results.stage(self.root)
        self.assertEqual(self.git('diff', '--cached', '--name-only'), '')

    def test_deferred_active_run_can_change_without_blocking_other_results(self):
        neighbor = self.root / 'results/current_finished/metrics.csv'
        neighbor.parent.mkdir()
        neighbor.write_bytes(b'auc\n0.95\n')
        plan = self.plan(excludes=['results/current/'])
        self.assertEqual(plan['excludes'], ['results/current'])
        self.assertIn('results/current_finished/metrics.csv', [x['path'] for x in plan['selected']])
        (self.root / 'results/current/train.log').write_bytes(b'epoch 2\n')
        (self.root / 'results/current/metrics.csv').write_bytes(b'auc\n0.8\n')
        with contextlib.redirect_stdout(io.StringIO()):
            results.stage(self.root)
        staged = self.git('diff', '--cached', '--name-only').splitlines()
        self.assertFalse(any(name.startswith('results/current/') for name in staged))
        self.assertIn('results/current_finished/metrics.csv', staged)
        self.assertEqual((self.root / 'results/current/train.log').read_bytes(), b'epoch 2\n')

    def test_exact_file_exclusion_and_invalid_paths(self):
        plan = self.plan(excludes=['results/current/train.log'])
        names = {item['path'] for item in plan['selected']}
        self.assertNotIn('results/current/train.log', names)
        self.assertIn('results/current/metrics.csv', names)
        for path in ('results/../tools', '/results/current', 'results/*', 'tools'):
            with self.assertRaisesRegex(ValueError, 'literal repository-relative'):
                self.plan(excludes=[path])

    def test_unrelated_staged_work_blocks(self):
        self.plan()
        (self.root / 'other.py').write_text('# unrelated\n')
        self.git('add', 'other.py')
        with self.assertRaisesRegex(ValueError, 'Existing staged'):
            results.stage(self.root)
        self.assertEqual(self.git('diff', '--cached', '--name-only'), 'other.py')


if __name__ == '__main__':
    unittest.main()
