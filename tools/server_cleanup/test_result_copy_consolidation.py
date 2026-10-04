import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consolidate_result_copies as copies


class CopyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'results').mkdir()
        self.source = self.root / 'results/copy.csv'
        self.target = self.root / 'results/canonical.csv'
        self.source.write_bytes(b'key,value\r\na,1\r\na,1\r\nb,2\r\n')
        self.target.write_bytes(b'\xef\xbb\xbfkey,value\nb,2\na,1\na,1\n')
        self.entries = [{'action': 'deduplicate_csv', 'source': 'results/copy.csv',
                         'target': 'results/canonical.csv', 'source_sha256': copies.sha(self.source)}]

    def test_equal_cells_preserve_original_bytes_and_repeat_safely(self):
        source, target = self.source.read_bytes(), self.target.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            copies.run(self.root, 'apply', self.entries)
            copies.run(self.root, 'apply', self.entries)
        self.assertFalse(self.source.exists())
        self.assertEqual(self.target.read_bytes(), target)
        self.assertEqual((self.root / copies.STATE / 'copies/results/copy.csv').read_bytes(), source)

    def test_repeated_row_count_mismatch_blocks_move(self):
        self.target.write_bytes(b'key,value\na,1\nb,2\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            copies.run(self.root, 'apply', self.entries)
        self.assertTrue(self.source.exists())

    def test_rename_and_conflicting_destination(self):
        self.entries[0].update(action='rename', target='results/descriptive.csv')
        destination = self.root / 'results/descriptive.csv'
        destination.write_bytes(b'unrelated')
        with self.assertRaisesRegex(ValueError, 'destination exists'):
            copies.run(self.root, 'apply', self.entries)
        self.assertTrue(self.source.exists())
        self.entries[0]['target'] = 'results/new/descriptive.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            copies.run(self.root, 'apply', self.entries)
        self.assertEqual(copies.sha(self.root / 'results/new/descriptive.csv'), self.entries[0]['source_sha256'])
        self.assertEqual(destination.read_bytes(), b'unrelated')


if __name__ == '__main__':
    unittest.main()
