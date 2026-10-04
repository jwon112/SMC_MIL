"""Exercise the first-time Git handover in isolated local repositories."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('sync',Path(__file__).resolve().parents[1]/'sync_server_checkout.py')
sync=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)

class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        def git(*args):
            return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.STDOUT).decode().strip()
        self.git=git
        git('init','-b','main')
        git('config','user.name','Test')
        git('config','user.email','test@example.invalid')
        git('config','core.autocrlf','false')
        (self.root/'reader.py').write_text('old\n')
        (self.root/'unrelated.txt').write_text('original\n')
        git('add','reader.py','unrelated.txt')
        git('commit','-m','base')
        self.base=git('rev-parse','HEAD')
        (self.root/'reader.py').write_text('new git reader\n')
        (self.root/'new_control.py').write_text('same installed control\n')
        meta=self.root/'tools/server_cleanup'
        meta.mkdir(parents=True)
        known={'reader.py':sync.normalized(b'reviewed ZIP reader\n')}
        (meta/'adoption_baselines.json').write_text(json.dumps(known))
        git('add','reader.py','new_control.py','tools')
        git('commit','-m','incoming')
        self.target=git('rev-parse','HEAD')
        git('update-ref','refs/remotes/origin/main',self.target)
        # Only an isolated fixture checkout is reset, never the user's repository.
        git('reset','--hard',self.base)
        (self.root/'reader.py').write_text('reviewed ZIP reader\n')
        (self.root/'new_control.py').write_text('same installed control\n')
        (self.root/'unrelated.txt').write_text('keep local work\n')

    def test_preserves_only_conflicting_reviewed_paths_and_fast_forwards(self):
        with contextlib.redirect_stdout(io.StringIO()):
            sync.synchronize(self.root,'origin/main',False)
            self.assertEqual(self.git('rev-parse','HEAD'),self.base)
            sync.synchronize(self.root,'origin/main',True)
        self.assertEqual(self.git('rev-parse','HEAD'),self.target)
        self.assertEqual((self.root/'reader.py').read_text(),'new git reader\n')
        self.assertEqual((self.root/'unrelated.txt').read_text(),'keep local work\n')
        self.assertEqual(self.git('show','stash@{0}:reader.py'),'reviewed ZIP reader')
        self.assertEqual(self.git('show','stash@{0}^3:new_control.py'),'same installed control')

    def test_unreviewed_edit_is_untouched(self):
        (self.root/'reader.py').write_text('new server work\n')
        with self.assertRaisesRegex(ValueError,'unreviewed'):
            sync.synchronize(self.root,'origin/main',True)
        self.assertEqual(self.git('rev-parse','HEAD'),self.base)
        self.assertEqual((self.root/'reader.py').read_text(),'new server work\n')
        self.assertEqual(self.git('stash','list'),'')

if __name__=='__main__':
    unittest.main()
