"""Verify patch-cap controls without external WSI files."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.patch_sampling_control import sample_patches


class PatchSamplingControlTest(unittest.TestCase):
    def test_training_subsets_are_nested(self):
        x=torch.arange(100).reshape(50,2).float()
        small=sample_patches(x,8,True,17)
        large=sample_patches(x,16,True,17)
        torch.testing.assert_close(small,large[:8])
        self.assertEqual(len(torch.unique(small[:,0])),8)

    def test_small_bags_are_unchanged(self):
        x=torch.randn(3,8)
        self.assertIs(sample_patches(x,8,True,1),x)
        self.assertIs(sample_patches(x,16,True,1),x)

    def test_validation_fixed_and_global_rng_untouched(self):
        x=torch.arange(100).reshape(50,2).float()
        state=torch.random.get_rng_state().clone()
        sample_patches(x,8,True,7)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        a=sample_patches(x,8,False,1)
        b=sample_patches(x,8,False,999)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a,x[torch.linspace(0,49,8).long()])

    def test_empty_and_invalid_caps_fail(self):
        with self.assertRaises(ValueError):sample_patches(torch.empty(0,8),8,True,1)
        with self.assertRaises(ValueError):sample_patches(torch.ones(5,8),0,True,1)


if __name__=='__main__':unittest.main()
