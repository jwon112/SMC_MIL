"""CPU checks for the same-architecture presence-mask control."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.stain_aware_event_mil import StainAwareEventMIL


class PresenceMaskControlTest(unittest.TestCase):
    def pair(self):
        torch.manual_seed(11)
        observed = StainAwareEventMIL(8, 4, 0, presence_mask_values="observed")
        torch.manual_seed(11)
        zero = StainAwareEventMIL(8, 4, 0, presence_mask_values="zero")
        return observed, zero

    def test_identical_parameters_and_only_mask_input_changes(self):
        observed, zero = self.pair()
        self.assertEqual(observed.classifier[0].in_features, 15)
        for name, value in observed.state_dict().items():
            torch.testing.assert_close(value, zero.state_dict()[name], rtol=0, atol=0)
        inputs = {}
        observed.classifier[0].register_forward_pre_hook(lambda m, a: inputs.update(observed=a[0].detach()))
        zero.classifier[0].register_forward_pre_hook(lambda m, a: inputs.update(zero=a[0].detach()))
        slides = [("HE", torch.randn(5, 8)), ("other", torch.randn(3, 8))]
        observed(slides); zero(slides)
        torch.testing.assert_close(inputs['observed'][:, :-3], inputs['zero'][:, :-3], rtol=0, atol=0)
        torch.testing.assert_close(inputs['observed'][:, -3:], torch.tensor([[1., 0., 1.]]))
        torch.testing.assert_close(inputs['zero'][:, -3:], torch.zeros(1, 3))

    def test_zero_mask_has_no_data_gradient_and_roundtrips(self):
        _, zero = self.pair()
        slides = [("HE", torch.randn(5, 8))]
        logits, _ = zero(slides)
        logits.sum().backward()
        self.assertEqual(torch.count_nonzero(zero.classifier[0].weight.grad[:, -3:]).item(), 0)
        restored = StainAwareEventMIL(8, 4, 0, presence_mask_values="zero")
        restored.load_state_dict(zero.state_dict(), strict=True)
        torch.testing.assert_close(zero(slides)[0], restored(slides)[0])

    def test_legacy_default_and_invalid_control(self):
        observed, _ = self.pair()
        legacy = StainAwareEventMIL(8, 4, 0)
        legacy.load_state_dict(observed.state_dict(), strict=True)
        self.assertEqual(legacy.presence_mask_values, "observed")
        with self.assertRaises(ValueError):
            StainAwareEventMIL(include_presence_masks=False, presence_mask_values="zero")


if __name__ == '__main__':
    unittest.main()
