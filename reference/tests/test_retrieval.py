"""Regression checks for evidence eligibility and facet routing."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import search_reference as search
from validate_reference import validate
from coverage import load


class RetrievalTests(unittest.TestCase):
    def test_source_provenance_and_cross_references(self):
        errors, _, counts = validate(strict_sources=True)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(counts['papers'], 20)

    def test_reviewed_only_even_if_atom_is_reviewed_but_card_is_draft(self):
        original = search.load_json
        def altered(path):
            value = original(path)
            if path.name == 'jaume2024_madeleine.json':
                value['review_status'] = 'draft'
            return value
        with patch.object(search, 'load_json', side_effect=altered):
            _, rows, *_ = search.search('MADELEINE stain fusion', False, 10)
        self.assertNotIn('jaume2024_madeleine', {a['paper_id'] for _, a in rows})

    def test_multiple_filters_are_conjunctive(self):
        _, rows, *_ = search.search('MADELEINE stain fusion', False, 10,
                                   {'stains': 'IHC', 'methods': 'cross_stain_alignment'})
        self.assertEqual({a['paper_id'] for _, a in rows}, {'jaume2024_madeleine'})
        _, none, *_ = search.search('MADELEINE', False, 10,
                                   {'organ_scope': 'heart', 'methods': 'cross_stain_alignment'})
        self.assertEqual(none, [])

    def test_shortcut_query_finds_source_bias(self):
        _, rows, *_ = search.search('Exp3 MRXS acquisition shortcut', False, 5)
        self.assertIn('howard2021_site_signatures', {a['paper_id'] for _, a in rows})

    def test_multiscale_query_distinguishes_spatial_fusion(self):
        _, rows, *_ = search.search('DSMIL multiscale scale fusion', False, 5)
        self.assertIn('atom-dsmil-spatial-fusion', {a['atom_id'] for _, a in rows})

    def test_unknown_query_does_not_invent_evidence(self):
        self.assertEqual(search.search('zzzxnonexistent', False, 5)[1], [])

    def test_pending_queue_does_not_leak_into_search(self):
        candidates = {r['paper_id'] for r in load('review_queue.json')['candidates']}
        _, rows, *_ = search.search('CONCH clinical biopsy', False, 100)
        self.assertFalse(candidates & {a['paper_id'] for _, a in rows})

    def test_cli_emits_clean_json_and_rejects_zero_top_k(self):
        result = subprocess.run([sys.executable, str(TOOLS/'search_reference.py'),
                                 'MADELEINE', '--json', '--focus', 'stain_fusion'],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['evidence'])
        invalid = subprocess.run([sys.executable, str(TOOLS/'search_reference.py'),
                                  'MIL', '--top-k', '0'], capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
