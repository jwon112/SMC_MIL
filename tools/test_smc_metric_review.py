"""Check clustered comparisons and event-level integrity independently of real data."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from sklearn.metrics import fbeta_score
from reanalyze_smc_event_controls import evaluate, check_same, cluster_deltas, load


class ReviewTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'event_id':['a','b','c','d','e','f'],
                             'case_id':['p1','p1','p2','p2','p3','p4'],
                             'label':[1,0,1,0,0,0],
                             'probability':[.8,.4,.3,.7,.1,.2]})

    def test_identity_pair_has_zero_cluster_interval(self):
        frame=self.frame()
        rows=cluster_deltas(frame,frame.copy(),100,20261005)
        self.assertGreater(rows[0]['valid_resamples'],50)
        for row in rows:
            for key in ('delta_right_minus_left','ci_low','ci_high'):
                self.assertEqual(row[key],0)

    def test_f2_and_patient_mapping(self):
        frame=self.frame()
        result=evaluate(frame)
        self.assertAlmostEqual(result['f2'],fbeta_score(frame.label,frame.probability>=.5,beta=2))
        self.assertEqual(result['patients'],4)
        altered=frame.copy();altered.loc[0,'case_id']='different'
        with self.assertRaisesRegex(ValueError,'case_id'):
            check_same(frame,altered)

    def test_duplicate_event_and_patient_fold_leakage_block(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'oof.csv'
            frame=self.frame();frame['fold']=[0,1,0,0,1,1]
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'multiple held-out folds'):
                load(path,{})
            frame['fold']=[0,0,1,1,2,3];frame.loc[1,'event_id']='a'
            frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'coverage'):
                load(path,{})


if __name__=='__main__': unittest.main()
