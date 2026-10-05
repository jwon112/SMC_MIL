"""Verify positive-class reporting on rare positives and degenerate predictions."""
import math
import unittest
import pandas as pd
from sklearn.metrics import f1_score, precision_score, matthews_corrcoef
from summarize_smc_event_stain_mil import metrics


class MetricTests(unittest.TestCase):
    def frame(self, tp, fp, tn, fn):
        return pd.DataFrame({'label':[1]*tp+[0]*fp+[0]*tn+[1]*fn,
                             'probability':[.8]*(tp+fp)+[.2]*(tn+fn)})

    def test_rare_positive_counts_agree_with_sklearn(self):
        frame=self.frame(tp=5,fp=6,tn=555,fn=9)
        result=metrics(frame,.5)
        prediction=frame.probability>=.5
        self.assertEqual([result[k] for k in ('tp','fp','tn','fn')],[5,6,555,9])
        self.assertEqual(result['events'],575)
        self.assertAlmostEqual(result['f1'],.4)
        self.assertAlmostEqual(result['f1'],f1_score(frame.label,prediction))
        self.assertAlmostEqual(result['precision'],precision_score(frame.label,prediction))
        self.assertAlmostEqual(result['mcc'],matthews_corrcoef(frame.label,prediction))

    def test_all_negative_predictions_have_zero_positive_f1(self):
        result=metrics(self.frame(tp=0,fp=0,tn=561,fn=14),.5)
        self.assertEqual(result['f1'],0)
        self.assertEqual(result['mcc'],0)
        self.assertTrue(math.isnan(result['precision']))
        self.assertEqual(result['sensitivity'],0)
        self.assertEqual(result['specificity'],1)

    def test_threshold_is_reported_and_includes_equal_probability(self):
        frame=pd.DataFrame({'label':[0,0,1,1], 'probability':[.1,.5,.5,.9]})
        default=metrics(frame,.5)
        higher=metrics(frame,.7)
        self.assertEqual(default['threshold'],.5)
        self.assertEqual((default['tp'],default['fp']),(2,1))
        self.assertEqual((higher['tp'],higher['fp'],higher['fn']),(1,0,1))
        self.assertEqual(default['auroc'],higher['auroc'])
        self.assertEqual(default['pr_auc'],higher['pr_auc'])


if __name__=='__main__': unittest.main()
