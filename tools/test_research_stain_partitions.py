"""Check the recovery comparison's patient separation and preserved baseline roles."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prepare_smc_research_stains import extend_partitions


class RecoveryPartitions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.old=pd.DataFrame([dict(event_id=f'event{i}',case_id=f'patient{i}',label=i//10) for i in range(20)])
        self.new=pd.DataFrame([dict(event_id=f'new-event{i}',case_id=f'new-patient{i}',label=1) for i in range(3)])
        self.all=pd.concat([self.old,self.new],ignore_index=True)
        for fold in range(5):
            p=self.old.copy();p['role']=['test' if i%5==fold else 'stop' if i%5==(fold+1)%5 else 'fit' for i in range(20)]
            folder=self.root/'seed1'/f'fold_{fold}';folder.mkdir(parents=True)
            p.to_csv(folder/'partitions.csv',index=False)

    def tearDown(self):self.temp.cleanup()

    def test_existing_roles_stay_fixed_and_new_patients_have_one_test_fold(self):
        parts,_,_=extend_partitions(self.root,self.all,self.new,[1])
        for fold in range(5):
            old=pd.read_csv(self.root/'seed1'/f'fold_{fold}'/'partitions.csv')
            actual=parts[1,fold]
            pd.testing.assert_frame_equal(actual[actual.event_id.isin(self.old.event_id)].sort_values('event_id').reset_index(drop=True),old.sort_values('event_id').reset_index(drop=True))
        for event in self.new.event_id:
            roles=[p.set_index('event_id').loc[event,'role'] for p in parts.values()]
            self.assertEqual(roles.count('test'),1);self.assertEqual(roles.count('stop'),1);self.assertEqual(roles.count('fit'),3)

    def test_recovered_patient_overlap_is_rejected(self):
        new=self.new.copy();new.loc[0,'case_id']=self.old.loc[0,'case_id']
        with self.assertRaisesRegex(ValueError,'already exists'):
            extend_partitions(self.root,pd.concat([self.old,new],ignore_index=True),new,[1])

    def test_reference_clinical_label_change_is_rejected(self):
        altered=self.all.copy();altered.loc[0,'label']=1
        with self.assertRaises(AssertionError):extend_partitions(self.root,altered,self.new,[1])


if __name__=='__main__':unittest.main()
