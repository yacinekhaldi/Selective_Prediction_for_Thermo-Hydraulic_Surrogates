"""Check U-bend-specific adaptations without rerunning the scientific study."""
import unittest
import numpy as np
import pandas as pd
import ubend_experiments as U

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=pd.read_csv(U.DATA/'ubend_longform.csv');cls.df['group_id']=cls.df.sample_id
    def test_signed_inputs_are_not_log_transformed(self):
        x=U.xspace(self.df.iloc[:20]);self.assertEqual(x.shape,(20,28))
        np.testing.assert_array_equal(x,self.df.iloc[:20][U.FEATURES].to_numpy())
        self.assertTrue((U.xspace(self.df)<0).any())
    def test_all_roles_disjoint_and_shift_definitions(self):
        for seed in U.SEEDS:
            for fold in U.FOLDS:
                p=U.partitions(self.df,seed,fold)
                roles=['fit','tune','scale','cal','test']+(['id_test'] if fold not in ['random_supplied','grouped_random'] else [])
                for i,a in enumerate(roles):
                    for b in roles[:i]:self.assertFalse(set(p[a].case_uid)&set(p[b].case_uid),(seed,fold,a,b))
                if fold not in ['random_supplied','grouped_random']:
                    supported,held=U.shift_mask(self.df,fold)
                    self.assertEqual(set(p['test'].case_uid),set(self.df.loc[held,'case_uid']))
                    for role in ['fit','tune','scale','cal','id_test']:self.assertTrue(set(p[role].case_uid)<=set(self.df.loc[supported,'case_uid']))
    def test_supplied_split_and_no_failure_sentinels(self):
        p=U.partitions(self.df,42,'random_supplied')
        self.assertEqual(len(p['test']),895)
        self.assertEqual(sum(len(p[k]) for k in ['fit','tune','scale','cal']),8051)
        self.assertTrue(self.df.DelP_Pa.between(0,10000,inclusive='neither').all())
        self.assertEqual(self.df.groupby(U.FEATURES).ngroups,len(self.df))
    def test_tie_metric_and_constant_signal(self):
        e=np.array([2.,20.,5.,10.]);s=np.ones(4)
        _,r=U.tie_curve(s,e);np.testing.assert_allclose(r,e.mean())
        self.assertTrue(np.isnan(U.metrics(s,e,np.ones(4,bool),5)['aurc']))
        np.testing.assert_array_equal(U.scaled(np.zeros(3),np.zeros(5)),0)

if __name__=='__main__':unittest.main()
