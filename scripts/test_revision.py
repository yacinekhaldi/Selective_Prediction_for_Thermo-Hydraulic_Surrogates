"""Numerical checks for leakage boundaries, score saturation and tied ranking."""
import unittest
import numpy as np
import pandas as pd
from revision_experiments import DATA,GROUP,partitions,tie_curve,scaled,metrics

class RevisionChecks(unittest.TestCase):
    def test_ties_are_permutation_invariant(self):
        s=np.array([1,0,1,0,2.]);e=np.array([5,2,4,8,9.])
        expected=tie_curve(s,e)[1]
        for seed in range(10):
            ix=np.random.default_rng(seed).permutation(len(s))
            np.testing.assert_allclose(tie_curve(s[ix],e[ix])[1],expected)
    def test_constant_score_is_random_order_reference(self):
        c,r=tie_curve(np.ones(4),np.array([1,3,5,7.]))
        np.testing.assert_allclose(r,4.)
        self.assertTrue(np.isnan(metrics(np.ones(4),np.arange(4.),np.ones(4,bool),2)['aurc']))
    def test_no_upper_clipping(self):
        z=scaled(np.array([0.,2.,20.]),np.arange(10.))
        self.assertGreater(z[-1],1.)
        np.testing.assert_allclose(scaled(np.zeros(3),np.zeros(10)),0.)
    def test_group_boundaries(self):
        d=pd.read_csv(DATA/'02_processed_data/ML_dataset_longform.csv')
        d['group_id']=d.groupby(GROUP).ngroup()
        for seed in [42,137,271]:
            for fold in ['grouped_random','Re25','Re100','Re250','Re500','G01','G06','combined']:
                p=partitions(d,seed,fold)
                names=['fit','tune','scale','cal','test']
                for i,a in enumerate(names):
                    for b in names[:i]:self.assertFalse(set(p[a].group_id)&set(p[b].group_id))
                self.assertFalse(set(p['test'].case_uid)&set(p['fit'].case_uid))

if __name__=='__main__':unittest.main()
