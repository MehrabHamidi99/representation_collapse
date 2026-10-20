import unittest
import numpy as np
from representation_collapse import tgrab_dwell as d

class DwellTests(unittest.TestCase):
    def test_schedule_exact_horizon_and_phase(self):
        for length in d.DWELLS:
            for start in ('A','B'):
                rows=list(d.schedule(length,start))
                self.assertEqual(len(rows),2048)
                self.assertEqual(rows[0],(0,start,0,0))
                self.assertNotEqual(rows[length][1],start)
                self.assertEqual(rows[2*length][1],start)
                self.assertEqual(sum(r[1]=='A' for r in rows),1024)
                for phase in range(length):
                    self.assertEqual(sum(r[2]==phase for r in rows),2048//length)
    def test_invalid_schedule(self):
        for length,start,horizon in [(7,'A',2048),(True,'A',2048),(32,'C',2048),(32,'A',100)]:
            with self.assertRaises(ValueError):list(d.schedule(length,start,horizon))
    def test_reductions_sign_and_counts(self):
        for length in d.DWELLS:
            rows=[{'t':t,'motif':m,'phase':p,'cycle':c,'f1':1-p/length} for t,m,p,c in d.schedule(length,'A')]
            result=d.reduce_rows(rows,length)
            for motif in ('A','B'):
                window=result['windows'][f'final4/{motif}/fixed2']
                self.assertEqual(window['samples_per_window'],8)
                self.assertAlmostEqual(window['early_minus_late_drop'],(length-3)/length)
                self.assertAlmostEqual(window['late_minus_early'],-window['early_minus_late_drop'])
    def test_geometry_calibration_zero_and_rank_one(self):
        zero=d.covariance_summary(np.ones((12,4)))
        self.assertFalse(zero['normalized_dimension_defined'])
        self.assertIsNone(zero['participation_ratio'])
        cloud=np.arange(12)[:,None]*np.array([[1,2,3,4]])
        result=d.covariance_summary(cloud)
        self.assertAlmostEqual(result['participation_ratio'],1.)
        self.assertEqual(result['numerical_rank'],1)
        self.assertEqual(result['time_rank_ceiling'],4)
    def test_fixed_geometry_indices_disjoint(self):
        for length in d.DWELLS:
            early={1,2,3};late={length-3,length-2,length-1}
            self.assertFalse(early&late)
            rows=list(d.schedule(length,'B'));cycles=2048//(2*length)
            for motif in ('A','B'):
                for window in (early,late):
                    self.assertEqual(sum(m==motif and p in window and c>=cycles-4 for t,m,p,c in rows),12)

if __name__=='__main__':unittest.main()
