import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'code/analysis/reduce_repetition_dwell.py'
spec=importlib.util.spec_from_file_location('dwell_reduction',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(dwell=8,start='A'):
    rows=[]
    for t in range(2048):
        confusion={'tp':10,'fp':0,'fn':0,'tn':9890,'f1':1.}
        rows.append(dict(confusion,t=t,phase=t%dwell,cycle=t//(2*dwell),motif=start if (t//dwell)%2==0 else ('B' if start=='A' else 'A'),startup=t==0,transition=t>0 and t%dwell==0,persistence=None if t==0 else confusion.copy(),hidden_l2=0.,cell_l2=0.,period_absolute_l2=None if t<2*dwell else 0.,period_reference_l2=None if t<2*dwell else 0.,period_relative_l2=None))
    return rows

class ReductionTests(unittest.TestCase):
    def test_undefined_geometry(self):
        import math
        self.assertTrue(math.isnan(m.geometry_ratio(1.,None)))
        self.assertTrue(math.isnan(m.geometry_ratio(None,1.)))
        self.assertTrue(math.isnan(m.geometry_ratio(0.,1.)))
        self.assertEqual(m.geometry_ratio(2.,1.),.5)
    def test_all_dwell_window_counts(self):
        for d in m.DWELLS:
            for start in ('A','B'):
                r=m.reduce_rows(fixture(d,start),d,start)
                self.assertEqual(r['windows']['final4/A/fixed2']['samples_per_window'],8)
                self.assertEqual(r['windows']['final4/B/fixed2']['drop'],0)
                self.assertEqual(r['mean_f1'],1)
    def test_reject_chronology(self):
        rows=fixture();rows[6]['motif']='B'
        with self.assertRaisesRegex(ValueError,'chronology'):m.verify_rows(rows,8,'A')
    def test_reject_f1(self):
        rows=fixture();rows[9]['f1']=.5
        with self.assertRaisesRegex(ValueError,'F1'):m.verify_rows(rows,8,'A')
    def test_reject_boolean_count(self):
        rows=fixture();rows[0]['fp']=False
        with self.assertRaises(ValueError):m.verify_rows(rows,8,'A')
    def test_reject_baseline(self):
        rows=fixture();rows[1]['persistence'].update(tp=9,fn=1,fp=1,tn=9889,f1=.9)
        with self.assertRaisesRegex(ValueError,'persistence'):m.verify_rows(rows,8,'A')
    def test_reject_missing(self):
        with self.assertRaises(ValueError):m.verify_rows(fixture()[:-1],8,'A')
    def test_actual_window_direction(self):
        rows=fixture()
        for r in rows:
            if r['phase'] in (6,7):r.update(tp=5,fn=5,fp=0,tn=9890,f1=2/3)
        result=m.reduce_rows(rows,8,'A')
        self.assertAlmostEqual(result['windows']['final4/A/fixed2']['drop'],1/3)
        self.assertEqual(result['phase_confusion']['A']['fn'],[0,0,0,0,0,0,5,5])
        self.assertEqual(result['phase_confusion']['B']['fp'],[0]*8)

if __name__=='__main__':unittest.main()
