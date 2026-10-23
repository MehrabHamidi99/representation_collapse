import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('onebit_reduce',Path(__file__).parents[1]/'code/analysis/reduce_onebit_memory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(n=2,interventions=True):
    result=[]
    for i in range(4*n):
        bit=i%4//2
        logits={'native':[float(bit==0),float(bit==1)]}
        if interventions:
            logits.update(swap=list(reversed(logits['native'])),self_copy=logits['native'][:],same_cue_copy=logits['native'][:],zero=[0.,0.])
        predictions={k:int(v[1]>v[0]) for k,v in logits.items()}
        result.append({'condition':'toy','forcing_unit':i//4,'bit':bit,'arm':i%2,'logits':logits,'prediction':predictions,
                       'error':{k:int(v!=bit) for k,v in predictions.items()},'oracle_prediction':bit,'swap_donor_correct':int(interventions)})
    return result

class ReductionTests(unittest.TestCase):
    def test_gate_threshold(self):
        self.assertTrue(m.gate([487]*4));self.assertFalse(m.gate([512,512,486,512]))
        with self.assertRaises(ValueError):m.gate([True,512,512,512])
    def test_paired_complete_reduction(self):
        actual=m.reduce_rows(fixture(),2,'toy',True)
        self.assertEqual(actual['mean_accuracy'],1);self.assertEqual(actual['swap_donor_accuracy'],1)
        self.assertEqual(actual['paired_error_difference'],0)
        self.assertEqual(actual['intervention_accuracy_against_original']['zero'],.5)
    def test_positive_periodic_excess_error_direction(self):
        data=fixture()
        for row in data:
            if row['arm']==0: row['logits']['native']=[1.,0.]
        for i,row in enumerate(data):
            row['logits']['swap']=data[i+2 if row['bit']==0 else i-2]['logits']['native'][:]
            row['logits']['self_copy']=row['logits']['native'][:]
            row['logits']['same_cue_copy']=row['logits']['native'][:]
            row['prediction']={k:int(v[1]>v[0]) for k,v in row['logits'].items()}
            row['error']={k:int(v!=row['bit']) for k,v in row['prediction'].items()}
            row['swap_donor_correct']=int(row['prediction']['swap']==1-row['bit'])
        actual=m.reduce_rows(data,2,'toy',True)
        self.assertEqual(actual['paired_error_difference'],.5)
        self.assertEqual(actual['arm_accuracy'],{'0':.5,'1':1.})
    def test_incomplete_and_duplicate_rejected(self):
        data=fixture()
        with self.assertRaises(ValueError):m.reduce_rows(data[:-1],2,'toy',True)
        data[4]=copy.deepcopy(data[0])
        with self.assertRaises(ValueError):m.reduce_rows(data,2,'toy',True)
    def test_wrong_argmax_error_donor_and_sham_rejected(self):
        for field in ('prediction','error','swap_donor_correct','self_copy'):
            data=fixture()
            if field=='prediction':data[0][field]['native']=1
            elif field=='error':data[0][field]['native']=1
            elif field=='swap_donor_correct':data[0][field]=0
            else:data[0]['logits'][field]=[2.,0.]
            with self.assertRaises(ValueError):m.reduce_rows(data,2,'toy',True)
    def test_nan_and_bool_rejected(self):
        for value in (float('nan'),True):
            data=fixture();data[0]['logits']['native'][0]=value
            with self.assertRaises(ValueError):m.reduce_rows(data,2,'toy',True)
    def test_validation_no_intervention(self):
        actual=m.reduce_rows(fixture(interventions=False),2,'toy',False)
        self.assertEqual(actual['counts'],{'bit0_arm0':2,'bit0_arm1':2,'bit1_arm0':2,'bit1_arm1':2})
    def test_control_schedule_validation(self):
        p=m.MOTIFS[0]*2;c=m.MOTIFS[1]+m.MOTIFS[0]
        m.designs([[p,c]],1,2,1)
        with self.assertRaises(ValueError):m.designs([[p,p]],1,2,1)
        with self.assertRaises(ValueError):m.designs([[p,m.MOTIFS[0]+m.MOTIFS[1]]],1,2,1)
    def test_summary_comparison(self):
        expected=m.reduce_rows(fixture(),2,'toy',True)
        actual=copy.deepcopy(expected);actual.update(cue_body_units=4,constant_zero_readout_accuracy=.5,register_oracle_accuracy=1.)
        m.compare_summary(actual,expected,True)
        actual['accuracy']['bit1_arm0']=.5
        with self.assertRaises(ValueError):m.compare_summary(actual,expected,True)
    def test_complete_failed_gate_audit_and_invalid_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def save(name,value): (root/name).write_text(json.dumps(value))
            def lines(name,value): (root/name).write_text(''.join(json.dumps(x)+'\n' for x in value))
            data=fixture(512,False)
            for row in data:
                row['condition']='validation-R0';row['logits']['native']=[1.,0.]
                row['prediction']['native']=0;row['error']['native']=row['bit']
            summary=m.reduce_rows(data,512,'validation-R0',False)
            summary.update(cue_body_units=1024,constant_zero_readout_accuracy=.5,register_oracle_accuracy=1.)
            stage={'stage':1,'update':1000,'passed':False,'checks':[summary]}
            save('stage-1.json',stage);lines('stage-1-validation.jsonl',data)
            save('validation-design.json',{'0':[['','']]*512,'1':[[m.MOTIFS[0],m.MOTIFS[0]]]*512,'2':[[m.MOTIFS[0]*2,m.MOTIFS[1]+m.MOTIFS[0]]]*512})
            identity={'source_sha256':'a'*64,'protocol_sha256':'b'*64,'validation_sha256':m.sha(root/'validation-design.json'),'initial_model_sha256':'c'*64,
                      'seed':0,'training_rng_seed':1000,'validation_rng_seed':2000,'test_rng_seed':3000}
            save('identity.json',identity)
            save('pre-evaluation-checkpoint.json',{'checkpoint_sha256':'d'*64,'model_sha256':'e'*64,'update':1000,'eligible':False})
            history=[{'update':i,'stage':1,'R':0,'loss':1.,'gradient_norm':.1} for i in range(1,1001)]
            lines('training.jsonl',history)
            result={k:identity[k] for k in ('source_sha256','protocol_sha256','validation_sha256','initial_model_sha256')}
            result.update(schema='onebit-memory-discovery-v1',updates=1000,stage_results=[stage],status='competence_failed',heldout_opened=False,
                          evaluation=[],checkpoint_sha256='d'*64,final_model_sha256='e'*64)
            save('result.json',result)
            inventory={p.name:{'sha256':m.sha(p),'bytes':p.stat().st_size} for p in root.iterdir()}
            inventory['checkpoint.pt']={'sha256':'d'*64,'bytes':10};save('artifact-inventory.json',inventory)
            accepted=m.audit(root,m.sha(root/'result.json'))
            self.assertEqual(accepted['status'],'competence_failed');self.assertEqual(accepted['updates_verified'],1000)
            lines('evaluation.jsonl',[])
            with self.assertRaises(ValueError):m.audit(root,m.sha(root/'result.json'))

if __name__=='__main__':unittest.main()
