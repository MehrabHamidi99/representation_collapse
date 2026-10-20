"""Pure title/provenance metadata checks; no states, model or numerical run."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('state_plot',ROOT/'code/analysis/plot_figure10_state_diagnostics.py')
plot=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(plot)


def inputs():
    result={'identities':{'evaluation_result_sha256':'a'*64,'trajectory_sha256':'b'*64}}
    accepted={'trajectory_sha256':'b'*64,'accepted_evaluation_receipt_sha256':'d'*64,
        'accepted_evaluation_kind':'figure10_primitive_v2_selected_evaluation_execution_acceptance_v3',
        'checkpoint_format':'primitive_v2'}
    upstream={'kind':'figure10_primitive_v2_selected_evaluation_execution_acceptance_v3','status':'accepted',
        'checkpoint_format':'primitive_v2','executed_long_contract_sha256':plot.LONG_CONTRACT_SHA256,'model_seed':0,'selected_epoch':160,
        'terminal_replay_verified':True,'test_rows_verified':True,'selected_pointer_sha256':'c'*64,
        'evaluation_result_sha256':'a'*64,'trajectory_sha256':'b'*64}
    return result,accepted,upstream


class StatePlotIdentityTests(unittest.TestCase):
    def test_legacy_default_keeps_epoch7_and_no_new_manifest_fields(self):
        self.assertEqual(plot.evaluation_identity({},{}),(7,{}))
        with self.assertRaises(ValueError):plot.evaluation_identity({}, {},receipt_sha256='d'*64)

    def test_pinned_upstream_derives_epoch_and_genuine_pointer_metadata(self):
        result,accepted,upstream=inputs()
        epoch,identity=plot.evaluation_identity(result,accepted,upstream,'d'*64)
        self.assertEqual(epoch,160)
        self.assertEqual(identity,{'evaluation_acceptance_sha256':'d'*64,'selected_epoch':160,
            'checkpoint_format':'primitive_v2','selected_pointer_sha256':'c'*64,
            'executed_long_contract_sha256':plot.LONG_CONTRACT_SHA256})
        self.assertNotIn('final_head_sha256',identity)

    def test_v3_upstream_omission_cannot_silently_use_legacy_epoch7(self):
        result,accepted,upstream=inputs()
        with self.assertRaises(ValueError):plot.evaluation_identity(result,accepted)
        for key in ('accepted_evaluation_kind','checkpoint_format'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                plot.evaluation_identity(result,{key:accepted[key]})
        accepted['accepted_evaluation_receipt_sha256']='e'*64
        with self.assertRaises(ValueError):plot.evaluation_identity(result,accepted,upstream,'d'*64)
        accepted.pop('accepted_evaluation_receipt_sha256')
        with self.assertRaises(ValueError):plot.evaluation_identity(result,accepted,upstream,'d'*64)

    def test_cross_result_and_trajectory_pins_rejected(self):
        for location,key in (('result','evaluation_result_sha256'),('result','trajectory_sha256'),
                             ('accepted','trajectory_sha256'),('upstream','trajectory_sha256')):
            result,accepted,upstream=inputs()
            target={'result':result['identities'],'accepted':accepted,'upstream':upstream}[location]
            target[key]='f'*64
            with self.subTest(location=location,key=key),self.assertRaises(ValueError):
                plot.evaluation_identity(result,accepted,upstream,'d'*64)

    def test_unaccepted_wrong_seed_epoch_format_and_ambiguous_identity_rejected(self):
        result,accepted,upstream=inputs()
        for key,value in (('kind','figure10_selected_evaluation_execution_acceptance_v2'),('status','rejected'),
            ('checkpoint_format','v1'),('executed_long_contract_sha256','f'*64),('model_seed',True),('model_seed',1),('selected_epoch',0),('selected_epoch',201),
            ('selected_epoch',True),('terminal_replay_verified',False),('test_rows_verified',False),
            ('selected_pointer_sha256','bad'),('final_head_sha256','e'*64)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                plot.evaluation_identity(result,accepted,{**upstream,key:value},'d'*64)
        with self.assertRaises(ValueError):plot.evaluation_identity(result,accepted,upstream,'bad')
        for key in upstream:
            bad=copy.deepcopy(upstream);bad.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):
                plot.evaluation_identity(result,accepted,bad,'d'*64)

    def test_prospective_v3_axes_expand_upward_and_legacy_unchanged(self):
        self.assertEqual(plot.pr_limits([1.1,8.0]),(.9,2.1))
        self.assertEqual(plot.pr_limits([1.1,1.5],True),(.9,2.1))
        self.assertEqual(plot.pr_limits([1.1,8.0],True),(.9,8.8))
        for values in ([],[float('nan')],[.5],[True]):
            with self.subTest(values=values),self.assertRaises(ValueError):plot.pr_limits(values,True)

    def test_unpaired_cli_rejected_before_any_inputs_or_output(self):
        args=['plot','--analysis','missing','--analysis-sha256','a'*64,'--acceptance','missing',
            '--acceptance-sha256','b'*64,'--output','unused','--run-id','synthetic']
        for pair in (['--evaluation-acceptance','missing'],['--evaluation-acceptance-sha256','c'*64]):
            with self.subTest(pair=pair),patch('sys.argv',args+pair),patch.object(Path,'read_bytes') as read:
                with self.assertRaises(SystemExit):plot.main()
                read.assert_not_called()

if __name__=='__main__':unittest.main()
