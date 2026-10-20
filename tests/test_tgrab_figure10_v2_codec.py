"""Synthetic primitive-v2 semantic tamper checks; no Torch or graph payloads."""
from __future__ import annotations
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from representation_collapse import tgrab_figure10_v2_codec as v2
from representation_collapse import tgrab_figure10_long_fit as writer
from tests.test_tgrab_figure10_selection_sink import _evidence

H = lambda s: hashlib.sha256(s.encode()).hexdigest()


def fixture():
    contract, catalog = v2._frozen()
    closure = {"src/representation_collapse/" + k: v for k, v in contract["input_pins"]["source_files_sha256"].items()}
    identities = {k: H(k) for k in v2.IDENTITY_KEYS}
    identities.update(executed_long_contract_sha256=v2.CONTRACT_SHA256, numerical_template_sha256=v2.TEMPLATE_SHA256,
        conversion_config_sha256=v2.TEMPLATE_SHA256, tensor_catalog_sha256=v2.enc.CATALOG_SHA256,
        raw_sha256=contract["data"]["raw_sha256"], prefix_reference_sha256=contract["prefix_parity"]["accepted_reference_sha256"],
        source_closure_sha256=writer.digest(writer.canonical(closure)))
    initialization = {key: {"relative_path": filename, "bytes": 100, "sha256": H(filename), "role": role, "encoding": "primitive_json"}
        for key, filename, role in (("seeding_record", "seeding.json", "seeding_before_models"),
            ("post_model_rng_record", "post-model-rng.json", "post_model_construction_rng"),
            ("negative_generator_record", "negative-generator.json", "negative_generator_initialization"))}
    initialization["initial_component_semantic_sha256"] = {k: H(k) for k in ("recurrent", "link_head")}
    expected = v2.V2ExpectedIdentity(run_id="synthetic-v2", source_commit="a"*40, slurm_job_id="123",
        identities=identities, source_closure=closure,
        runtime_environment={"requirements_lock_sha256": contract["input_pins"]["environment_lock_sha256"],
            "wheel_manifest_sha256": contract["input_pins"]["wheel_manifest_sha256"]}, initialization=initialization,
        module_modes={"relative_path": "module-modes.json", "bytes": 100, "sha256": H("modes")},
        module_state_dict_versions={k: {"": 1} for k in ("recurrent", "link_head")},
        optimizer_parameter_order=[f"{k}.{name}" for k in ("recurrent", "link_head") for name in catalog["components"][k]["serialization_order"]],
        actual_job={"job_id": "123", "account": "mila", "partition": "main-cpu", "qos": "normal", "nodes": "1", "tasks": "1",
            "cpus_on_node": "1", "cpus_per_task": "1", "gpu_allocation": "0", "memory_max_bytes": "2147483648", "time_limit": "2-00:00:00"})
    p = {k: val for k, val in asdict(expected).items() if k != "source_closure"}
    p.update(kind="tgrab_long_fit_candidate_provenance_v2", schema_version=2, attempt_id="longfit01",
        executed_long_contract_sha256=v2.CONTRACT_SHA256, numerical_template_sha256=v2.TEMPLATE_SHA256,
        conversion_config_sha256=v2.TEMPLATE_SHA256, model_seed=0, negative_seed=1, seed_before_construction=True,
        construction_order=["seed_all", "recurrent", "link_head", "optimizer", "negative_generator"],
        runtime_policy={"device": "cpu", "precision": "float32", "cpu_threads": 1, "amp": False, "deterministic_algorithms": True,
            "deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True, "work_seconds": 172680, "slurm_seconds": 172800},
        optimizer_state_included=False, rng_restore_allowed=False, test_payload_opened=False, complete_epoch=1,
        complete_epoch_counters={"complete_epochs": 1, "validation_epochs": 1, "attempted_updates": 2559,
            "completed_updates": 2559, "optimizer_steps_confirmed": 2559},
        journal_epoch_seal={"epoch": 1, "records": 5118, "kind": "figure10_longfit_epoch_chunk_seal_v2",
            "run_id": "synthetic-v2", "attempt_id": "longfit01", "contract_sha256": v2.CONTRACT_SHA256,
            "admission_sha256": identities["actual_supervisor_receipt_sha256"], "filename": "epoch-001.jsonl", "receipt_filename": "epoch-001.seal.json", "attempted_updates": 2559, "completed_updates": 2559,
            "optimizer_steps_confirmed": 2559, "chunk_sha256": H("chunk"), "last_record_sha256": H("record"), "receipt_sha256": H("receipt")},
        created_at_utc="2026-10-20T00:00:00Z")
    return expected, p


def repin(root, name, obj):
    (root/name).write_bytes(v2.enc._canonical(obj))
    m=json.loads((root/'manifest.json').read_bytes())
    if name != 'manifest.json':
        raw=(root/name).read_bytes(); m['files'][name]={'bytes':len(raw),'sha256':v2.enc._sha(raw)}
        (root/'manifest.json').write_bytes(v2.enc._canonical(m))
    return v2.enc._sha((root/'manifest.json').read_bytes())


class V2CodecTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.expected,p=fixture()
        evidence,_=_evidence(1,0.0)
        ref=writer.write_candidate_v2(Path(self.tmp.name).resolve(), evidence, p)
        self.root=Path(self.tmp.name).resolve()/'epoch-001'; self.hash=ref['manifest_sha256']

    def read(self, sha=None):
        return v2.read_candidate_v2(self.root,sha or self.hash,expected=self.expected)

    def test_original_v2_payloads_reusable_without_v1_manifest(self):
        result=self.read()
        self.assertEqual(result['manifest']['kind'],'tgrab_complete_epoch_candidate_v2')
        self.assertEqual(result['terminal_state'],(self.root/'validation-terminal.f32le').read_bytes())
        self.assertEqual(set(result['files_sha256']),set(v2.enc.FILES))
        self.assertEqual(result['recurrent'],(self.root/'recurrent.f32le').read_bytes())

    def test_source_closure_uses_actual_writer_without_newline(self):
        no_newline=writer.digest(writer.canonical(dict(self.expected.source_closure)))
        self.assertEqual(self.expected.identities['source_closure_sha256'],no_newline)
        p=json.loads((self.root/'provenance.json').read_bytes())
        wrong=v2.enc._sha(v2.enc._canonical(dict(self.expected.source_closure)))
        self.assertNotEqual(wrong,no_newline)
        p['identities']['source_closure_sha256']=wrong
        self.expected=replace(self.expected,identities=copy.deepcopy(p['identities']))
        with self.assertRaises(v2.enc.CheckpointError):self.read(repin(self.root,'provenance.json',p))

    def test_all_identity_policy_tampering_rejected_even_rehashed(self):
        original=(self.root/'provenance.json').read_bytes()
        for key in v2.IDENTITY_KEYS:
            with self.subTest(key=key):
                p=json.loads(original); p['identities'][key]=H('tampered')
                sha=repin(self.root,'provenance.json',p)
                with self.assertRaises(v2.enc.CheckpointError):self.read(sha)

    def test_shape_offset_catalog_dtype_and_node_axis_tampering(self):
        original=(self.root/'manifest.json').read_bytes()
        for key,value in (('shape',[1,128,100]),('offset',4),('dtype','float64_le'),('order','F')):
            with self.subTest(key=key):
                m=json.loads(original);m['terminal_state']['tensors'][0][key]=value
                sha=repin(self.root,'manifest.json',m)
                with self.assertRaises(v2.enc.CheckpointError):self.read(sha)
        m=json.loads(original); m['components']['recurrent']['tensors'][0]['name']='wrong'
        with self.assertRaises(v2.enc.CheckpointError): self.read(repin(self.root,'manifest.json',m))

    def test_nonfinite_component_rejected_with_consistent_hashes(self):
        raw=bytearray((self.root/'link-head.f32le').read_bytes());raw[:4]=b'\x00\x00\xc0\x7f'
        (self.root/'link-head.f32le').write_bytes(raw)
        m=json.loads((self.root/'manifest.json').read_bytes());d=m['components']['link_head'];d['sha256']=v2.enc._sha(raw)
        m['files']['link-head.f32le']['sha256']=d['sha256']
        t=d['tensors'][0];chunk=bytes(raw[:t['nbytes']]);t['data_sha256']=v2.enc._sha(chunk)
        t['semantic_sha256']=H('nonfinite-has-no-valid-semantic-hash')
        with self.assertRaises(v2.enc.CheckpointError):self.read(repin(self.root,'manifest.json',m))

    def test_validation_chronology_negative_counts_and_score_tampering(self):
        original=(self.root/'validation.json').read_bytes()
        for key,value in (('target_timestamp',2561),('tp',-1),('f1',1.0),('change_point',False)):
            with self.subTest(key=key):
                v=json.loads(original);v['rows'][0][key]=value
                with self.assertRaises(v2.enc.CheckpointError):self.read(repin(self.root,'validation.json',v))

    def test_external_source_environment_seed_modes_job_counters(self):
        original=(self.root/'provenance.json').read_bytes()
        for key,value in (('source_commit','b'*40),('initialization',{}),('module_modes',{}),('actual_job',{}),
                          ('runtime_environment',{}),('complete_epoch_counters',{}),('complete_epoch',2),('runtime_policy',{})):
            with self.subTest(key=key):
                p=json.loads(original);p[key]=value
                with self.assertRaises(v2.enc.CheckpointError):self.read(repin(self.root,'provenance.json',p))

    def test_inventory_symlink_hardlink_and_manifest_pin(self):
        with self.assertRaises(v2.enc.CheckpointError):self.read(H('notmanifest'))
        (self.root/'extra').write_text('x')
        with self.assertRaises(v2.enc.CheckpointError):self.read()
        (self.root/'extra').unlink()
        target=self.root/'validation.json'; raw=target.read_bytes(); target.unlink(); target.symlink_to('provenance.json')
        with self.assertRaises(v2.enc.CheckpointError):self.read()
        target.unlink();target.write_bytes(raw)
        import os
        os.link(target,self.root.parent/'hardlink')
        with self.assertRaises(v2.enc.CheckpointError):self.read()

    def test_duplicate_keys_and_frozen_catalog_pin(self):
        original=(self.root/'manifest.json').read_bytes()
        raw=b'{"epoch":1,'+original[1:]
        (self.root/'manifest.json').write_bytes(raw)
        with self.assertRaises(v2.enc.CheckpointError):self.read(v2.enc._sha(raw))
        (self.root/'manifest.json').write_bytes(original)
        with patch.object(v2.enc,'CATALOG_SHA256',H('wrongcatalog')):
            with self.assertRaises(v2.enc.CheckpointError):self.read()

if __name__=='__main__': unittest.main()
