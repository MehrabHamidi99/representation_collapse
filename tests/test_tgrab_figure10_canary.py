import contextlib
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse.tgrab_figure10_canary import (  # noqa: E402
    CanaryError, CanaryGuard, ENVIRONMENT_LOCK_SHA256, WHEEL_MANIFEST_SHA256,
    CELL_SHA256, REQUIRED_VERSIONS, MAX_UPDATES, ProcessOwner, PROTOCOL_ID,
    publish_terminal_manifest, read_pinned_bytes, run_prefix_updates,
    validate_environment_receipt,
    validate_node2a_receipt, reconcile_attempts, witnessed_native_update,
    validate_supervisor_runtime, OptimizerWitnessError,
)


class Graph:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.nodes = 100
        self.node_ids = tuple(range(100))


class State:
    def __init__(self, next_target_t):
        self.next_target_t = next_target_t


class SpyBackend:
    def __init__(self, fail_target=None):
        self.events = []
        self.fail_target = fail_target

    def reset(self):
        self.events.append(("reset", 0))
        return State(0)

    def training_mode(self):
        return contextlib.nullcontext()

    def detach(self, state):
        self.events.append(("detach", state.next_target_t))
        return state

    def consume(self, state, graph):
        self.events.append(("consume", graph.timestamp))
        if graph.timestamp != state.next_target_t:
            raise AssertionError("bad consume ordering")
        return State(graph.timestamp + 1)

    def loss_and_update(self, state, target, negatives):
        self.events.append(("update", target.timestamp, state.next_target_t))
        if state.next_target_t != target.timestamp:
            raise AssertionError("score state must precede the target")
        if target.timestamp == self.fail_target:
            raise ArithmeticError("synthetic update failure")
        return 1.0 / target.timestamp


class CanaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = [Graph(t) for t in range(257)]
        path = ROOT / "code/ops/tgrab_figure10_canary_supervisor.py"
        spec = importlib.util.spec_from_file_location("canary_supervisor_test", path)
        cls.supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.supervisor)
        if sys.platform.startswith("linux"):
            cls.supervisor.enable_child_subreaper()

    def test_module_import_is_stdlib_only_and_never_loads_model_data(self):
        tree = ast.parse((ROOT / "src/representation_collapse/tgrab_figure10_canary.py").read_text())
        top_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level_imports.append(node.module or "")
        forbidden = {"torch", "numpy", "representation_collapse.graph_data",
                     "representation_collapse.graph_reconstruction",
                     "representation_collapse.pinned_graph_import",
                     "representation_collapse.reporting"}
        self.assertFalse(forbidden.intersection(top_level_imports))
        source = (ROOT / "src/representation_collapse/tgrab_figure10_canary.py").read_text()
        main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        runtime_call = next(node for node in ast.walk(main_node)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "_runtime_train")
        reporter_import = next(node for node in ast.walk(main_node)
                               if isinstance(node, ast.ImportFrom) and node.module == "reporting")
        self.assertLess(reporter_import.lineno, runtime_call.lineno)
        runtime_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_runtime_train")
        calls = [node for node in ast.walk(runtime_node) if isinstance(node, ast.Call)]
        train_loads = [node for node in calls if isinstance(node.func, ast.Attribute)
                       and node.func.attr == "_load_split"]
        self.assertEqual(len(train_loads), 1)
        self.assertIsInstance(train_loads[0].args[1], ast.Constant)
        self.assertEqual(train_loads[0].args[1].value, "train")
        self.assertNotIn("open_test", source)

    def test_exact_256_chronological_predecessor_consumptions_and_updates(self):
        backend = SpyBackend()
        records = []
        calls = []
        result = run_prefix_updates(backend=backend, graphs=self.graphs,
            sampler=lambda target: calls.append(("sample", target.timestamp)) or ("neg",),
            guard=lambda: None, on_update=records.append)
        self.assertEqual(result["attempted_updates"], 256)
        self.assertEqual(result["completed_updates"], 256)
        self.assertEqual(result["first_target"], 1)
        self.assertEqual(result["last_target"], 256)
        self.assertFalse(result["validation_replayed"])
        self.assertFalse(result["selection_performed"])
        self.assertFalse(result["checkpoint_written"])
        self.assertFalse(result["test_payload_opened"])
        self.assertEqual(len(records), 256)
        self.assertTrue(all(record["completed"] for record in records))
        self.assertEqual(len(calls), 256)
        self.assertEqual([event[1] for event in backend.events if event[0] == "consume"], list(range(256)))
        self.assertEqual([event[1] for event in backend.events if event[0] == "update"], list(range(1, 257)))
        self.assertEqual(backend.events[0], ("reset", 0))
        self.assertEqual(backend.events[2], ("consume", 0))
        self.assertEqual(backend.events[3], ("update", 1, 1))

    def test_limit_and_short_or_nonchronological_prefix_fail_before_update(self):
        for kwargs in ({"limit": 255}, {"graphs": self.graphs[:256]}):
            backend = SpyBackend()
            args = {"backend": backend, "graphs": self.graphs, "sampler": lambda _: (),
                    "guard": lambda: None, "on_update": lambda _: None}
            args.update(kwargs)
            with self.assertRaises(CanaryError):
                run_prefix_updates(**args)
            self.assertFalse(any(event[0] == "update" for event in backend.events))
        bad = list(self.graphs)
        bad[100] = Graph(101)
        backend = SpyBackend()
        with self.assertRaises(CanaryError):
            run_prefix_updates(backend=backend, graphs=bad, sampler=lambda _: (),
                               guard=lambda: None, on_update=lambda _: None)
        self.assertFalse(any(event[0] == "update" for event in backend.events))
        self.assertEqual(MAX_UPDATES, 256)

    def test_stop_guard_prevents_any_model_update(self):
        backend = SpyBackend()
        checks = {"n": 0}
        def stop():
            checks["n"] += 1
            if checks["n"] == 3:
                raise CanaryError("STOP")
        with self.assertRaisesRegex(CanaryError, "STOP"):
            run_prefix_updates(backend=backend, graphs=self.graphs,
                sampler=lambda _: (), guard=stop, on_update=lambda _: None)
        self.assertFalse(any(event[0] == "update" for event in backend.events))

    def test_failed_update_is_recorded_and_is_not_retried(self):
        backend = SpyBackend(fail_target=3)
        records = []
        with self.assertRaises(ArithmeticError):
            run_prefix_updates(backend=backend, graphs=self.graphs,
                sampler=lambda target: (target.timestamp,), guard=lambda: None,
                on_update=records.append)
        self.assertEqual([record["attempt"] for record in records], [1, 2, 3])
        self.assertEqual(records[-1]["completed"], False)
        self.assertEqual(records[-1]["optimizer_mutation_status"], "unknown")
        self.assertEqual(len([event for event in backend.events if event[0] == "update"]), 3)

    def _journal(self, root):
        from representation_collapse.tgrab_figure10_attempt_journal import Figure10AttemptJournal
        from representation_collapse.tgrab_selection_transaction import SelectionContext
        from representation_collapse.tgrab_selected_checkpoint import SCHEMA_SHA256
        path = Path(root) / "attempts"
        path.mkdir()
        return Figure10AttemptJournal(path, SelectionContext("run", "prefix01", "a" * 64, SCHEMA_SHA256, None))

    def test_intent_fsync_failure_prevents_backend_step_and_is_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._journal(tmp) as journal:
                backend = SpyBackend()
                with mock.patch("representation_collapse.tgrab_figure10_attempt_journal.os.fsync", side_effect=OSError("disk")):
                    with self.assertRaises(Exception):
                        run_prefix_updates(backend=backend, graphs=self.graphs,
                            sampler=lambda _: (), guard=lambda: None,
                            on_attempt=journal.on_attempt, on_update=journal.on_update)
                self.assertFalse(any(event[0] == "update" for event in backend.events))
                counts = reconcile_attempts(journal.path)
                self.assertEqual(counts["attempted_updates"], 1)
                self.assertEqual(counts["optimizer_mutation_unknown_attempts"], 1)
                self.assertIsNone(counts["committed_updates"])

    def test_failed_outcome_fsync_cannot_be_upgraded_from_visible_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._journal(tmp) as journal:
                def outcome(record):
                    with mock.patch("representation_collapse.tgrab_figure10_attempt_journal.os.fsync", side_effect=OSError("disk")):
                        journal.on_update(record)
                with self.assertRaises(Exception):
                    run_prefix_updates(backend=SpyBackend(), graphs=self.graphs,
                        sampler=lambda _: (), guard=lambda: None,
                        on_attempt=journal.on_attempt, on_update=outcome)
                counts = reconcile_attempts(journal.path, terminal_journal_error=True)
                self.assertEqual(counts["attempted_updates"], 1)
                self.assertEqual(counts["optimizer_steps_confirmed"], 0)
                self.assertEqual(counts["optimizer_mutation_unknown_attempts"], 1)
                self.assertIsNone(counts["committed_updates"])

    def test_unresolved_intent_survives_missing_post_step_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._journal(tmp) as journal:
                def lost_record(_):
                    raise OSError("post-step storage unavailable")
                with self.assertRaises(OSError):
                    run_prefix_updates(backend=SpyBackend(), graphs=self.graphs,
                        sampler=lambda _: (), guard=lambda: None,
                        on_attempt=journal.on_attempt, on_update=lost_record)
                counts = reconcile_attempts(journal.path)
                self.assertEqual(counts["attempted_updates"], 1)
                self.assertEqual(counts["completed_updates"], 0)
                self.assertEqual(counts["optimizer_mutation_unknown_attempts"], 1)

    def test_bad_loss_after_return_is_a_completed_backend_call(self):
        for loss in (float("nan"), -1.0, "bad"):
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as tmp:
                with self._journal(tmp) as journal:
                    records = []
                    def record(value):
                        journal.on_update(value)
                        records.append(value)
                    with self.assertRaises(CanaryError):
                        run_prefix_updates(backend=SpyBackend(), graphs=self.graphs,
                            sampler=lambda _: (), guard=lambda: None,
                            on_attempt=journal.on_attempt, on_update=record,
                            update=lambda *_: loss)
                    counts = reconcile_attempts(journal.path)
                    self.assertEqual(counts["completed_updates"], 1)
                    self.assertEqual(counts["optimizer_steps_confirmed"], 1)
                    self.assertEqual(counts["optimizer_mutation_unknown_attempts"], 0)
                    self.assertEqual(records[0]["raw_loss_repr"], repr(loss))

    def test_native_step_witness_distinguishes_mutation_states_without_retry(self):
        class Optimizer:
            def __init__(self, fail=False):
                self.calls = 0
                self.fail = fail
            def step(self):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("inside step")
        class Backend:
            def __init__(self, mode):
                self.mode = mode
                self.optimizer = Optimizer(mode == "inside")
            def loss_and_update(self, *_):
                if self.mode == "before":
                    raise ValueError("before step")
                self.optimizer.step()
                if self.mode == "after":
                    raise ValueError("after step")
                return "bad loss"
        for mode, expected in (("before", "not_started"), ("inside", "unknown"), ("after", "completed")):
            backend = Backend(mode)
            with self.assertRaises(OptimizerWitnessError) as caught:
                witnessed_native_update(backend, None, None, None)
            self.assertEqual(caught.exception.step_status, expected)
            self.assertEqual(backend.optimizer.calls, 0 if mode == "before" else 1)
            self.assertNotIn("step", backend.optimizer.__dict__)
        self.assertEqual(witnessed_native_update(Backend("success"), None, None, None), "bad loss")

    def test_exception_outcomes_have_accurate_confirmed_step_counters(self):
        for status, expected in (("not_started", 0), ("unknown", 0), ("completed", 1)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                with self._journal(tmp) as journal:
                    def fail(*_):
                        raise OptimizerWitnessError("failure", step_status=status)
                    with self.assertRaises(OptimizerWitnessError):
                        run_prefix_updates(backend=SpyBackend(), graphs=self.graphs,
                            sampler=lambda _: (), guard=lambda: None,
                            on_attempt=journal.on_attempt, on_update=journal.on_update,
                            update=fail)
                    result = reconcile_attempts(journal.path)
                    self.assertEqual(result["attempted_updates"], 1)
                    self.assertEqual(result["completed_updates"], 0)
                    self.assertEqual(result["optimizer_steps_confirmed"], expected)
                    self.assertEqual(result["optimizer_mutation_unknown_attempts"], int(status == "unknown"))

    def test_runtime_receipt_cpus_on_node_and_relative_deadline(self):
        env = {"SLURM_JOB_ID": "123", "SLURM_JOB_ACCOUNT": "acct",
               "SLURM_JOB_PARTITION": "cpu", "SLURM_CPUS_ON_NODE": "1"}
        receipt = {"job": {"job_id": "123", "account": "acct", "partition": "cpu", "cpus_on_node": "1"},
                   "work_seconds": 540, "wall_started_epoch_ns": 2_000_000_000_000_000_000,
                   "deadline_unix": 2_000_000_540.0}
        validate_supervisor_runtime(receipt, runtime_seconds=540, deadline_unix=2_000_000_540.0, environ=env)
        bad = dict(env)
        bad.pop("SLURM_JOB_ACCOUNT")
        bad["SBATCH_ACCOUNT"] = "acct"
        bad["SBATCH_ACCOUNT"] = "not-the-allocation"
        validate_supervisor_runtime(receipt, runtime_seconds=540, deadline_unix=2_000_000_540.0, environ=bad)
        missing = dict(receipt, job=dict(receipt["job"], account=None))
        with self.assertRaisesRegex(CanaryError, "account"):
            validate_supervisor_runtime(missing, runtime_seconds=540, deadline_unix=2_000_000_540.0, environ=bad)
        with self.assertRaisesRegex(CanaryError, "deadline"):
            validate_supervisor_runtime(receipt, runtime_seconds=540, deadline_unix=1_000.0, environ=env)

    def test_dangling_stop_symlink_is_a_stop_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = ProcessOwner.acquire(Path(tmp) / "attempt")
            stop = Path(tmp) / "STOP"
            stop.symlink_to(Path(tmp) / "not-created")
            try:
                with self.assertRaisesRegex(CanaryError, "STOP"):
                    CanaryGuard(stop, 4_000_000_000, owner)()
            finally:
                owner.close()

    def test_environment_receipt_matches_preparation_script_schema_and_pins(self):
        receipt = {"state": "installed_metadata_verified_native_smoke_pending",
                   "requirements_sha256": ENVIRONMENT_LOCK_SHA256,
                   "artifact_manifest_sha256": WHEEL_MANIFEST_SHA256,
                   "installed_cell_sha256": CELL_SHA256,
                   "installed_metadata_verified": True,
                   "installed_versions": dict(REQUIRED_VERSIONS)}
        validate_environment_receipt(receipt)
        wrong = dict(receipt)
        wrong["state"] = "staged"
        with self.assertRaisesRegex(CanaryError, "completed isolated CPU install"):
            validate_environment_receipt(wrong)
        wrong = dict(receipt)
        wrong["installed_cell_sha256"] = "0" * 64
        with self.assertRaisesRegex(CanaryError, "installed_cell_sha256"):
            validate_environment_receipt(wrong)
        wrong = dict(receipt)
        wrong["installed_versions"] = dict(REQUIRED_VERSIONS, torch="2.1.0")
        with self.assertRaisesRegex(CanaryError, "distribution versions"):
            validate_environment_receipt(wrong)

    def test_node2a_must_be_a_passed_zero_training_synthetic_gate(self):
        receipt = {"schema": "node2a-synthetic-bootstrap-pass-v1", "status": "pass",
                   "both_model_components_roundtripped": True, "complete_hc_state_roundtripped": True,
                   "forward_loss_gradient_parity": True, "mode_dropout_parity": True,
                   "transaction_crash_reopen": True, "optimizer_updates": 0,
                   "primary_payload_opened": False, "validation_payload_opened": False,
                   "test_payload_opened": False}
        validate_node2a_receipt(receipt)
        for mutated, error in ((dict(receipt, status="pending"), "accepted node2a"),
                               (dict(receipt, optimizer_updates=1), "no-training/no-data boundary"),
                               (dict(receipt, test_payload_opened=True), "no-training/no-data boundary")):
            with self.assertRaisesRegex(CanaryError, error):
                validate_node2a_receipt(mutated)

    def test_supervisor_requires_hashed_node2a_artifact_before_child(self):
        receipt = {"schema": "node2a-synthetic-bootstrap-pass-v1", "status": "pass",
                   "both_model_components_roundtripped": True, "complete_hc_state_roundtripped": True,
                   "forward_loss_gradient_parity": True, "mode_dropout_parity": True,
                   "transaction_crash_reopen": True, "optimizer_updates": 0,
                   "primary_payload_opened": False, "validation_payload_opened": False,
                   "test_payload_opened": False}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "node2a.json"
            raw = json.dumps(receipt, sort_keys=True).encode()
            path.write_bytes(raw)
            spec = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            self.assertEqual(self.supervisor._verify_node2a(spec), spec)
            path.unlink()
            with self.assertRaisesRegex(SystemExit, "bounded, single-link regular"):
                self.supervisor._verify_node2a(spec)

    def test_input_pins_are_stable_regular_files_and_hash_checked(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            path = Path(tmp) / "pin.json"
            path.write_bytes(b"{}\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(read_pinned_bytes(path, digest, max_bytes=100), b"{}\n")
            with self.assertRaises(CanaryError):
                read_pinned_bytes(path, "0" * 64, max_bytes=100)
            link = Path(tmp) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(CanaryError):
                read_pinned_bytes(link, digest, max_bytes=100)

    def test_stop_deadline_and_process_ownership_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "attempt"
            owner = ProcessOwner.acquire(run_dir)
            owner.check()
            stop = Path(tmp) / "STOP"
            guard = CanaryGuard(stop, 4_000_000_000, owner)
            guard()
            stop.write_text("stop")
            with self.assertRaisesRegex(CanaryError, "STOP"):
                guard()
            with self.assertRaises(FileExistsError):
                ProcessOwner.acquire(run_dir)
            owner.close()

    def test_terminal_manifest_is_atomic_and_releases_owner_on_write_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "attempt"
            owner = ProcessOwner.acquire(run_dir)
            target = run_dir / "manifest.json"
            publish_terminal_manifest(owner, {"status": "failed", "updates": 3})
            self.assertEqual(json.loads(target.read_text()), {"status": "failed", "updates": 3})
            with self.assertRaises(OSError):
                owner.check()
            owner2 = ProcessOwner.acquire(Path(tmp) / "attempt-two")
            target2 = owner2.run_dir / "manifest.json"
            target2.write_text("occupied")
            with self.assertRaises(FileExistsError):
                publish_terminal_manifest(owner2, {"status": "overwritten"})
            self.assertFalse((owner2.run_dir / "manifest.json.pending").exists())
            with self.assertRaises(OSError):
                owner2.check()

    def test_supervisor_binds_scheduler_resources_and_runs_as_stdlib_outer_layer(self):
        path = ROOT / "code/ops/tgrab_figure10_canary_supervisor.py"
        supervisor = self.supervisor
        args = ["--run-id", "run-a", "--run-dir", "/scratch/run-a", "--stop-file", "/scratch/STOP",
                "--runtime-seconds", "540", "--environment-receipt", "/scratch/env.json",
                "--environment-receipt-sha256", "b" * 64,
                "--node2a-receipt", "/scratch/node2a.json", "--node2a-receipt-sha256", "d" * 64]
        actual = {"job_id": "123", "account": "mila", "partition": "main-cpu", "qos": "normal",
                  "nodes": 1, "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1",
                  "node_list": "cn001", "time_limit": "00:10:00", "gpu_allocation": "0",
                  "slurm_environment": {"SLURM_JOB_ID": "123"}}
        memory = {"effective_limit_bytes": 2_147_483_648, "current_path": "/memory.current",
                  "peak_path": None, "identity": {}}
        admission = {"schema": "tgrab-figure10-prefix-admission-v2", "protocol_id": PROTOCOL_ID,
                    "source_commit": "a" * 40,
                    "closure_sha256": {}, "run_id": "run-a", "canary_args": args,
                    "work_seconds": 540, "outer_wall_seconds": 570,
                    "interpreter": {"path": "/pinned/python", "sha256": "c" * 64},
                    "environment_receipt": {"path": "/scratch/env.json", "sha256": "b" * 64},
                    "node2a_receipt": {"path": "/scratch/node2a.json", "sha256": "d" * 64},
                    "job": {"job_id": None, "account": "mila", "partition": "main-cpu", "qos": "normal",
                            "nodes": 1, "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1",
                            "node_list": None, "time_limit": "00:10:00", "gpu_allocation": "0",
                            "memory_max_bytes": 2_147_483_648}}
        with mock.patch.dict(os.environ, {"RC_JOB_WALL_SECONDS": "570"}, clear=False):
            observed, _ = supervisor.validate_admission(admission, actual=actual, memory=memory)
            self.assertEqual(observed["job_id"], "123")
            bad = dict(admission, job=dict(admission["job"], account="wrong"))
            with self.assertRaisesRegex(SystemExit, "allocation"):
                supervisor.validate_admission(bad, actual=actual, memory=memory)
        slurm = {"SLURM_JOB_ID": "123", "SLURM_JOB_ACCOUNT": "acct", "SLURM_JOB_PARTITION": "cpu",
                 "SLURM_JOB_QOS": "normal", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
                 "SLURM_CPUS_ON_NODE": "1", "SLURM_CPUS_PER_TASK": "1",
                 "SLURM_JOB_NODELIST": "cn001", "SLURM_TIMELIMIT": "00:10:00", "SLURM_JOB_GPUS": ""}
        scheduler_record = ("JobId=123 JobState=RUNNING Account=acct Partition=cpu QOS=normal "
            "NumNodes=1 NumTasks=1 NumCPUs=1 CPUs/Task=1 NodeList=cn001 TimeLimit=00:10:00 "
            "ReqTRES=cpu=1,mem=2G,node=1 AllocTRES=cpu=1,mem=2G,node=1 Requeue=0 Restarts=0")
        with mock.patch.object(supervisor.subprocess, "run", return_value=mock.Mock(stdout=scheduler_record)):
            with mock.patch.dict(os.environ, slurm, clear=True):
                self.assertEqual(supervisor.scheduler_identity()["job_id"], "123")
            untrusted = dict(slurm)
            untrusted.pop("SLURM_JOB_ACCOUNT")
            untrusted["SBATCH_ACCOUNT"] = "wrong-request"
            with mock.patch.dict(os.environ, untrusted, clear=True):
                self.assertEqual(supervisor.scheduler_identity()["account"], "acct")
        tree = ast.parse(path.read_text())
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): imports.append(node.module or "")
        self.assertFalse({"torch", "numpy", "wandb"}.intersection(imports))
        launcher = (ROOT / "code/ops/tgrab_figure10_canary_job.sh").read_text()
        self.assertLess(launcher.index("RC_WALL_START_EPOCH_NS"), launcher.index("PINNED_PYTHON="))
        self.assertIn("timeout --signal=TERM --kill-after=10", launcher)
        self.assertIn("/environments/graph-native-cpu-20261015-calibration01/venv/bin/python", launcher)
        self.assertIn("exec(compile(raw, path, \"exec\"), ns, ns)", supervisor.BOOTSTRAP)
        self.assertIn("sys.dont_write_bytecode = True", supervisor.BOOTSTRAP)
        closure = supervisor.source_closure(ROOT)
        self.assertIn("code/ops/tgrab_figure10_canary_job.sh", closure)
        self.assertIn("src/representation_collapse/graph_reconstruction.py", closure)
        self.assertEqual(PROTOCOL_ID, supervisor.PROTOCOL_ID)

    def test_cgroup2_mount_mapping_effective_ancestor_and_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cgroup mount"
            leaf = root / "job" / "step"
            leaf.mkdir(parents=True)
            (leaf / "memory.max").write_text("4096")
            (leaf / "memory.current").write_text("1024")
            (leaf / "memory.peak").write_text("2048")
            (root / "job" / "memory.max").write_text("8192")
            (root / "memory.max").write_text("16384")
            cg, mi = Path(tmp) / "proc-cgroup", Path(tmp) / "mountinfo"
            cg.write_text("0::/tenant/job/step\n")
            escaped = str(root).replace(" ", "\\040")
            mi.write_text(f"31 22 0:33 /tenant {escaped} rw,nosuid - cgroup2 cgroup rw\n")
            data = self.supervisor.cgroup_memory(cg, mi)
            self.assertEqual(data["effective_limit_bytes"], 4096)
            self.assertEqual(data["peak_path"], str((leaf / "memory.peak").resolve()))
            self.assertEqual(len(data["limiting_ancestors"]), 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc and process-group controls required")
    def test_owned_process_group_normal_exit_and_deadline_kill_descendants(self):
        supervisor = self.supervisor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_info = (root / "descendant.pid").resolve()
            script = ("import subprocess,sys; "
                      "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                      f"open({str(child_info)!r},'w').write(str(p.pid))")
            import subprocess
            child = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
            events = []
            status, reason = supervisor.supervise_child(child, stop_path=root / "STOP",
                deadline_unix=time.time() + 10, memory=None, memory_limit=None,
                check_interval=0.02, cleanup_grace=0.15,
                journal=lambda event, **fields: events.append((event, fields)))
            self.assertEqual(status, 0)
            self.assertIsNone(reason)
            self.assertTrue(child_info.exists())
            self.assertEqual(supervisor.group_members(child.pid), [])
            self.assertIn("process_group_empty", [event for event, _ in events])

            kill_script = ("import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                           "p=subprocess.Popen([sys.executable,'-c',\"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)\"]); "
                           f"open({str(child_info)!r},'w').write(str(p.pid)); time.sleep(60)")
            child_info.unlink()
            child = subprocess.Popen([sys.executable, "-c", kill_script], start_new_session=True)
            end = time.time() + 2
            while not child_info.exists() and time.time() < end:
                time.sleep(0.01)
            events = []
            _status, reason = supervisor.supervise_child(child, stop_path=root / "STOP",
                deadline_unix=time.time() + 0.12, memory=None, memory_limit=None,
                check_interval=0.02, cleanup_grace=0.1,
                journal=lambda event, **fields: events.append((event, fields)))
            self.assertEqual(reason, "deadline")
            self.assertIn("process_group_kill", [event for event, _ in events])
            self.assertIn("process_group_empty", [event for event, _ in events])
            self.assertEqual(supervisor.group_members(child.pid), [])
            self.assertTrue(any(event == "resource_sample" for event, _ in events))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc and process-group controls required")
    def test_stop_terminates_and_reaps_process_group(self):
        import subprocess
        supervisor = self.supervisor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = root / "STOP"
            stop.symlink_to(root / "dangling-target")
            events = []
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            _status, reason = supervisor.supervise_child(child, stop_path=stop,
                deadline_unix=time.time() + 10, memory=None, memory_limit=None,
                check_interval=0.02, cleanup_grace=0.1,
                journal=lambda event, **fields: events.append(event))
            self.assertEqual(reason, "STOP")
            self.assertEqual(supervisor.group_members(child.pid), [])
            self.assertIn("process_group_empty", events)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc and process-group controls required")
    def test_signal_and_journal_error_paths_reap_owned_group(self):
        import subprocess
        supervisor = self.supervisor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            _status, reason = supervisor.supervise_child(child, stop_path=root / "STOP",
                deadline_unix=time.time() + 10, memory=None, memory_limit=None,
                signal_state={"number": 15}, check_interval=0.02, cleanup_grace=0.1)
            self.assertEqual(reason, "outer_signal_15")
            self.assertEqual(supervisor.group_members(child.pid), [])

            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            def broken_journal(*_args, **_kwargs):
                raise OSError("synthetic journal failure")
            with self.assertRaisesRegex(OSError, "journal failure"):
                supervisor.supervise_child(child, stop_path=root / "STOP",
                    deadline_unix=time.time() + 10, memory=None, memory_limit=None,
                    check_interval=0.02, cleanup_grace=0.1, journal=broken_journal)
            self.assertEqual(supervisor.group_members(child.pid), [])


if __name__ == "__main__":
    unittest.main()
