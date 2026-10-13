"""Explicit, mandatory W&B lifecycle for future admitted experiments.

No experiment is launched here. The caller enters the reporter before work.
Only caller-selected scalar metadata/metrics and hexadecimal identities enter
W&B; arbitrary files, model objects, environment mappings and arrays are not
accepted. Offline durability is distinct from independently verified delivery.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping
import uuid

SDK_VERSION = "0.30.0"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{0,95}\Z")
_PRIVATE_KEY = re.compile(r"(^|[_.\-/])(api.?key|token|secret|password|credential|endpoint|environment|env|raw|weights|dataset)([_.\-/]|$)", re.I)
# Version-pinned internal SDK settings are intentional: see wandb-reporting.md.
PRIVACY_SETTINGS = {
    "disable_code": True, "save_code": False, "disable_git": True,
    "disable_job_creation": True, "console": "off", "capture_loggers": {},
    "config_paths": (), "x_disable_meta": True, "x_disable_stats": True,
    "x_disable_machine_info": True, "x_save_requirements": False,
    "ignore_globs": ("*", "**/*"), "symlink": False,
    "sagemaker_disable": True, "launch": False,
    "host": "not-collected",
}


class ReportingError(RuntimeError):
    """Reporting failed; scientific work must not continue silently."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _scalars(values: Mapping[str, Any], *, hashes: bool = False) -> dict:
    if not isinstance(values, Mapping) or len(values) > 64:
        raise ValueError("reporting requires a compact explicit mapping (at most 64 entries)")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _KEY.fullmatch(key) or key.startswith("reporting/"):
            raise ValueError("invalid or reserved reporting key")
        if hashes:
            if not isinstance(value, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
                raise ValueError("identities must be lowercase SHA-1/SHA-256 hex digests")
        else:
            if _PRIVATE_KEY.search(key):
                raise ValueError("private/raw-data metadata key is prohibited; use a hash identity")
            if type(value) not in (str, int, float, bool, type(None)):
                raise ValueError("only explicit JSON scalars are accepted, never arrays/files/objects")
            if isinstance(value, str) and len(value) > 256:
                raise ValueError("reporting strings must be compact public metadata")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError("nonfinite reporting values require an explicit null/reason")
        result[key] = value
    if len(json.dumps(result, allow_nan=False).encode()) > 16384:
        raise ValueError("reporting payload exceeds 16 KiB")
    return result


class ExperimentReporter:
    """Enter before work; errors propagate and completion always has a receipt.

    ``output`` must be a new directory. Existing W&B IDs are never resumed;
    forks/resumes require a separate contract and a new lineage identity.
    Online mode requires an explicitly supplied authenticated entity. This
    class does not authenticate, inspect credentials or retry in another mode.
    """

    def __init__(self, output: str | Path, *, project: str, config: Mapping[str, Any],
                 identities: Mapping[str, str], entity: str | None = None,
                 mode: str = "offline", run_id: str | None = None):
        if mode not in ("offline", "online"):
            raise ValueError("mode must be explicit offline or online; disabled mode is prohibited")
        if not isinstance(project, str) or not _SAFE_NAME.fullmatch(project):
            raise ValueError("explicit valid project required")
        if entity is not None and (not isinstance(entity, str) or not _SAFE_NAME.fullmatch(entity)):
            raise ValueError("invalid entity")
        if mode == "online" and entity is None:
            raise ValueError("online mode requires the actual authenticated entity")
        self.run_id = run_id or uuid.uuid4().hex
        if not isinstance(self.run_id, str) or not _SAFE_NAME.fullmatch(self.run_id):
            raise ValueError("invalid run ID")
        self.config = _scalars(config)
        self.identities = _scalars(identities, hashes=True)
        if not self.identities:
            raise ValueError("at least one explicit source/config identity is required")
        self.output = Path(output).resolve()
        self.project, self.entity, self.mode = project, entity, mode
        self._run = None
        self._state = "new"
        self._last_step = -1
        self._clock = time.monotonic()
        self._receipt: dict = {}

    @property
    def receipt_path(self) -> Path:
        return self.output / "reporting-receipt.json"

    def _persist(self, **updates) -> None:
        self._receipt.update(updates)
        self._receipt.update(state=self._state, last_completed_step=self._last_step,
                             updated_at_utc=_now(), elapsed_seconds=time.monotonic() - self._clock)
        _atomic_json(self.receipt_path, self._receipt)

    def __enter__(self) -> "ExperimentReporter":
        if self._state != "new":
            raise ReportingError("a reporter instance may only be entered once")
        self.output.mkdir(parents=True, exist_ok=False)
        self._state = "initializing"
        self._receipt = {
            "schema_version": 1, "sdk_version_required": SDK_VERSION,
            "project": self.project, "entity_requested": self.entity, "mode": self.mode,
            "run_id": self.run_id, "started_at_utc": _now(), "work_may_start": False,
            "remote_delivery_verified": False, "remote_history_verified": False,
            "config": self.config, "identities": self.identities,
            "privacy_settings": PRIVACY_SETTINGS,
        }
        self._persist()
        try:
            # Reject automatic configuration sources rather than reading their values.
            if any(os.environ.get(k) for k in ("WANDB_SWEEP_ID", "WANDB_CONFIG_PATHS", "WANDB_LAUNCH",
                                               "WANDB_CONFIG", "WANDB_LAUNCH_CONFIG_PATH",
                                               "WANDB_SWEEP_PARAM_PATH")):
                raise ReportingError("ambient sweep/launch/config injection is prohibited")
            if Path("config-defaults.yaml").exists():
                raise ReportingError("automatic config-defaults.yaml injection is prohibited")
            sdk = importlib.import_module("wandb")
            self._sdk = sdk
            if sdk.__version__ != SDK_VERSION:
                raise ReportingError("installed W&B SDK does not match exact reporting pin")
            payload = {"experiment": self.config, "identities": self.identities}
            settings = sdk.Settings(**PRIVACY_SETTINGS, mode=self.mode,
                                    init_timeout=20, finish_timeout=8,
                                    finish_timeout_raises=True, quiet=True)
            self._run = sdk.init(project=self.project, entity=self.entity, id=self.run_id,
                                 dir=str(self.output), mode=self.mode, config=payload,
                                 settings=settings, save_code=False, reinit="create_new",
                                 resume="never", sync_tensorboard=False)
            if self._run is None or self._run.id != self.run_id:
                raise ReportingError("W&B did not create the requested run")
            if self._run.settings.mode != self.mode:
                raise ReportingError("effective reporting mode differs from explicit request")
            for key, value in PRIVACY_SETTINGS.items():
                actual = getattr(self._run.settings, key)
                if key in ("ignore_globs", "config_paths"):
                    actual = tuple(actual or ())
                if actual != value:
                    raise ReportingError("effective SDK privacy setting differs from required value")
            if dict(self._run.config) != payload:
                raise ReportingError("W&B configuration differs from explicit metadata")
            self._run.summary.update({"reporting/state": "running", "reporting/last_step": -1})
            self._state = "running"
            self._persist(work_may_start=True, sdk_version=sdk.__version__,
                          entity_actual=self._run.entity, wandb_files_dir=str(self._run.dir),
                          wandb_run_dir=str(Path(self._run.dir).parent),
                          run_url=self._run.url if self.mode == "online" else None)
            return self
        except BaseException as error:
            self._state = "initialization_failed"
            cleanup_error = None
            if self._run is not None:
                try:
                    self._run.finish(exit_code=1)
                except BaseException as cleanup:
                    cleanup_error = type(cleanup).__name__
            self._persist(work_may_start=False, error_type=type(error).__name__, cleanup_error_type=cleanup_error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ReportingError("W&B initialization failed; inspect the durable receipt") from None

    def _require_running(self) -> None:
        if self._state != "running":
            raise ReportingError("reporter must be successfully entered and still running")

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        self._require_running()
        if type(step) is not int or step < 0 or step <= self._last_step:
            raise ValueError("reporting steps must be nonnegative, strictly increasing integers")
        values = _scalars(metrics)
        if not values:
            raise ValueError("empty metric record")
        self._persist(last_attempted_step=step)
        try:
            self._run.log(values, step=step, commit=True)
            self._last_step = step
            self._persist()
        except BaseException as error:
            self._state = "reporting_failed"
            self._persist(work_may_start=False, error_type=type(error).__name__)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ReportingError("W&B metric logging failed; work must stop") from None

    def log_manifest(self, name: str, *, metadata: Mapping[str, Any],
                     identities: Mapping[str, str]) -> None:
        """Create one small JSON artifact ourselves; never accept a source path."""
        self._require_running()
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError("manifest name must be a simple name, never a path")
        payload = {"schema_version": 1, "metadata": _scalars(metadata),
                   "identities": _scalars(identities, hashes=True)}
        if len(json.dumps(payload).encode()) > 16384:
            raise ValueError("manifest exceeds 16 KiB")
        directory = self.output / "metadata-artifacts"
        directory.mkdir(exist_ok=True)
        path = directory / (name + ".json")
        if path.exists():
            raise ValueError("manifest names are immutable within a run")
        try:
            _atomic_json(path, payload)
            digest = file_sha256(path)
            artifact = self._sdk.Artifact(name, type="calibration-metadata",
                                          metadata={"manifest_sha256": digest})
            # Keep offline artifact bytes inside the durable run evidence tree.
            # This API never overwrites a manifest and sync must recheck its hash.
            artifact.add_file(str(path), name="manifest.json", policy="immutable", skip_cache=True)
            self._run.log_artifact(artifact)
            manifests = list(self._receipt.get("metadata_artifacts", []))
            manifests.append({"name": name, "path": str(path), "sha256": digest,
                              "remote_delivery_verified": False})
            self._persist(metadata_artifacts=manifests)
        except BaseException as error:
            self._state = "reporting_failed"
            self._persist(work_may_start=False, error_type=type(error).__name__)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ReportingError("metadata artifact logging failed; work must stop") from None

    def finish(self, *, failed: bool = False, error_type: str | None = None) -> None:
        if self._state not in ("running", "reporting_failed"):
            raise ReportingError("reporter is not active or was already finished")
        failed = failed or self._state == "reporting_failed"
        terminal = "failed" if failed else "completed"
        summary = {"reporting/state": terminal, "reporting/last_step": self._last_step,
                   "reporting/elapsed_seconds": time.monotonic() - self._clock}
        # Never persist exception text or traceback, which may contain credentials.
        if error_type is not None:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type):
                raise ValueError("error_type must be a class name, never an exception message")
            summary["reporting/error_type"] = error_type
        self._state = "finishing"
        self._persist(work_may_start=False, terminal_intent=terminal)
        failure = None
        try:
            self._run.summary.update(summary)
        except BaseException as error:
            failure = error
        try:
            self._run.finish(exit_code=1 if failed or failure is not None else 0)
        except BaseException as error:
            failure = error
        self._state = "finish_failed" if failure is not None else terminal
        self._persist(summary=summary, finish_returned=failure is None,
                      finish_error_type=type(failure).__name__ if failure is not None else None,
                      sync_required=self.mode == "offline")
        if failure is not None:
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            raise ReportingError("W&B finish failed; retain local evidence and stop") from None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._state in ("running", "reporting_failed"):
            try:
                self.finish(failed=exc_type is not None,
                            error_type=exc_type.__name__ if exc_type is not None else None)
            except BaseException:
                # Preserve the original experiment exception; receipt captures finish failure.
                if exc_type is None:
                    raise
        return False


def file_sha256(path: str | Path) -> str:
    """Hash a file without deserializing or uploading its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
