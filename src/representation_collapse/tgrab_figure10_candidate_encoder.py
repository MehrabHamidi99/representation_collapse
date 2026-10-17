"""Pure data encoder for one complete Figure 10 epoch candidate.

This module converts detached evidence plus caller-supplied identity records to
the frozen six-file selected-checkpoint v1 representation. It validates hashes
and schema, but cannot authenticate the caller's provenance or admission.
"""
from __future__ import annotations

import hashlib
import math
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from . import tgrab_selected_checkpoint as codec

if TYPE_CHECKING:
    from .tgrab_figure10_native_fit import Figure10EpochEvidence


class CandidateEncodingError(codec.CheckpointError):
    """Epoch evidence or external identity mappings do not satisfy v1."""


_IDENTITY_KEYS = frozenset((
    "executed_config_sha256", "schema_sha256", "codec_source_sha256",
    "trainer_source_sha256", "environment_receipt_sha256",
    "converted_manifest_sha256", "train_file_sha256", "validation_file_sha256",
    "test_file_sha256", "train_semantic_sha256", "validation_semantic_sha256",
    "test_semantic_sha256", "node_order_semantic_sha256",
    "feature_encoding_semantic_sha256", "weight_encoding_semantic_sha256",
    "conversion_receipt_sha256", "admission_sha256", "reporting_identity_sha256",
    "tensor_catalog_sha256", "raw_sha256", "reconstruction_config_sha256",
    "converter_sha256", "conversion_protocol_sha256",
))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_copy(v) for v in value]
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


def _cpu_f32_bytes(tensor: Any, *, name: str, shape: tuple[int, ...]) -> bytes:
    if (tuple(int(d) for d in getattr(tensor, "shape", ())) != shape
            or str(getattr(tensor, "dtype", "")) != "torch.float32"
            or str(getattr(tensor, "device", "")) != "cpu"):
        raise CandidateEncodingError(f"{name}: expected CPU FP32 tensor with frozen shape")
    numpy_fn = getattr(tensor, "numpy", None)
    if not callable(numpy_fn):
        raise CandidateEncodingError(f"{name}: detached evidence must expose NumPy-compatible bytes")
    try:
        array = numpy_fn()
        if tuple(array.shape) != shape or str(array.dtype) != "float32":
            raise CandidateEncodingError(f"{name}: NumPy view differs from frozen FP32 tensor")
        if not bool(array.flags.c_contiguous):
            array = array.copy(order="C")
        if sys.byteorder != "little":
            array = array.astype("<f4", copy=True)
        payload = array.tobytes(order="C")
    except CandidateEncodingError:
        raise
    except BaseException as exc:
        raise CandidateEncodingError(f"{name}: tensor could not be read as data bytes") from exc
    if len(payload) != math.prod(shape) * 4:
        raise CandidateEncodingError(f"{name}: FP32 byte length differs from frozen shape")
    try:
        codec._finite_f32(payload)
        return payload
    except codec.CheckpointError as exc:
        raise CandidateEncodingError(f"{name}: invalid finite FP32 payload") from exc


def _component_bytes(component: str, state_dict: Mapping[str, Any], catalog: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    spec = catalog["components"][component]
    ordered = spec["serialization_order"]
    descriptors = {entry["name"]: entry for entry in spec["tensors"]}
    if (not isinstance(state_dict, Mapping) or set(state_dict) != set(ordered)
            or tuple(state_dict) != tuple(spec["native_state_dict_order"])):
        raise CandidateEncodingError(f"{component}: tensor names differ from pinned catalog")
    parts: list[bytes] = []
    tensor_rows: list[dict[str, Any]] = []
    offset = 0
    for name in ordered:
        entry = descriptors[name]
        shape = tuple(entry["shape"])
        raw = _cpu_f32_bytes(state_dict[name], name=f"{component}.{name}", shape=shape)
        if len(raw) != entry["nbytes"] or offset != entry["payload_offset"]:
            raise CandidateEncodingError(f"{component}.{name}: catalog byte offset/length mismatch")
        tensor_rows.append({
            "name": name, "dtype": "float32_le", "shape": list(shape), "order": "C",
            "offset": offset, "nbytes": len(raw), "data_sha256": _sha(raw),
            "semantic_sha256": codec.tensor_semantic_sha256(component, name, list(shape), raw),
        })
        parts.append(raw)
        offset += len(raw)
    payload = b"".join(parts)
    if len(payload) != spec["nbytes"]:
        raise CandidateEncodingError(f"{component}: concatenated payload length differs from catalog")
    return payload, {"filename": spec["payload_filename"], "bytes": len(payload),
                     "sha256": _sha(payload), "tensors": tensor_rows}


def _terminal_bytes(evidence: Any, epoch: int) -> tuple[bytes, dict[str, Any]]:
    terminal = evidence.terminal_state
    if (type(getattr(terminal, "next_target_t", None)) is not int
            or terminal.next_target_t != 2816
            or tuple(getattr(terminal, "node_ids", ())) != tuple(range(100))):
        raise CandidateEncodingError("terminal state is not canonical post-consume s[2816]")
    chunks = []
    tensor_rows = []
    for name, offset in (("hidden", 0), ("cell", 51200)):
        raw = _cpu_f32_bytes(getattr(terminal, name, None), name=f"terminal.{name}", shape=(1, 100, 128))
        if len(raw) != 51200:
            raise CandidateEncodingError(f"terminal.{name}: frozen tensor byte count mismatch")
        chunks.append(raw)
        tensor_rows.append({"name": name, "dtype": "float32_le", "shape": [1, 100, 128],
                            "order": "C", "offset": offset, "nbytes": len(raw),
                            "data_sha256": _sha(raw),
                            "semantic_sha256": codec.tensor_semantic_sha256(
                                "terminal_state", name, [1, 100, 128], raw)})
    payload = b"".join(chunks)
    return payload, {"filename": "validation-terminal.f32le", "bytes": len(payload),
                     "sha256": _sha(payload), "tensors": tensor_rows,
                     "next_target_t": 2816, "last_consumed_t": 2815,
                     "construction": "fresh_zero_eval_replay_0_through_2815", "epoch": epoch}


def _validation_rows(evidence: Any, epoch: int) -> tuple[dict[str, Any], float]:
    source = evidence.validation_rows
    if not isinstance(source, (tuple, list)) or len(source) != 256:
        raise CandidateEncodingError("complete candidate requires 256 validation rows")
    rows = []
    for i, row in enumerate(source):
        target = 2560 + i
        if not isinstance(row, Mapping) or row.get("epoch") != epoch:
            raise CandidateEncodingError("validation row epoch/order mismatch")
        if row.get("target_timestamp") != target:
            raise CandidateEncodingError("validation target timestamps are not exactly ordered")
        keys = {"epoch", "target_timestamp", "source_timestamp", "target_phase", "source_phase",
                "phase_transition_expected", "change_point", "pairs", "tp", "fp", "fn", "tn",
                "precision", "recall", "f1", "accuracy"}
        if set(row) != keys:
            raise CandidateEncodingError("validation row fields differ from frozen schema")
        rows.append({k: row[k] for k in keys if k != "epoch"})
    mean = math.fsum(row["f1"] for row in rows) / 256
    validation = {"kind": "tgrab_complete_validation_rows_v1", "schema_version": 1,
                  "split": "validation", "epoch": epoch, "rows": rows,
                  "time_mean_f1": mean}
    try:
        codec._validate_validation_score({"epoch": epoch, "validation": {"time_mean_f1": mean}}, validation)
    except codec.CheckpointError as exc:
        raise CandidateEncodingError("validation rows or recomputed mean violate frozen score rules") from exc
    if evidence.record.validation_mean_f1 != mean:
        raise CandidateEncodingError("epoch record mean differs from recomputed validation F1")
    return validation, mean


def encode_epoch_candidate(evidence: "Figure10EpochEvidence", *, provenance: Mapping[str, Any],
                           manifest: Mapping[str, Any]) -> Mapping[str, bytes]:
    """Return canonical six-file v1 bytes for a complete epoch.

    ``provenance`` must be the full caller-supplied provenance record, including
    exact epoch counters. ``manifest`` must be ``{"identities": {...}}`` with
    every frozen external identity pin. Neither mapping is authenticated here.
    """
    try:
        epoch = evidence.record.epoch
        if type(epoch) is not int or not 1 <= epoch <= 200:
            raise CandidateEncodingError("epoch is outside the frozen complete-epoch range")
        record = evidence.record
        if (record.attempted_updates != 2559 or record.completed_updates != 2559
                or record.optimizer_steps_confirmed != 2559):
            raise CandidateEncodingError("epoch is not exactly 2559 attempted/completed optimizer steps")
        if set(manifest) != {"identities"} or not isinstance(manifest["identities"], Mapping):
            raise CandidateEncodingError("manifest input must contain exactly the full identities mapping")
        identities = _copy(manifest["identities"])
        if set(identities) != _IDENTITY_KEYS:
            raise CandidateEncodingError("manifest identity mapping is incomplete or has unknown pins")
        prov = _copy(provenance)
        if not isinstance(prov, dict):
            raise CandidateEncodingError("provenance must be a complete mapping")
        counters = prov.get("counters")
        if not isinstance(counters, dict):
            raise CandidateEncodingError("provenance counters are required")
        expected_counters = {"complete_epochs": epoch, "validation_epochs": epoch,
                             "attempted_updates": epoch * 2559, "completed_updates": epoch * 2559,
                             "selected_load_attempts": 0, "test_payload_read_attempts": 0}
        if counters != expected_counters:
            raise CandidateEncodingError("provenance counters differ from exact complete-epoch policy")
        required_cross = {"admission_sha256": prov.get("admission_sha256"),
                          "executed_config_sha256": prov.get("executed_config_sha256"),
                          "reconstruction_config_sha256": prov.get("reconstruction_config_sha256"),
                          "reporting_identity_sha256": prov.get("reporting_identity_sha256")}
        for key, value in required_cross.items():
            if identities.get(key) != value:
                raise CandidateEncodingError(f"manifest/provenance identity differs: {key}")
        source_pins = prov.get("source_sha256", {})
        environment = prov.get("environment", {})
        for identity_key, external_value in (
                ("schema_sha256", source_pins.get("schema")),
                ("codec_source_sha256", source_pins.get("codec")),
                ("trainer_source_sha256", source_pins.get("trainer")),
                ("environment_receipt_sha256", environment.get("environment_receipt_sha256"))):
            if identities.get(identity_key) != external_value:
                raise CandidateEncodingError(f"manifest/provenance source pin differs: {identity_key}")
        if identities.get("schema_sha256") != codec.SCHEMA_SHA256 or identities.get("tensor_catalog_sha256") != codec.CATALOG_SHA256:
            raise CandidateEncodingError("manifest does not pin the frozen schema and catalog")

        _, catalog = codec._schema_and_catalog()
        recurrent, recurrent_desc = _component_bytes("recurrent", evidence.model_state_dicts["node_emb"], catalog)
        link_head, head_desc = _component_bytes("link_head", evidence.model_state_dicts["link_pred"], catalog)
        terminal_bytes, terminal_desc = _terminal_bytes(evidence, epoch)
        validation, mean = _validation_rows(evidence, epoch)
        validation_bytes = codec._canonical(validation)
        provenance_bytes = codec._canonical(prov)
        if len(provenance_bytes) > 65536:
            raise CandidateEncodingError("provenance exceeds frozen byte maximum")

        schema, _ = codec._schema_and_catalog()
        md = schema["$defs"]["manifest"]
        out_manifest = {
            "kind": md["properties"]["kind"]["const"],
            "schema_version": md["properties"]["schema_version"]["const"],
            "artifact_role": md["properties"]["artifact_role"]["const"],
            "epoch": epoch,
            "architecture": _copy(md["properties"]["architecture"]["const"]),
            "semantics": _copy(md["properties"]["semantics"]["const"]),
            "identities": identities,
            "components": {"recurrent": recurrent_desc, "link_head": head_desc},
            "terminal_state": {**terminal_desc,
                               "recurrent_sha256": recurrent_desc["sha256"],
                               "link_head_sha256": head_desc["sha256"]},
            "validation": {"filename": "validation.json", "bytes": len(validation_bytes),
                           "sha256": _sha(validation_bytes), "split": "validation",
                           "interval": [2560, 2816], "row_count": 256,
                           "time_mean_f1": mean},
            "provenance": {"filename": "provenance.json", "bytes": len(provenance_bytes),
                           "sha256": _sha(provenance_bytes)},
        }
        # Validate before returning so malformed identity/provenance mappings
        # cannot produce bytes that only fail after publication.
        codec.validate_metadata(codec._canonical(validation), expected_kind="tgrab_complete_validation_rows_v1")
        codec.validate_metadata(provenance_bytes, expected_kind="tgrab_selected_inference_provenance_v1",
                                max_bytes=65536)
        manifest_bytes = codec._canonical(out_manifest)
        codec.validate_metadata(manifest_bytes, expected_kind="tgrab_selected_inference_v1")
        codec._validate_cross_identities(out_manifest, validation, prov)
        files = {"recurrent.f32le": recurrent, "link-head.f32le": link_head,
                 "validation-terminal.f32le": terminal_bytes, "validation.json": validation_bytes,
                 "provenance.json": provenance_bytes, "manifest.json": manifest_bytes}
        if set(files) != set(codec.FILES):
            raise CandidateEncodingError("encoder did not construct the exact frozen six files")
        return MappingProxyType(files)
    except CandidateEncodingError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise CandidateEncodingError("epoch evidence or required identity pin is malformed") from exc
