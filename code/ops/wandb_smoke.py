"""Small reporting lifecycle smoke; no model, dataset, training or cluster work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import zlib

from representation_collapse.reporting import ExperimentReporter, file_sha256


def verify_smoke_stream(path: Path, receipt: dict) -> dict:
    """CRC-checked single-block fixture reader for W&B transaction format v0.

    Scope is this tiny smoke, not arbitrary production logs. The official
    v0.30.0 transactionlog reader specifies IEEE CRC32 and protobuf Records.
    """
    from wandb.proto.wandb_internal_pb2 import Record
    data = path.read_bytes()
    if len(data) >= 32768 or data[:7] != struct.pack('<4sHB', b':W&B', 0xBEE1, 0):
        raise ValueError('unsupported smoke stream header/size')
    offset = 7
    records = []
    while offset < len(data):
        if len(data) - offset < 7:
            raise ValueError('truncated record header')
        checksum, size, kind = struct.unpack('<IHB', data[offset:offset+7])
        offset += 7
        body = data[offset:offset+size]
        offset += size
        if kind != 1 or len(body) != size or zlib.crc32(bytes([kind]) + body) != checksum:
            raise ValueError('unsupported, truncated or corrupt smoke record')
        record = Record()
        record.ParseFromString(body)
        records.append(record)
    kinds = [r.WhichOneof('record_type') for r in records]
    if any(k in kinds for k in ('environment', 'files', 'stats', 'output', 'output_raw')):
        raise ValueError('automatic environment/files/stats/console records found')
    def items(entries):
        return {(e.key if e.key else '/'.join(e.nested_key)): json.loads(e.value_json) for e in entries}
    runs = [r.run for r in records if r.WhichOneof('record_type') == 'run']
    if len(runs) != 1 or runs[0].host != 'not-collected':
        raise ValueError('unexpected run count or automatic hostname capture')
    config = items(runs[0].config.update)
    if config != {'_wandb': {}, 'experiment': receipt['config'], 'identities': receipt['identities']}:
        raise ValueError('unexpected automatic configuration')
    history = [items(r.history.item) for r in records if r.WhichOneof('record_type') == 'history']
    if len(history) != 2 or [(h['_step'], h['smoke/sequence'], h['smoke/check_passed']) for h in history] != [(0,0,True),(1,1,True)]:
        raise ValueError('two expected committed history records are missing')
    summary = {}
    for r in records:
        if r.WhichOneof('record_type') == 'summary':
            summary.update(items(r.summary.update))
    if summary.get('reporting/state') != 'completed' or summary.get('reporting/last_step') != 1:
        raise ValueError('completed summary missing')
    if 'exit' not in kinds or 'artifact' not in kinds:
        raise ValueError('exit or metadata artifact record missing')
    artifacts = [r.artifact for r in records if r.WhichOneof('record_type') == 'artifact']
    if len(artifacts) != 1 or artifacts[0].type != 'calibration-metadata':
        raise ValueError('unexpected artifact count/type')
    contents = artifacts[0].manifest.contents
    expected = receipt['metadata_artifacts'][0]
    if len(contents) != 1 or contents[0].path != 'manifest.json' or contents[0].local_path != expected['path']:
        raise ValueError('artifact bytes must remain within the durable receipt tree')
    if file_sha256(contents[0].local_path) != expected['sha256']:
        raise ValueError('metadata artifact content differs from its receipt')
    return {'records':len(records), 'record_types':sorted(set(kinds)), 'history_records':2,
            'config_verified':True, 'completed_summary_verified':True,
            'automatic_host_environment_code_files_absent':True, 'metadata_artifact_record_present':True,
            'metadata_artifact_bytes_in_durable_run_tree_verified':True,
            'format':'v0 single block, FULL records, IEEE CRC32, SDK protobuf'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New, dedicated smoke directory")
    parser.add_argument("--project", default="representation-collapse")
    parser.add_argument("--entity", help="Actual owning entity; required for explicit online mode")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    args = parser.parse_args(argv)
    if args.mode == "online" and not args.entity:
        parser.error("online smoke requires the actual authenticated entity")
    source = Path(__file__).resolve()
    reporter = ExperimentReporter(args.output, project=args.project, entity=args.entity, mode=args.mode,
        config={"protocol_id": "reporting-infrastructure-smoke", "scientific_acceptance": False,
                "fits": 0, "native_forwards": 0, "records": 2},
        identities={"smoke_source_sha256": file_sha256(source),
                    "reporter_source_sha256": file_sha256(source.parents[2] / "src/representation_collapse/reporting.py")})
    with reporter:
        reporter.log({"smoke/sequence": 0, "smoke/check_passed": True}, step=0)
        reporter.log({"smoke/sequence": 1, "smoke/check_passed": True}, step=1)
        reporter.log_manifest('reporting-smoke', metadata={'protocol_id': 'reporting-infrastructure-smoke',
                              'native_forwards': 0}, identities=reporter.identities)
    receipt = json.loads(reporter.receipt_path.read_text())
    streams = list(Path(receipt["wandb_run_dir"]).glob("run-*.wandb"))
    if not streams or any(p.stat().st_size == 0 for p in streams):
        raise RuntimeError("durable W&B transaction stream missing")
    verification = {"schema_version": 1, "kind": "reporting_lifecycle_smoke",
        "scientific_acceptance": False, "native_forwards": 0, "fits": 0,
        "mode": args.mode, "run_id": receipt["run_id"], "run_url": receipt["run_url"],
        "receipt_sha256": file_sha256(reporter.receipt_path),
        "transaction_streams": [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in streams],
        "completed": receipt["state"] == "completed", "remote_history_verified": False,
        "stream_verification": [verify_smoke_stream(p, receipt) for p in streams],
        "limitation": "Local transaction stream and finish receipt do not establish server receipt; independently query remote history/summary after online run or authenticated sync."}
    path = args.output / "smoke-verification.json"
    path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verification_path": str(path), "run_id": receipt["run_id"],
                      "mode": args.mode, "completed": verification["completed"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
