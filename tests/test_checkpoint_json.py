"""Local primitive metadata fixtures; no checkpoint or numerical payloads."""
import hashlib
import math
from pathlib import Path
import unittest
from unittest import mock

from representation_collapse import checkpoint_json as cj


class CheckpointJSONTests(unittest.TestCase):
    def test_pinned_accepted_schema_snapshot(self):
        data = (Path(__file__).resolve().parents[1]
                / "configs/graph/tgrab-selected-checkpoint-schema-v1.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), cj.SCHEMA_SHA256)
        cj.verify_schema_bytes(data)
        for altered in (data[:-1], b" " + data[1:], bytearray(data), data.decode("ascii")):
            with self.subTest(type=type(altered), length=len(altered)):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.verify_schema_bytes(altered)

    def test_exact_canonical_output_and_round_trip(self):
        value = {"z": [True, None, -9999999999, 9999999999, 1.0],
                 "a": {"quote": '"\\\t\n', "empty": {}}}
        expected = b'{"a":{"empty":{},"quote":"\\\"\\\\\\t\\n"},"z":[true,null,-9999999999,9999999999,1.0]}\n'
        self.assertEqual(cj.canonical_json(value), expected)
        self.assertEqual(cj.parse_canonical_json(expected), value)

    def test_scalars_and_descriptor_arrays_are_supported(self):
        for value in (None, True, False, 0, -1, 0.0, "", ["hidden", "cell"], {}):
            with self.subTest(value=value):
                result = cj.parse_canonical_json(cj.canonical_json(value))
                self.assertIs(type(result), type(value))
                self.assertEqual(result, value)

    def test_negative_float_zero_and_small_finite_float(self):
        for value in (-0.0, 5e-324, 1.7976931348623157e308):
            result = cj.parse_canonical_json(cj.canonical_json(value))
            self.assertEqual(result, value)
            self.assertEqual(math.copysign(1, result), math.copysign(1, value))

    def test_noncanonical_alternate_encodings(self):
        for data in (b'{}', b'{}\n\n', b'{}\r\n', b' {}\n', b'{ }\n',
                     b'{"b":1,"a":2}\n', b'1e0\n', b'-0\n',
                     b'"\\u0061"\n', b'"\\/"\n', b'1e-999\n'):
            with self.subTest(data=data):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.parse_canonical_json(data)

    def test_duplicate_keys_including_decoded_escape_alias(self):
        for data in (b'{"a":1,"a":2}\n', b'{"a":1,"\\u0061":2}\n',
                     b'{"nested":{"x":0,"x":0}}\n'):
            with self.subTest(data=data):
                with self.assertRaisesRegex(cj.CheckpointJSONError, "duplicate JSON key"):
                    cj.parse_canonical_json(data)

    def test_depth_eight_succeeds_nine_rejected_before_decode(self):
        value = 0
        for _ in range(8):
            value = [value]
        self.assertEqual(cj.parse_canonical_json(cj.canonical_json(value)), value)
        with self.assertRaisesRegex(cj.CheckpointJSONError, "depth"):
            cj.canonical_json([value])
        with mock.patch.object(cj.json, "loads", side_effect=AssertionError("decode called")):
            with self.assertRaisesRegex(cj.CheckpointJSONError, "depth"):
                cj.parse_canonical_json(b"[" * 9 + b"0" + b"]" * 9 + b"\n")

    def test_string_brackets_and_escaped_quotes_do_not_change_depth(self):
        value = {"s": ('[{' + '\\"' + '}]') * 20}
        self.assertEqual(cj.parse_canonical_json(cj.canonical_json(value)), value)

    def test_object_key_bounds(self):
        value = {str(i): i for i in range(64)}
        self.assertEqual(cj.parse_canonical_json(cj.canonical_json(value)), value)
        value["64"] = 64
        with self.assertRaisesRegex(cj.CheckpointJSONError, "key limit"):
            cj.canonical_json(value)
        data = ("{" + ",".join('"k%d":0' % i for i in range(65)) + "}\n").encode()
        with self.assertRaisesRegex(cj.CheckpointJSONError, "key limit"):
            cj.parse_canonical_json(data)

    def test_ascii_string_and_key_bounds(self):
        for value in ("a" * 256, {"a" * 256: "ok"}):
            self.assertEqual(cj.parse_canonical_json(cj.canonical_json(value)), value)
        for value in ("a" * 257, {"a" * 257: 0}, "é", {"é": 0}, "\ud800"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.canonical_json(value)
        for data in (b'"\\u00e9"\n', b'{"\\ud800":0}\n', b'"' + b"a" * 257 + b'"\n',
                     '"é"\n'.encode()):
            with self.subTest(data=data):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.parse_canonical_json(data)

    def test_nonfinite_numbers_and_constants(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.canonical_json(value)
        for data in (b'NaN\n', b'Infinity\n', b'-Infinity\n', b'1e999\n'):
            with self.subTest(data=data):
                with self.assertRaisesRegex(cj.CheckpointJSONError, "finite"):
                    cj.parse_canonical_json(data)

    def test_numeric_token_limits_before_conversion(self):
        for data in (b'10000000000\n', b'-10000000000\n', b'9' * 5000 + b'\n'):
            with self.assertRaisesRegex(cj.CheckpointJSONError, "integer token limit"):
                cj.parse_canonical_json(data)
        for data in (b'0.' + b'0' * 63 + b'1\n', b'1e' + b'0' * 63 + b'1\n'):
            with self.assertRaisesRegex(cj.CheckpointJSONError, "number token limit"):
                cj.parse_canonical_json(data)
        for value in (10000000000, -10000000000):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.canonical_json(value)

    def test_byte_caps_include_terminal_lf(self):
        self.assertEqual(cj.canonical_json({}, max_bytes=3), b'{}\n')
        self.assertEqual(cj.parse_canonical_json(b'{}\n', max_bytes=3), {})
        with self.assertRaises(cj.CheckpointJSONError):
            cj.canonical_json({}, max_bytes=2)
        with self.assertRaises(cj.CheckpointJSONError):
            cj.parse_canonical_json(b'{}\n', max_bytes=2)
        with mock.patch.object(cj.json, "loads", side_effect=AssertionError("decode called")):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.parse_canonical_json(b" " * (cj.MAX_METADATA_BYTES + 1))
        with self.assertRaises(cj.CheckpointJSONError):
            cj.canonical_json([0] * cj.MAX_METADATA_BYTES, max_bytes=4096)

    def test_invalid_caps_and_nonbytes_inputs(self):
        for cap in (True, 1.0, 0, -1, cj.MAX_METADATA_BYTES + 1):
            with self.subTest(cap=cap):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.canonical_json({}, max_bytes=cap)
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.parse_canonical_json(b'{}\n', max_bytes=cap)
        for value in ('{}\n', bytearray(b'{}\n'), memoryview(b'{}\n'), None, b''):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.parse_canonical_json(value)

    def test_unsupported_types_and_subclasses(self):
        class IntChild(int):
            pass
        class DictChild(dict):
            pass
        class StrChild(str):
            pass
        for value in ((1, 2), {1, 2}, b"bytes", object(), IntChild(1), DictChild(),
                      StrChild("ok"), {1: "bad"}, {StrChild("a"): 0}):
            with self.subTest(type=type(value)):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.canonical_json(value)
        value = []
        value.append(value)
        with self.assertRaisesRegex(cj.CheckpointJSONError, "depth"):
            cj.canonical_json(value)

    def test_invalid_json_and_extra_documents(self):
        for data in (b'{\n', b']\n', b'[{]}\n', b'{"a":1,}\n', b'01\n',
                     b'{}\n{}\n', b'"raw\nnewline"\n', b'\xef\xbb\xbf{}\n'):
            with self.subTest(data=data):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.parse_canonical_json(data)

    def test_strict_integer_field_helper(self):
        self.assertEqual(cj.require_int(1, minimum=1, maximum=200), 1)
        self.assertEqual(cj.require_int(200, minimum=1, maximum=200), 200)
        for value in (True, False, 1.0, "1", None, 0, 201):
            with self.subTest(value=value):
                with self.assertRaises(cj.CheckpointJSONError):
                    cj.require_int(value, minimum=1, maximum=200)
        for bounds in ((True, 200), (1, 200.0), (2, 1)):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.require_int(1, minimum=bounds[0], maximum=bounds[1])
        with self.assertRaises(cj.CheckpointJSONError):
            cj.require_int(10000000000, minimum=0, maximum=10000000000)

    def test_closed_record_helper_rejects_unknown_shapes(self):
        record = {"kind": "fixture", "schema_version": 1}
        self.assertIs(cj.require_keys(record, ("kind", "schema_version")), record)
        for value in ({"kind": "fixture"}, {**record, "extra": None}, [], None, {1: 0}):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.require_keys(value, ("kind", "schema_version"))
        for keys in (["kind"], ("kind", "kind"), (1,), ("é",), tuple(str(i) for i in range(65))):
            with self.assertRaises(cj.CheckpointJSONError):
                cj.require_keys({}, keys)

    def test_parser_does_not_claim_artifact_schema_validation(self):
        value = cj.parse_canonical_json(b'{"unknown_artifact":true}\n')
        with self.assertRaises(cj.CheckpointJSONError):
            cj.require_keys(value, ("kind", "schema_version"))
        value = cj.parse_canonical_json(b'{"epoch":true}\n')
        with self.assertRaises(cj.CheckpointJSONError):
            cj.require_int(value["epoch"], minimum=1, maximum=200)

    def test_primitive_operations_perform_no_file_io(self):
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
            self.assertEqual(cj.parse_canonical_json(cj.canonical_json({"a": [1]})), {"a": [1]})


if __name__ == "__main__":
    unittest.main()
