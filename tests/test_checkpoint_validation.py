"""Synthetic serialized metrics only; no data, model, or native imports."""
import ast
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path
import unittest
from unittest import mock

from representation_collapse import checkpoint_json as cj
from representation_collapse.checkpoint_validation import validate_validation_artifact


ROOT = Path(__file__).resolve().parents[1]


def valid_artifact(*, epoch=1, counts=None):
    """Fictional count witnesses, including each zero-division branch."""
    cases = counts or ((0, 0, 0, 9900), (0, 0, 4, 9896), (0, 7, 0, 9893),
                       (7, 5, 3, 9885), (9900, 0, 0, 0))
    rows = []
    for index, target in enumerate(range(2560, 2816)):
        tp, fp, fn, tn = cases[index % len(cases)]
        rows.append({
            "target_timestamp": target, "source_timestamp": target - 1,
            "target_phase": target % 64, "source_phase": (target - 1) % 64,
            "phase_transition_expected": target % 32 == 0,
            "change_point": target % 32 == 0, "pairs": 9900,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / 9900,
        })
    return {"kind": "tgrab_complete_validation_rows_v1", "schema_version": 1,
            "split": "validation", "epoch": epoch, "rows": rows,
            "time_mean_f1": math.fsum(row["f1"] for row in rows) / 256}


class CheckpointValidationTests(unittest.TestCase):
    def test_valid_artifact_and_epoch_endpoints_without_mutation(self):
        for epoch in (1, 200):
            value = valid_artifact(epoch=epoch)
            before = deepcopy(value)
            self.assertIsNone(validate_validation_artifact(value))
            self.assertEqual(value, before)
            parsed = cj.parse_canonical_json(cj.canonical_json(value))
            validate_validation_artifact(parsed)

    def test_schema_pin_and_every_row_shape_constant(self):
        data = (ROOT / "configs/graph/tgrab-selected-checkpoint-schema-v1.json").read_bytes()
        cj.verify_schema_bytes(data)
        schema = json.loads(data)["$defs"]["validation"]
        value = valid_artifact()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(value))
        self.assertEqual(set(schema["properties"]), set(value))
        self.assertEqual(schema["properties"]["epoch"], {"type": "integer", "minimum": 1, "maximum": 200})
        for key in ("kind", "schema_version", "split"):
            self.assertEqual(schema["properties"][key]["const"], value[key])
        row_schemas = schema["properties"]["rows"]
        self.assertEqual(row_schemas["minItems"], 256)
        self.assertEqual(row_schemas["maxItems"], 256)
        self.assertIs(row_schemas["items"], False)
        for row, row_schema in zip(value["rows"], row_schemas["prefixItems"]):
            self.assertFalse(row_schema["additionalProperties"])
            self.assertEqual(set(row_schema["required"]), set(row))
            self.assertEqual(set(row_schema["properties"]), set(row))
            for key, field in row_schema["properties"].items():
                if "const" in field:
                    self.assertIs(type(row[key]), bool if field["type"] == "boolean" else int)
                    self.assertEqual(field["const"], row[key])
                elif field["type"] == "integer":
                    self.assertEqual((field["minimum"], field["maximum"]), (0, 9900))
                else:
                    self.assertEqual(field, {"type": "number", "minimum": 0, "maximum": 1})

    def test_against_actual_gr_scalar_formula_source_without_import(self):
        tree = ast.parse((ROOT / "src/representation_collapse/graph_reconstruction.py").read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        # Evaluate only the four existing return-value expressions with synthetic
        # integer counts; importing GR would import NumPy and is out of scope.
        returned = next(node.value for node in functions["snapshot_metrics"].body if isinstance(node, ast.Return))
        expressions = {key.value: val for key, val in zip(returned.keys, returned.values)
                       if isinstance(key, ast.Constant) and key.value in ("precision", "recall", "f1", "accuracy")}
        self.assertEqual(set(expressions), {"precision", "recall", "f1", "accuracy"})
        for row in valid_artifact()["rows"][:5]:
            namespace = {key: row[key] for key in ("tp", "fp", "fn", "tn")}
            namespace["pairs"] = range(9900)
            for key, expression in expressions.items():
                result = eval(compile(ast.Expression(expression), "gr-scalar-source", "eval"), {"__builtins__": {"len": len}}, namespace)
                self.assertEqual(result, row[key])
        # Execute only the two dependency-free scalar helper bodies. Replace
        # their boundary helpers with strict test stubs, never importing GR.
        scalar_functions = deepcopy([functions["phase_record"], functions["time_mean_f1"]])
        for function in scalar_functions:
            function.returns = None
            for argument in (*function.args.args, *function.args.kwonlyargs):
                argument.annotation = None
        module = ast.Module(body=scalar_functions, type_ignores=[])
        namespace = {"__builtins__": {"len": len}, "math": math,
                     "_integer": lambda value, *args: cj.require_int(value, minimum=0, maximum=9999999999),
                     "_finite": lambda value, name: float(value), "ContractError": ValueError}
        exec(compile(module, "gr-scalar-helpers-source", "exec"), namespace)
        artifact = valid_artifact()
        for row in artifact["rows"]:
            self.assertEqual(namespace["phase_record"](row["target_timestamp"]),
                             {key: row[key] for key in ("target_timestamp", "source_timestamp", "target_phase", "source_phase", "phase_transition_expected")})
        self.assertEqual(namespace["time_mean_f1"](artifact["rows"]), artifact["time_mean_f1"])

    def test_top_level_missing_and_unknown_keys(self):
        for key in valid_artifact():
            value = valid_artifact(); del value[key]
            with self.subTest(missing=key), self.assertRaises(cj.CheckpointJSONError):
                validate_validation_artifact(value)
        value = valid_artifact(); value["extra"] = None
        with self.assertRaises(cj.CheckpointJSONError):
            validate_validation_artifact(value)

    def test_row_missing_and_unknown_keys(self):
        for key in valid_artifact()["rows"][0]:
            value = valid_artifact(); del value["rows"][0][key]
            with self.subTest(missing=key), self.assertRaises(cj.CheckpointJSONError):
                validate_validation_artifact(value)
        value = valid_artifact(); value["rows"][0]["extra"] = None
        with self.assertRaises(cj.CheckpointJSONError):
            validate_validation_artifact(value)

    def test_kind_version_split_epoch_mutations(self):
        cases = {"kind": ("unknown", None, True), "split": ("test", "train", None),
                 "schema_version": (2, True, 1.0, "1"),
                 "epoch": (0, 201, True, 1.0, "1", None)}
        for key, wrong_values in cases.items():
            for wrong in wrong_values:
                value = valid_artifact(); value[key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaises(cj.CheckpointJSONError):
                    validate_validation_artifact(value)

    def test_builtin_record_and_rows_types_only(self):
        class Record(dict):
            pass
        class Rows(list):
            pass
        for value in (None, [], Record(valid_artifact())):
            with self.assertRaises(cj.CheckpointJSONError):
                validate_validation_artifact(value)
        for wrong in (None, {}, tuple(valid_artifact()["rows"]), Rows(valid_artifact()["rows"])):
            value = valid_artifact(); value["rows"] = wrong
            with self.assertRaises(cj.CheckpointJSONError):
                validate_validation_artifact(value)
        for wrong in (None, [], Record(valid_artifact()["rows"][0])):
            value = valid_artifact(); value["rows"][0] = wrong
            with self.assertRaises(cj.CheckpointJSONError):
                validate_validation_artifact(value)

    def test_exact_complete_ordered_timestamps(self):
        for operation in ("remove", "append", "swap", "duplicate"):
            value = valid_artifact(); rows = value["rows"]
            if operation == "remove": rows.pop()
            elif operation == "append": rows.append(deepcopy(rows[-1]))
            elif operation == "swap": rows[0], rows[1] = rows[1], rows[0]
            else: rows[1] = deepcopy(rows[0])
            with self.subTest(operation=operation), self.assertRaises(cj.CheckpointJSONError):
                validate_validation_artifact(value)
        for key in ("target_timestamp", "source_timestamp", "target_phase", "source_phase", "pairs"):
            original = valid_artifact()["rows"][0][key]
            for wrong in (original + 1, float(original), True, str(original)):
                value = valid_artifact(); value["rows"][0][key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaises(cj.CheckpointJSONError):
                    validate_validation_artifact(value)

    def test_phase_and_change_flags_include_both_boundaries(self):
        for index in (0, 1, 31, 32, 63, 64, 255):
            for key in ("phase_transition_expected", "change_point"):
                for wrong in (not valid_artifact()["rows"][index][key], 0, 1, None):
                    value = valid_artifact(); value["rows"][index][key] = wrong
                    with self.subTest(index=index, key=key, wrong=wrong), self.assertRaises(cj.CheckpointJSONError):
                        validate_validation_artifact(value)

    def test_confusion_count_bounds_types_and_sum(self):
        for key in ("tp", "fp", "fn", "tn"):
            for wrong in (-1, 9901, True, False, 0.0, "0", None):
                value = valid_artifact(); value["rows"][3][key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaises(cj.CheckpointJSONError):
                    validate_validation_artifact(value)
        value = valid_artifact(); value["rows"][3]["tn"] += 1
        with self.assertRaisesRegex(cj.CheckpointJSONError, "sum"):
            validate_validation_artifact(value)

    def test_metrics_strict_types_ranges_and_nonfinite(self):
        class FloatChild(float):
            pass
        for key in ("precision", "recall", "f1", "accuracy", "time_mean_f1"):
            for wrong in (True, False, -0.01, 1.01, math.nan, math.inf, -math.inf,
                          "0", None, Decimal("0"), FloatChild(0.0), 10 ** 5000):
                value = valid_artifact()
                (value if key == "time_mean_f1" else value["rows"][3])[key] = wrong
                with self.subTest(key=key, wrong_type=type(wrong)), self.assertRaises(cj.CheckpointJSONError):
                    validate_validation_artifact(value)

    def test_formula_mismatch_by_one_float_step_rejected(self):
        for key in ("precision", "recall", "f1", "accuracy"):
            value = valid_artifact()
            value["rows"][3][key] = math.nextafter(value["rows"][3][key], 0.0)
            with self.subTest(key=key), self.assertRaisesRegex(cj.CheckpointJSONError, "formula"):
                validate_validation_artifact(value)

    def test_zero_division_and_integer_number_endpoints(self):
        value = valid_artifact(counts=((0, 0, 0, 9900),))
        for row in value["rows"]:
            for key in ("precision", "recall", "f1"):
                row[key] = 0
            row["accuracy"] = 1
        value["time_mean_f1"] = 0
        validate_validation_artifact(value)

    def test_mean_exact_fsum_and_no_pooled_f1(self):
        count_cases = tuple(
            (tp, fp, fn, 9900 - tp - fp - fn)
            for tp, fp, fn in ((1, 1, 0), (2, 3, 1), (3, 2, 4),
                               (11, 17, 5), (23, 7, 29))
        )
        value = valid_artifact(counts=(count_cases * 52)[:256])
        # Python 3.14's built-in sum uses compensated summation for floats, so
        # model the historical left-to-right reduction explicitly here.
        left_fold = 0.0
        for row in value["rows"]:
            left_fold += row["f1"]
        left_fold /= 256
        self.assertNotEqual(left_fold, value["time_mean_f1"])
        value["time_mean_f1"] = left_fold
        with self.assertRaisesRegex(cj.CheckpointJSONError, "time-mean"):
            validate_validation_artifact(value)
        value = valid_artifact()
        tp, fp, fn = (sum(row[key] for row in value["rows"]) for key in ("tp", "fp", "fn"))
        pooled = 2 * tp / (2 * tp + fp + fn)
        self.assertNotEqual(pooled, value["time_mean_f1"])
        value["time_mean_f1"] = pooled
        with self.assertRaisesRegex(cj.CheckpointJSONError, "time-mean"):
            validate_validation_artifact(value)
        value = valid_artifact()
        value["time_mean_f1"] = math.nextafter(value["time_mean_f1"], 1.0)
        with self.assertRaises(cj.CheckpointJSONError):
            validate_validation_artifact(value)

    def test_no_io_needed_for_validation(self):
        value = valid_artifact()
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
            validate_validation_artifact(value)


if __name__ == "__main__":
    unittest.main()
