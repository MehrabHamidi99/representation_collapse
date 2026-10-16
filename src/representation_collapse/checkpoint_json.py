"""Bounded canonical JSON primitives for the selected-inference v1 protocol.

This module performs no I/O and imports no numerical or model libraries. It is
not a JSON Schema validator: callers must verify the pinned schema bytes and
apply its closed artifact shapes, field bounds, and mandatory semantic rules.
In particular, parsing a record does not establish its identity or authority.
The per-file cap can be lowered for smaller artifacts; it cannot exceed 256 KiB.
"""
from __future__ import annotations

import hashlib
import json
import math


SCHEMA_SHA256 = "db2baaa64f891533f5c2c36229dd250733f675a0a2045d0436fc93d3145f2acc"
SCHEMA_BYTES = 849118
MAX_METADATA_BYTES = 262144
MAX_DEPTH = 8
MAX_OBJECT_KEYS = 64
MAX_STRING_BYTES = 256
MAX_INTEGER_DIGITS = 10
MAX_NUMBER_CHARS = 64


class CheckpointJSONError(ValueError):
    """Invalid protocol encoding, primitive value, or helper contract."""


def verify_schema_bytes(data: bytes) -> None:
    """Check a caller-supplied snapshot against the accepted exact registry.

    Registry bytes have a separate bound; metadata instance limits do not apply
    to the schema itself. No untrusted schema is parsed or used as authority.
    """
    if (type(data) is not bytes or len(data) != SCHEMA_BYTES
            or hashlib.sha256(data).hexdigest() != SCHEMA_SHA256):
        raise CheckpointJSONError("accepted v1 schema bytes required")


def require_int(value: object, *, minimum: int, maximum: int) -> int:
    """Require a built-in integer, excluding bool/float, in inclusive bounds."""
    if type(minimum) is not int or type(maximum) is not int or minimum > maximum:
        raise CheckpointJSONError("invalid integer field bounds")
    if (type(value) is not int or not minimum <= value <= maximum
            or not -(10 ** MAX_INTEGER_DIGITS) < value < 10 ** MAX_INTEGER_DIGITS):
        raise CheckpointJSONError("bounded built-in integer required")
    return value


def _string(value: object) -> str:
    if type(value) is not str or not value.isascii() or len(value) > MAX_STRING_BYTES:
        raise CheckpointJSONError("bounded ASCII string required")
    return value


def require_keys(value: object, keys: tuple[str, ...]) -> dict:
    """Check one closed record against caller-supplied trusted keys.

    This returns the original dict; nested shapes/values are not validated.
    Keys must be an exact tuple of distinct, bounded ASCII strings.
    """
    if type(keys) is not tuple or len(keys) > MAX_OBJECT_KEYS:
        raise CheckpointJSONError("bounded trusted key tuple required")
    for key in keys:
        _string(key)
    if len(set(keys)) != len(keys):
        raise CheckpointJSONError("duplicate trusted key")
    if type(value) is not dict or len(value) != len(keys):
        raise CheckpointJSONError("closed built-in record required")
    for key in value:
        _string(key)
    if set(value) != set(keys):
        raise CheckpointJSONError("unknown or missing record key")
    return value


def _byte_cap(max_bytes: int) -> int:
    return require_int(max_bytes, minimum=1, maximum=MAX_METADATA_BYTES)


def canonical_json(value: object, *, max_bytes: int = MAX_METADATA_BYTES) -> bytes:
    """Encode an exact primitive tree as sorted compact ASCII JSON plus one LF.

    Only built-in dict/list/str/int/float/bool/None are supported. Container
    depth counts the root container as one. Output is bounded while encoding;
    list lengths and artifact-specific keys remain the caller's responsibility.
    """
    limit = _byte_cap(max_bytes)
    output = bytearray()

    def emit(part: bytes) -> None:
        if len(output) + len(part) > limit:
            raise CheckpointJSONError("metadata byte limit exceeded")
        output.extend(part)

    def encode(item: object, depth: int) -> None:
        kind = type(item)
        if kind in (dict, list):
            if depth >= MAX_DEPTH:
                raise CheckpointJSONError("container depth limit exceeded")
            if kind is dict:
                if len(item) > MAX_OBJECT_KEYS:
                    raise CheckpointJSONError("object key limit exceeded")
                for key in item:
                    _string(key)
                emit(b"{")
                for index, key in enumerate(sorted(item)):
                    if index:
                        emit(b",")
                    emit(json.dumps(key, ensure_ascii=True).encode("ascii"))
                    emit(b":")
                    encode(item[key], depth + 1)
                emit(b"}")
            else:
                emit(b"[")
                for index, child in enumerate(item):
                    if index:
                        emit(b",")
                    encode(child, depth + 1)
                emit(b"]")
        elif kind is str:
            emit(json.dumps(_string(item), ensure_ascii=True).encode("ascii"))
        elif kind is bool:
            emit(b"true" if item else b"false")
        elif item is None:
            emit(b"null")
        elif kind is int:
            require_int(item, minimum=-9999999999, maximum=9999999999)
            emit(str(item).encode("ascii"))
        elif kind is float:
            if not math.isfinite(item):
                raise CheckpointJSONError("finite number required")
            token = json.dumps(item, allow_nan=False)
            if len(token) > MAX_NUMBER_CHARS:
                raise CheckpointJSONError("number token limit exceeded")
            emit(token.encode("ascii"))
        else:
            raise CheckpointJSONError("unsupported primitive type")

    encode(value, 0)
    emit(b"\n")
    return bytes(output)


def _check_depth(text: str) -> None:
    """Bound containers before json.loads allocates the decoded tree."""
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise CheckpointJSONError("container depth limit exceeded")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise CheckpointJSONError("invalid container nesting")


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    if len(pairs) > MAX_OBJECT_KEYS:
        raise CheckpointJSONError("object key limit exceeded")
    result = {}
    for key, value in pairs:
        _string(key)
        if key in result:
            raise CheckpointJSONError("duplicate JSON key")
        result[key] = value
    return result


def _integer(token: str) -> int:
    if len(token.lstrip("-")) > MAX_INTEGER_DIGITS or len(token) > MAX_NUMBER_CHARS:
        raise CheckpointJSONError("integer token limit exceeded")
    return int(token)


def _float(token: str) -> float:
    if len(token) > MAX_NUMBER_CHARS:
        raise CheckpointJSONError("number token limit exceeded")
    value = float(token)
    if not math.isfinite(value):
        raise CheckpointJSONError("finite number required")
    return value


def _constant(token: str) -> None:
    raise CheckpointJSONError("nonfinite JSON constant forbidden")


def parse_canonical_json(data: bytes, *, max_bytes: int = MAX_METADATA_BYTES) -> object:
    """Decode bounded canonical bytes; reject alternate encodings of a value.

    Duplicate keys are rejected after JSON escape decoding, numeric token limits
    apply before conversion, and container depth is checked before decoding.
    There is no text/bytearray coercion, file access, schema-shape inference,
    timestamp interpretation, or authority inferred from the decoded tree.
    """
    limit = _byte_cap(max_bytes)
    if type(data) is not bytes or not 1 <= len(data) <= limit or not data.isascii():
        raise CheckpointJSONError("bounded ASCII metadata bytes required")
    text = data.decode("ascii")
    _check_depth(text)
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_int=_integer,
                           parse_float=_float, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CheckpointJSONError("invalid JSON metadata") from exc
    if canonical_json(value, max_bytes=limit) != data:
        raise CheckpointJSONError("canonical JSON bytes required")
    return value
