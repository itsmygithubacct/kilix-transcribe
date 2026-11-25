"""Bounded development JSON numbers; no global interpreter-limit changes."""
import math
from .fd import Refusal, require

MAX_NUMBER_CHARS = 128
MIN_INTEGER = -(1 << 63)
MAX_INTEGER = (1 << 63) - 1

def parse_integer(token):
    require(len(token) <= MAX_NUMBER_CHARS, 'MALFORMED_JSON')
    try:
        value = int(token)
    except ValueError as error:
        raise Refusal('MALFORMED_JSON') from error
    require(MIN_INTEGER <= value <= MAX_INTEGER, 'MALFORMED_JSON')
    return value

def parse_float(token):
    require(len(token) <= MAX_NUMBER_CHARS, 'MALFORMED_JSON')
    try:
        value = float(token)
    except (ValueError, OverflowError) as error:
        raise Refusal('MALFORMED_JSON') from error
    require(math.isfinite(value), 'NONFINITE_JSON_VALUE')
    # A genuine textual zero remains zero. A nonzero token that rounds to zero
    # loses evidence and must not masquerade as a silence/time/distance value.
    mantissa = token.lower().split('e', 1)[0]
    require(value != 0.0 or not any(c in '123456789' for c in mantissa), 'MALFORMED_JSON')
    return value
