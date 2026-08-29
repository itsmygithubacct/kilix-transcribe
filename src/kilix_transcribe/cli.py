"""Candidate command shell that refuses every operation requiring a runtime."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .surface import COMMANDS, RuntimeUnselected, SurfaceError, inspect_command


def parser() -> argparse.ArgumentParser:
    candidate = argparse.ArgumentParser(prog="kilix-transcribe")
    commands = candidate.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        commands.add_parser(command)
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        payload = inspect_command(arguments.command)
    except RuntimeUnselected as error:
        print(str(error), file=sys.stderr)
        return 69
    except SurfaceError as error:
        print(f"KILIX_TRANSCRIBE_REFUSAL [{error.code}] {error}", file=sys.stderr)
        return 64
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0
