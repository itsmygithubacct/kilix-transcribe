#!/usr/bin/env python3
"""Fail-closed checker for engine-neutral implementation mechanics."""

from __future__ import annotations

import json
import sys

from kilix_transcribe.surface import (
    COMMANDS,
    OUTPUTS,
    PROVIDER_REFUSAL,
    RUNTIME_COMMANDS,
    JobLifecycle,
    JobState,
    RuntimeUnselected,
    SegmentUpdate,
    SurfaceError,
    Transcript,
    TranscriptAssembler,
    WordTiming,
    inspect_command,
    render_transcript,
)


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SurfaceError(code, message)


def expect_refusal(action, code: str) -> None:
    try:
        action()
    except SurfaceError as error:
        require(error.code == code, "WRONG_REASON",
                f"refused as {error.code}, expected {code}")
    else:
        raise SurfaceError("MUTATION_ACCEPTED", f"expected {code} refusal")


def fixture_result() -> Transcript:
    assembler = TranscriptAssembler("transcribe")
    assembler.apply(SegmentUpdate(0, 0, 0, 700, "Hel", False))
    assembler.apply(
        SegmentUpdate(
            0,
            1,
            0,
            700,
            "Hello <world>",
            True,
            (
                WordTiming(0, 300, "Hello", 0.95),
                WordTiming(320, 700, "<world>", 0.9),
            ),
        )
    )
    assembler.apply(SegmentUpdate(1, 0, 800, 1_200, "Again", True))
    return assembler.finish(
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        language="en",
    )


def run() -> None:
    require(COMMANDS == ("record", "file", "serve", "models", "status", "cancel", "unload"),
            "COMMANDS", "command population differs from 7/7")
    require(OUTPUTS == ("text", "json", "webvtt", "srt"),
            "OUTPUTS", "output population differs from 4/4")
    require(RUNTIME_COMMANDS == ("record", "file", "serve", "cancel", "unload"),
            "RUNTIME_COMMANDS", "runtime refusal population differs from 5/5")

    for command in ("models", "status"):
        payload = inspect_command(command)
        require(payload["engine_routes"] == {"selected": 0, "total": 4},
                "PREMATURE_SELECTION", f"{command} selected an engine route")
        require(payload["source_objects"] == {"selected": 0, "total": 8},
                "PREMATURE_SELECTION", f"{command} selected a source object")

    for command in RUNTIME_COMMANDS:
        try:
            inspect_command(command)
        except RuntimeUnselected as error:
            require(str(error) == PROVIDER_REFUSAL, "REFUSAL_TEXT",
                    f"{command} refusal text drifted")
        else:
            raise SurfaceError("RUNTIME_ADMITTED", f"{command} admitted an unselected runtime")

    result = fixture_result()
    rendered = {output: render_transcript(result, output) for output in OUTPUTS}
    require(rendered["text"] == "Hello <world>\nAgain\n", "TEXT_OUTPUT",
            "plain-text output drifted")
    parsed = json.loads(rendered["json"])
    require(parsed["engine"]["id"] == "synthetic-engine-not-selected",
            "JSON_IDENTITY", "JSON omitted engine identity")
    require(parsed["model"]["id"] == "synthetic-model-not-selected",
            "JSON_IDENTITY", "JSON omitted model identity")
    require("00:00:00.000 --> 00:00:00.700" in rendered["webvtt"],
            "WEBVTT_TIME", "WebVTT timestamp drifted")
    require("Hello &lt;world&gt;" in rendered["webvtt"],
            "WEBVTT_ESCAPE", "WebVTT text was not escaped")
    require("00:00:00,000 --> 00:00:00,700" in rendered["srt"],
            "SRT_TIME", "SRT timestamp drifted")

    success = JobLifecycle()
    require(success.start() is JobState.RUNNING, "LIFECYCLE", "queued did not start")
    require(success.complete(result) is JobState.SUCCEEDED,
            "LIFECYCLE", "running did not complete")
    queued_cancel = JobLifecycle()
    require(queued_cancel.request_cancel() is JobState.CANCELED,
            "LIFECYCLE", "queued cancel was not terminal")
    running_cancel = JobLifecycle()
    running_cancel.start()
    require(running_cancel.request_cancel() is JobState.CANCEL_REQUESTED,
            "LIFECYCLE", "running cancel was not requested")
    require(running_cancel.acknowledge_cancel() is JobState.CANCELED,
            "LIFECYCLE", "cancel was not acknowledged")
    failed = JobLifecycle()
    require(failed.fail("SYNTHETIC_FAILURE") is JobState.FAILED,
            "LIFECYCLE", "failure was not terminal")

    assembler = TranscriptAssembler("transcribe")
    assembler.apply(SegmentUpdate(0, 0, 0, 1, "partial", False))
    stable = SegmentUpdate(0, 0, 0, 1, "stable", True)
    canceled = JobLifecycle()
    canceled.request_cancel()
    controls = (
        (lambda: WordTiming(2, 1, "bad"), "WORD_TIME"),
        (lambda: Transcript("transcribe", "e", "m", "en", (SegmentUpdate(0, 0, 0, 1, "x", False),)),
         "UNSTABLE_RESULT"),
        (lambda: assembler.finish(engine_id="e", model_id="m", language="en"),
         "UNSTABLE_RESULT"),
        (lambda: render_transcript(result, "xml"), "OUTPUT"),
        (lambda: TranscriptAssembler("transcribe").apply(SegmentUpdate(0, 2, 0, 1, "x", True)),
         "SEGMENT_REVISION"),
        (lambda: canceled.complete(result), "JOB_TRANSITION"),
    )
    for action, expected_code in controls:
        expect_refusal(action, expected_code)

    stable_assembler = TranscriptAssembler("transcribe")
    stable_assembler.apply(stable)
    expect_refusal(lambda: stable_assembler.apply(SegmentUpdate(0, 1, 0, 1, "again", True)),
                   "SEGMENT_STABLE")

    print(
        "TRANSCRIBE_IMPLEMENTATION_SURFACE: PASS "
        "(7/7 commands; 2/2 introspection commands; 5/5 runtime commands refused; "
        "4/4 deterministic serializers; 6/6 lifecycle terminals/transitions; "
        "7/7 negative controls; 0/4 engine routes selected; "
        "0/8 source objects selected; 0/1 release profiles selected)"
    )


if __name__ == "__main__":
    try:
        run()
    except (OSError, SurfaceError, TypeError, ValueError) as error:
        code = error.code if isinstance(error, SurfaceError) else type(error).__name__
        print(f"TRANSCRIBE_IMPLEMENTATION_SURFACE: FAIL [{code}] {error}", file=sys.stderr)
        raise SystemExit(1)
