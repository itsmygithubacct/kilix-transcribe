from __future__ import annotations

import contextlib
import io
import json
import unittest

from kilix_transcribe.cli import main
from kilix_transcribe.surface import (
    PROVIDER_REFUSAL,
    RUNTIME_COMMANDS,
    JobLifecycle,
    JobState,
    SegmentUpdate,
    SurfaceError,
    Transcript,
    TranscriptAssembler,
    WordTiming,
    render_transcript,
)


def result_fixture() -> Transcript:
    assembler = TranscriptAssembler("diarize")
    assembler.apply(SegmentUpdate(1, 0, 1_000, 2_000, "Second", True))
    assembler.apply(SegmentUpdate(0, 0, 0, 900, "First", False))
    assembler.apply(
        SegmentUpdate(
            0,
            1,
            0,
            900,
            "First <cue>",
            True,
            (WordTiming(0, 400, "First", 1), WordTiming(450, 900, "<cue>", 0.5)),
            "speaker-1",
            0.75,
        )
    )
    return assembler.finish(
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        language="en",
    )


class ValueTests(unittest.TestCase):
    def refusal(self, action, code: str) -> None:
        with self.assertRaises(SurfaceError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_word_timing_normalizes_confidence(self) -> None:
        self.assertEqual(WordTiming(0, 1, "a", 1).confidence, 1.0)

    def test_word_end_before_start_is_refused(self) -> None:
        self.refusal(lambda: WordTiming(2, 1, "bad"), "WORD_TIME")

    def test_word_outside_segment_is_refused(self) -> None:
        self.refusal(
            lambda: SegmentUpdate(0, 0, 10, 20, "x", True, (WordTiming(0, 20, "x"),)),
            "WORD_TIME",
        )

    def test_unordered_words_are_refused(self) -> None:
        self.refusal(
            lambda: SegmentUpdate(
                0, 0, 0, 20, "x", True,
                (WordTiming(10, 20, "b"), WordTiming(5, 9, "a")),
            ),
            "WORD_ORDER",
        )

    def test_speaker_fields_must_be_paired(self) -> None:
        self.refusal(lambda: SegmentUpdate(0, 0, 0, 1, "x", True, speaker="s"),
                     "SPEAKER_PAIR")

    def test_control_character_is_refused(self) -> None:
        self.refusal(lambda: SegmentUpdate(0, 0, 0, 1, "bad\x00text", True),
                     "SEGMENT_TEXT")

    def test_crlf_is_normalized(self) -> None:
        self.assertEqual(SegmentUpdate(0, 0, 0, 1, "a\r\nb", True).text, "a\nb")


class AssemblerTests(unittest.TestCase):
    def refusal(self, action, code: str) -> None:
        with self.assertRaises(SurfaceError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_unstable_revision_can_be_replaced(self) -> None:
        assembler = TranscriptAssembler("transcribe")
        assembler.apply(SegmentUpdate(0, 0, 0, 1, "a", False))
        assembler.apply(SegmentUpdate(0, 1, 0, 1, "b", True))
        self.assertEqual(assembler.pending_segments, 0)

    def test_first_revision_must_be_zero(self) -> None:
        assembler = TranscriptAssembler("transcribe")
        self.refusal(lambda: assembler.apply(SegmentUpdate(0, 1, 0, 1, "x", True)),
                     "SEGMENT_REVISION")

    def test_revision_must_advance_by_one(self) -> None:
        assembler = TranscriptAssembler("transcribe")
        assembler.apply(SegmentUpdate(0, 0, 0, 1, "x", False))
        self.refusal(lambda: assembler.apply(SegmentUpdate(0, 2, 0, 1, "x", True)),
                     "SEGMENT_REVISION")

    def test_stable_segment_cannot_be_replaced(self) -> None:
        assembler = TranscriptAssembler("transcribe")
        assembler.apply(SegmentUpdate(0, 0, 0, 1, "x", True))
        self.refusal(lambda: assembler.apply(SegmentUpdate(0, 1, 0, 1, "y", True)),
                     "SEGMENT_STABLE")

    def test_unstable_result_cannot_finish(self) -> None:
        assembler = TranscriptAssembler("transcribe")
        assembler.apply(SegmentUpdate(0, 0, 0, 1, "x", False))
        self.refusal(lambda: assembler.finish(engine_id="e", model_id="m", language="en"),
                     "UNSTABLE_RESULT")

    def test_finished_assembler_is_sealed(self) -> None:
        assembler = TranscriptAssembler("transcribe")
        assembler.finish(engine_id="e", model_id="m", language="en")
        self.refusal(lambda: assembler.finish(engine_id="e", model_id="m", language="en"),
                     "ASSEMBLER_SEALED")

    def test_final_segments_are_sorted_by_time(self) -> None:
        self.assertEqual([segment.segment_id for segment in result_fixture().segments], [0, 1])


class SerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = result_fixture()

    def test_text_output(self) -> None:
        self.assertEqual(render_transcript(self.result, "text"), "First <cue>\nSecond\n")

    def test_json_output_is_canonical_and_complete(self) -> None:
        rendered = render_transcript(self.result, "json")
        self.assertEqual(rendered, json.dumps(json.loads(rendered), ensure_ascii=False,
                                              separators=(",", ":"), sort_keys=True) + "\n")
        payload = json.loads(rendered)
        self.assertEqual(payload["engine"]["id"], "synthetic-engine-not-selected")
        self.assertEqual(payload["model"]["id"], "synthetic-model-not-selected")
        self.assertEqual(payload["segments"][0]["speaker"]["id"], "speaker-1")

    def test_webvtt_output(self) -> None:
        rendered = render_transcript(self.result, "webvtt")
        self.assertTrue(rendered.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:00.000 --> 00:00:00.900", rendered)
        self.assertIn("First &lt;cue&gt;", rendered)

    def test_srt_output(self) -> None:
        rendered = render_transcript(self.result, "srt")
        self.assertTrue(rendered.startswith("1\n00:00:00,000 --> 00:00:00,900\n"))

    def test_unknown_output_is_refused(self) -> None:
        with self.assertRaises(SurfaceError) as caught:
            render_transcript(self.result, "xml")
        self.assertEqual(caught.exception.code, "OUTPUT")

    def test_empty_result_serializers_are_deterministic(self) -> None:
        empty = Transcript("transcribe", "e", "m", "und", ())
        self.assertEqual(render_transcript(empty, "text"), "")
        self.assertEqual(render_transcript(empty, "webvtt"), "WEBVTT\n\n")
        self.assertEqual(render_transcript(empty, "srt"), "")


class LifecycleTests(unittest.TestCase):
    def test_success_has_exactly_one_result(self) -> None:
        job = JobLifecycle()
        job.start()
        self.assertIs(job.complete(result_fixture()), JobState.SUCCEEDED)
        self.assertIsNotNone(job.result)
        with self.assertRaises(SurfaceError):
            job.fail("LATE_FAILURE")

    def test_queued_cancel_is_terminal(self) -> None:
        job = JobLifecycle()
        self.assertIs(job.request_cancel(), JobState.CANCELED)
        self.assertIsNone(job.result)

    def test_running_cancel_requires_acknowledgement(self) -> None:
        job = JobLifecycle()
        job.start()
        self.assertIs(job.request_cancel(), JobState.CANCEL_REQUESTED)
        self.assertIs(job.acknowledge_cancel(), JobState.CANCELED)

    def test_cancel_is_idempotent_while_pending_or_canceled(self) -> None:
        job = JobLifecycle()
        job.start()
        self.assertIs(job.request_cancel(), JobState.CANCEL_REQUESTED)
        self.assertIs(job.request_cancel(), JobState.CANCEL_REQUESTED)
        job.acknowledge_cancel()
        self.assertIs(job.request_cancel(), JobState.CANCELED)

    def test_complete_after_cancel_is_refused(self) -> None:
        job = JobLifecycle()
        job.request_cancel()
        with self.assertRaises(SurfaceError) as caught:
            job.complete(result_fixture())
        self.assertEqual(caught.exception.code, "JOB_TRANSITION")

    def test_failure_retains_no_result(self) -> None:
        job = JobLifecycle()
        self.assertIs(job.fail("SYNTHETIC_FAILURE"), JobState.FAILED)
        self.assertIsNone(job.result)


class CommandTests(unittest.TestCase):
    def invoke(self, command: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([command])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_status_reports_zero_selection(self) -> None:
        code, stdout, stderr = self.invoke("status")
        self.assertEqual((code, stderr), (0, ""))
        payload = json.loads(stdout)
        self.assertEqual(payload["engine_routes"], {"selected": 0, "total": 4})
        self.assertEqual(payload["source_objects"], {"selected": 0, "total": 8})
        self.assertEqual(payload["release_profiles"], {"selected": 0, "total": 1})

    def test_models_reports_empty_population(self) -> None:
        code, stdout, stderr = self.invoke("models")
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["models"], [])

    def test_all_runtime_commands_refuse_verbatim(self) -> None:
        for command in RUNTIME_COMMANDS:
            with self.subTest(command=command):
                code, stdout, stderr = self.invoke(command)
                self.assertEqual((code, stdout), (69, ""))
                self.assertEqual(stderr, PROVIDER_REFUSAL + "\n")


if __name__ == "__main__":
    unittest.main()
