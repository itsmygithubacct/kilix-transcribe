"""kilix-transcribe-youtube, with yt-dlp, ffmpeg and MiniMax replaced by fakes.

Nothing here reaches the network or reads a real key.
"""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from kilix_transcribe_youtube import align, captions, cli, cues, fetch, minimax, paths, score
from kilix_transcribe_youtube.captions import TimedWord
from kilix_transcribe_youtube.minimax import Segment
from kilix_transcribe_youtube.video import VideoError, video_id

KEY = "fixture-key-NEVER-PRINT-THIS"
VID = "C9HS3zMrGo0"


def json3(events: list[dict]) -> str:
    return json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 99_000, "id": 1}] + events})


AUTO_TRACK = json3([
    {"tStartMs": 0, "dDurationMs": 3000, "wWinId": 1, "segs": [
        {"utf8": "Okay,"}, {"utf8": " so", "tOffsetMs": 400}, {"utf8": " quen", "tOffsetMs": 900}]},
    {"tStartMs": 2900, "dDurationMs": 100, "wWinId": 1, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 3000, "dDurationMs": 2000, "wWinId": 1, "segs": [
        {"utf8": "models"}, {"utf8": " are", "tOffsetMs": 500}, {"utf8": " here.", "tOffsetMs": 900}]},
])
UPLOADER_TRACK = json3([
    {"tStartMs": 0, "dDurationMs": 5000, "segs": [{"utf8": "Okay, so Qwen models are here."}]},
])


class TempHome(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        self.env = {"KILIX_TRANSCRIBE_YOUTUBE_HOME": str(self.root), "HOME": self.tmp.name}

    def write_key(self, mode: int = 0o600, value: str = KEY) -> Path:
        paths.private_directory(self.root)
        key = self.root / paths.KEY_NAME
        key.write_text(value + "\n")
        key.chmod(mode)
        return key


class VideoIdTests(unittest.TestCase):
    def test_ids_and_urls(self) -> None:
        for value in (VID, f"https://www.youtube.com/watch?v={VID}&t=30",
                      f"https://youtu.be/{VID}?si=x", f"https://www.youtube.com/shorts/{VID}",
                      f"https://m.youtube.com/watch?v={VID}"):
            self.assertEqual(video_id(value), VID, value)

    def test_other_hosts_and_bad_ids_are_refused(self) -> None:
        for value in (f"https://example.com/watch?v={VID}", "short", "https://youtu.be/",
                      f"file:///watch?v={VID}"):
            with self.assertRaises(VideoError, msg=value):
                video_id(value)


class KeyTests(TempHome):
    def test_default_root_is_under_local_share(self) -> None:
        self.assertEqual(paths.data_root({"HOME": "/srv/u"}),
                         Path("/srv/u/.local/share/kilix-transcribe-youtube"))
        self.assertEqual(paths.data_root({"HOME": "/srv/u", "XDG_DATA_HOME": "/x"}),
                         Path("/x/kilix-transcribe-youtube"))

    def test_private_key_is_read(self) -> None:
        self.write_key()
        self.assertEqual(paths.read_key(self.root), KEY)

    def test_a_key_others_can_read_is_refused_without_echoing_it(self) -> None:
        self.write_key(0o644)
        with self.assertRaises(paths.PrivateDataError) as caught:
            paths.read_key(self.root)
        self.assertIn("chmod 600", str(caught.exception))
        self.assertNotIn(KEY, str(caught.exception))

    def test_missing_and_symlinked_keys_are_refused(self) -> None:
        paths.private_directory(self.root)
        with self.assertRaises(paths.PrivateDataError):
            paths.read_key(self.root)
        target = Path(self.tmp.name) / "elsewhere"
        target.write_text(KEY)
        target.chmod(0o600)
        (self.root / paths.KEY_NAME).symlink_to(target)
        with self.assertRaises(paths.PrivateDataError):
            paths.read_key(self.root)

    def test_run_directories_are_private(self) -> None:
        run = paths.run_directory(self.root, VID)
        for directory in (self.root, self.root / "runs", run):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700, directory)


class CaptionTests(TempHome):
    def test_words_take_event_start_plus_offset(self) -> None:
        track = Path(self.tmp.name) / "auto.json3"
        track.write_text(AUTO_TRACK)
        words = captions.timed_words(track)
        self.assertEqual([w.text for w in words], ["Okay,", "so", "quen", "models", "are", "here."])
        self.assertEqual([w.start_ms for w in words], [0, 400, 900, 3000, 3500, 3900])
        # A word ends where the next starts; the last ends with its event.
        self.assertEqual([w.end_ms for w in words], [400, 900, 3000, 3500, 3900, 5000])


class AlignTests(unittest.TestCase):
    def test_agreed_words_take_caption_times_and_the_rest_fill_the_gap(self) -> None:
        caps = [TimedWord(0, 400, "Okay,"), TimedWord(400, 900, "so"), TimedWord(900, 3000, "quen"),
                TimedWord(3000, 3500, "models")]
        words = align.align([Segment(0, 60_000, "Okay, so Qwen models")], caps)
        self.assertEqual([w.text for w in words], ["Okay,", "so", "Qwen", "models"])
        self.assertEqual((words[0].start_ms, words[1].start_ms, words[3].start_ms), (0, 400, 3000))
        # "Qwen" disagreed with "quen", so it is placed between its neighbours.
        self.assertEqual((words[2].start_ms, words[2].end_ms), (900, 3000))

    def test_a_far_away_match_is_not_an_anchor(self) -> None:
        caps = [TimedWord(500_000, 500_400, "hello")]
        words = align.align([Segment(0, 10_000, "hello world")], caps)
        self.assertLess(words[0].start_ms, 10_000)

    def test_without_captions_words_are_spread_by_length(self) -> None:
        words = align.align([Segment(1000, 3000, "a bbb")])
        self.assertEqual(words[0].start_ms, 1000)
        self.assertEqual(words[-1].end_ms, 3000)
        self.assertLess(words[0].end_ms - words[0].start_ms, words[1].end_ms - words[1].start_ms)

    def test_times_never_go_backwards(self) -> None:
        caps = [TimedWord(5000, 5100, "b"), TimedWord(1000, 1100, "a")]
        words = align.align([Segment(0, 9000, "b a c")], sorted(caps, key=lambda w: w.text))
        starts = [w.start_ms for w in words]
        self.assertEqual(starts, sorted(starts))


class CueTests(unittest.TestCase):
    def test_long_runs_and_pauses_split_cues(self) -> None:
        words = [TimedWord(i * 300, i * 300 + 250, "word") for i in range(40)]
        words.append(TimedWord(40 * 300 + 5000, 40 * 300 + 5200, "later"))
        groups = cues.cues(words)
        self.assertGreater(len(groups), 2)
        self.assertEqual([w.text for w in groups[-1]], ["later"])
        for group in groups:
            self.assertLessEqual(sum(len(w.text) + 1 for w in group) - 1, cues.MAX_CUE_CHARS)

    def test_transcript_renders_as_srt(self) -> None:
        from kilix_transcribe.surface import render_transcript
        result = cues.transcript([TimedWord(0, 500, "Hello"), TimedWord(500, 1200, "there.")])
        self.assertEqual(render_transcript(result, "srt"),
                         "1\n00:00:00,000 --> 00:00:01,200\nHello there.\n")


def text_reply(text: str, duration: float) -> str:
    return json.dumps({"text": text, "duration": duration, "base_resp": {"status_code": 0}})


def piece_reply(text: str, duration: float, segments: list[tuple[float, float, str]]) -> str:
    return json.dumps({"text": text, "duration": duration, "base_resp": {"status_code": 0},
                       "segments": [{"id": i, "start": s, "end": e, "text": t, "speaker": "S1"}
                                    for i, (s, e, t) in enumerate(segments)]})


def requested_format(body: bytes) -> str:
    marker = b'name="response_format"\r\n\r\n'
    start = body.index(marker) + len(marker)
    return body[start:body.index(b"\r\n", start)].decode()


class FakeService:
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, str]] = []
        self.formats: list[str] = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append(headers)
        self.formats.append(requested_format(body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def fake_cutter(count: int):
    def cut(audio: Path, directory: Path) -> list[Path]:
        directory.mkdir(exist_ok=True)
        pieces = []
        for index in range(count):
            piece = directory / f"piece-{index:03d}.mp3"
            if not piece.exists():
                piece.write_bytes(b"mp3")
            pieces.append(piece)
        return pieces
    return cut


class MiniMaxTests(TempHome):
    def run_service(self, service, pieces: int = 2, segment_times: bool = False):
        work = Path(self.tmp.name) / "work"
        work.mkdir(exist_ok=True)
        return minimax.transcribe(KEY, Path("audio.webm"), work, segment_times=segment_times,
                                  transport=service, sleep=lambda s: None,
                                  cutter=fake_cutter(pieces),
                                  seconds=lambda p: self.fail("probe needed"))

    def test_words_come_from_json_mode_on_the_whole_videos_clock(self) -> None:
        service = FakeService([(200, text_reply("One. Two.", 420.0)),
                               (200, text_reply("Three.", 100.5))])
        segments = self.run_service(service)
        self.assertEqual(segments, [Segment(0, 420_000, "One. Two."),
                                    Segment(420_000, 520_500, "Three.")])
        self.assertEqual(service.formats, ["json", "json"])
        self.assertEqual(service.calls[0]["Authorization"], f"Bearer {KEY}")

    def test_segment_times_keep_json_words_on_verbose_times(self) -> None:
        # verbose_json words differ ("I am" for "I'm"); only its times are used.
        service = FakeService([
            (200, text_reply("I'm here. Then more words follow.", 60.0)),
            (200, piece_reply("I am here. Then more words follow.", 60.0,
                              [(0, 10, "I am here."), (20, 60, "Then more words follow.")])),
        ])
        segments = self.run_service(service, pieces=1, segment_times=True)
        self.assertEqual(service.formats, ["json", "verbose_json"])
        self.assertEqual(" ".join(s.text for s in segments), "I'm here. Then more words follow.")
        self.assertEqual([(s.start_ms, s.end_ms) for s in segments], [(0, 10_000), (20_000, 60_000)])

    def test_pieces_are_cut_mono_at_16_khz_under_the_cap(self) -> None:
        seen: list[list[str]] = []
        directory = Path(self.tmp.name) / "pieces"

        def fake_run(command, **kwargs):
            seen.append(command)
            if command[0] == "ffmpeg":
                (directory / "piece-000.mp3").write_bytes(b"mp3")
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 0, "300.0\n", "")
        original = minimax.subprocess.run
        minimax.subprocess.run = fake_run
        try:
            pieces = minimax.cut_pieces(Path("audio.webm"), directory)
        finally:
            minimax.subprocess.run = original
        ffmpeg = seen[0]
        self.assertEqual(ffmpeg[ffmpeg.index("-ar") + 1], "16000")
        self.assertEqual(ffmpeg[ffmpeg.index("-ac") + 1], "1")
        self.assertEqual(ffmpeg[ffmpeg.index("-segment_time") + 1], "420")
        self.assertEqual([p.name for p in pieces], ["piece-000.mp3"])

    def test_saved_pieces_are_not_sent_again(self) -> None:
        self.run_service(FakeService([(200, text_reply("A.", 10))] * 2))
        again = FakeService([])
        self.assertEqual(len(self.run_service(again)), 2)
        self.assertEqual(again.calls, [])

    def test_resolver_failures_and_server_errors_are_retried(self) -> None:
        service = FakeService([urllib.error.URLError("Temporary failure in name resolution"),
                               (502, "bad gateway"),
                               (200, text_reply("A.", 10))])
        self.assertEqual(len(self.run_service(service, pieces=1)), 1)
        self.assertEqual(len(service.calls), 3)

    def test_a_plan_limit_stops_without_retrying_and_without_the_key(self) -> None:
        limit = json.dumps({"base_resp": {"status_code": 2056, "status_msg": "usage limit exceeded"}})
        service = FakeService([(200, text_reply("A.", 10)), (200, limit)])
        with self.assertRaises(minimax.MiniMaxLimit) as caught:
            self.run_service(service)
        self.assertNotIn(KEY, str(caught.exception))
        self.assertEqual(len(service.calls), 2)
        # The first piece is kept for the rerun.
        self.assertTrue((Path(self.tmp.name) / "work" / "pieces-16000" / "piece-000.text.json").is_file())


class FetchTests(TempHome):
    def make_runner(self, failures: dict[str, int] | None = None):
        failures = dict(failures or {})
        calls: list[list[str]] = []

        def runner(command: list[str]) -> subprocess.CompletedProcess:
            calls.append(command)
            out = Path(command[command.index("-o") + 1]).parent if "-o" in command else None
            kind = ("auto" if "--write-auto-subs" in command else
                    "subs" if "--write-subs" in command else "audio")
            if failures.get(kind):
                failures[kind] -= 1
                return subprocess.CompletedProcess(command, 1, "", "ERROR: HTTP Error 429: Too Many Requests")
            if kind == "subs":
                (out / "audio.en-US.json3").write_text(UPLOADER_TRACK)
            elif kind == "auto":
                (out / "audio.en-orig.json3").write_text(AUTO_TRACK)
            else:
                (out / "audio.webm").write_bytes(b"webm")
            return subprocess.CompletedProcess(command, 0, "", "")
        return runner, calls

    def test_captions_and_audio_are_separate_requests_without_translations(self) -> None:
        runner, calls = self.make_runner()
        run = paths.run_directory(self.root, VID)
        fetched = fetch.Fetcher("yt-dlp", runner=runner, sleep=lambda s: None).fetch(VID, run)
        self.assertEqual(len(calls), 3)
        auto = next(c for c in calls if "--write-auto-subs" in c)
        self.assertEqual(auto[auto.index("--sub-langs") + 1], "en-orig")
        self.assertNotIn("--write-subs", auto)
        self.assertEqual(fetched.audio.name, "audio.webm")
        self.assertEqual(fetched.uploader_captions.name, "audio.en-US.json3")
        for path in run.iterdir():
            if path.is_file():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path)

    def test_an_info_json_beside_the_audio_is_not_taken_for_it(self) -> None:
        runner, calls = self.make_runner()
        run = paths.run_directory(self.root, VID)
        (run / "audio.info.json").write_text("{}")
        fetched = fetch.Fetcher("yt-dlp", runner=runner, sleep=lambda s: None).fetch(VID, run)
        self.assertEqual(fetched.audio.name, "audio.webm")

    def test_a_second_fetch_downloads_nothing(self) -> None:
        runner, calls = self.make_runner()
        run = paths.run_directory(self.root, VID)
        fetcher = fetch.Fetcher("yt-dlp", runner=runner, sleep=lambda s: None)
        fetcher.fetch(VID, run)
        fetcher.fetch(VID, run)
        self.assertEqual(len(calls), 3)

    def test_rate_limits_are_waited_out(self) -> None:
        runner, calls = self.make_runner({"auto": 2})
        waits: list[float] = []
        run = paths.run_directory(self.root, VID)
        fetch.Fetcher("yt-dlp", runner=runner, sleep=waits.append).fetch(VID, run)
        self.assertEqual(waits, [20.0, 40.0])

    def test_cookies_are_passed_only_when_asked_for(self) -> None:
        runner, calls = self.make_runner()
        run = paths.run_directory(self.root, VID)
        fetch.Fetcher("yt-dlp", runner=runner, cookies_from_browser="chrome").fetch(VID, run)
        self.assertTrue(all("--cookies-from-browser" in c for c in calls))
        runner, calls = self.make_runner()
        fetch.Fetcher("yt-dlp", runner=runner).fetch(VID, paths.run_directory(self.root, "x" * 11))
        self.assertFalse(any("--cookies-from-browser" in c for c in calls))


class ScoreTests(unittest.TestCase):
    def test_style_differences_do_not_count_when_normalized(self) -> None:
        self.assertEqual(score.score("data sets, etc. 5 models", "datasets et cetera five models").wer, 0.0)
        self.assertGreater(score.score("data sets 5", "datasets five", normalized=False).wer, 0.0)

    def test_errors_are_counted_by_kind(self) -> None:
        measured = score.score("the cat sat down", "the cat sat on the mat")
        self.assertEqual(measured.reference_words, 6)
        self.assertEqual(measured.substitutions + measured.insertions + measured.deletions, 3)


class EndToEndTests(TempHome):
    def setUp(self) -> None:
        super().setUp()
        self.runner, self.calls = FetchTests.make_runner(self)

    def factory(self, cookies):
        return fetch.Fetcher("yt-dlp", runner=self.runner, sleep=lambda s: None,
                             cookies_from_browser=cookies)

    def invoke(self, *argv: str, service=None) -> tuple[int, str, str]:
        service = service or FakeService([(200, text_reply("Okay, so Qwen models are here.", 5.0))])
        out, err = io.StringIO(), io.StringIO()
        original = minimax.cut_pieces
        minimax.cut_pieces = fake_cutter(1)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(list(argv), fetcher_factory=self.factory, transport=service,
                                environ=self.env)
        finally:
            minimax.cut_pieces = original
        return code, out.getvalue(), err.getvalue()

    def test_srt_from_a_url_with_youtube_word_timing(self) -> None:
        self.write_key()
        code, out, err = self.invoke(f"https://youtu.be/{VID}", "--format", "srt", "--score")
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "1\n00:00:00,000 --> 00:00:05,000\nOkay, so Qwen models are here.\n")
        self.assertIn("WER against audio.en-US.json3: 0.00% normalized", err)
        record = json.loads((self.root / "runs" / VID / "result.json").read_text())
        self.assertEqual(record["timing"], "youtube-auto-captions")
        self.assertEqual(record["wer_normalized"], 0.0)
        self.assertNotIn(KEY, out + err + json.dumps(record))

    def test_with_captions_only_json_mode_is_requested(self) -> None:
        self.write_key()
        service = FakeService([(200, text_reply("Okay, so Qwen models are here.", 5.0))])
        code, out, err = self.invoke(VID, service=service)
        self.assertEqual(code, 0, err)
        self.assertEqual(service.formats, ["json"])

    def test_without_captions_minimax_times_the_segments(self) -> None:
        self.write_key()
        plain = self.runner

        def runner(command):
            result = plain(command)
            for track in Path(command[command.index("-o") + 1]).parent.glob("audio.en-orig.json3"):
                track.unlink()
            return result
        self.runner = runner
        service = FakeService([(200, text_reply("Hello there.", 4.0)),
                               (200, piece_reply("Hello there.", 4.0, [(1.0, 3.0, "Hello there.")]))])
        code, out, err = self.invoke(VID, "--format", "srt", service=service)
        self.assertEqual(code, 0, err)
        self.assertEqual(service.formats, ["json", "verbose_json"])
        self.assertIn("asking MiniMax for segment times", err)
        self.assertTrue(out.startswith("1\n00:00:01,000 --> 00:00:03,000\n"), out)

    def test_a_missing_key_stops_before_any_download(self) -> None:
        code, out, err = self.invoke(VID)
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("no MiniMax key", err)
        self.assertEqual(self.calls, [])

    def test_a_media_tool_failure_is_a_message_not_a_traceback(self) -> None:
        self.write_key()
        out, err = io.StringIO(), io.StringIO()

        def broken(audio, directory):
            raise subprocess.CalledProcessError(183, ["ffmpeg", "-i", str(audio)])
        original = minimax.cut_pieces
        minimax.cut_pieces = broken
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main([VID], fetcher_factory=self.factory, transport=FakeService([]),
                                environ=self.env)
        finally:
            minimax.cut_pieces = original
        self.assertEqual(code, cli.EXIT_FAILED)
        self.assertIn("ffmpeg failed with exit status 183", err.getvalue())

    def test_a_service_limit_exits_unavailable(self) -> None:
        self.write_key()
        limit = json.dumps({"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}})
        code, out, err = self.invoke(VID, service=FakeService([(200, limit)]))
        self.assertEqual(code, cli.EXIT_UNAVAILABLE)
        self.assertIn("a rerun resumes", err)
        self.assertNotIn(KEY, out + err)


if __name__ == "__main__":
    unittest.main()
