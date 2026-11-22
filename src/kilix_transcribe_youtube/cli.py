"""kilix-transcribe-youtube: transcribe a YouTube video.

  kilix-transcribe-youtube VIDEO [--format text|json|webvtt|srt] [--output FILE]
                                 [--score] [--cookies-from-browser BROWSER]
  kilix-transcribe-youtube search QUERY [--count N]

VIDEO is a video id or a youtube.com / youtu.be URL. Audio, captions and the
service's replies are kept under the private data directory, so a second run
of the same video reuses them and spends nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kilix_transcribe.surface import render_transcript

from kilix_transcribe_youtube import ENGINE_ID, MODEL_ID, SCHEMA, align, captions, cues, minimax
from kilix_transcribe_youtube import paths, score
from kilix_transcribe_youtube.fetch import Fetcher, FetchError
from kilix_transcribe_youtube.video import VideoError, video_id

EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 69  # the service refused: plan, balance or key


def _stderr(line: str) -> None:
    print(f"kilix-transcribe-youtube: {line}", file=sys.stderr, flush=True)


def _search(argv: list[str], fetcher_factory) -> int:
    parser = argparse.ArgumentParser(prog="kilix-transcribe-youtube search")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--count", type=int, default=5, choices=range(1, 51), metavar="N")
    args = parser.parse_args(argv)
    for hit in fetcher_factory(None).search(" ".join(args.query), args.count):
        print(f"{hit.video_id}\t{hit.duration}\t{hit.channel}\t{hit.title}")
    return 0


def _transcribe(argv: list[str], fetcher_factory, transport, environ) -> int:
    parser = argparse.ArgumentParser(prog="kilix-transcribe-youtube", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--format", choices=("text", "json", "webvtt", "srt"), default="text")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score", action="store_true",
                        help="report the word-error rate against the uploader's captions, if any")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="pass a browser profile to yt-dlp (age-restricted videos)")
    args = parser.parse_args(argv)

    vid = video_id(args.video)
    root = paths.data_root(environ)
    key = paths.read_key(root)  # before any download: fail fast, print nothing of it
    run = paths.run_directory(root, vid)

    fetched = fetcher_factory(args.cookies_from_browser).fetch(vid, run)
    _stderr(f"{vid}: audio and captions ready")
    caption_words = captions.timed_words(fetched.auto_captions) if fetched.auto_captions else []
    if not caption_words:
        _stderr("no automatic caption track: asking MiniMax for segment times as well")
    work = paths.private_directory(run / "minimax")
    segments = minimax.transcribe(key, fetched.audio, work, segment_times=not caption_words,
                                  transport=transport, progress=_stderr)
    words = align.align(segments, caption_words)
    result = cues.transcript(words)
    rendered = render_transcript(result, args.format)

    text = " ".join(segment.text for segment in segments)
    record = {"schema": SCHEMA, "video_id": vid, "engine": ENGINE_ID, "model": MODEL_ID,
              "timing": "youtube-auto-captions" if caption_words else "segment-spread",
              "words": len(words), "cues": len(result.segments)}
    if args.score:
        if fetched.uploader_captions:
            for label, normalized in (("normalized", True), ("raw", False)):
                measured = score.score(text, captions.plain_text(fetched.uploader_captions),
                                       normalized=normalized)
                record[f"wer_{label}"] = round(measured.wer, 4)
            record["reference"] = fetched.uploader_captions.name
            _stderr(f"WER against {fetched.uploader_captions.name}: "
                    f"{record['wer_normalized']:.2%} normalized, {record['wer_raw']:.2%} raw")
        else:
            _stderr("no uploader caption track to score against")
    previous_path = run / "result.json"
    if not args.score and previous_path.is_file():
        # A run without --score keeps the score an earlier run measured.
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        for name in ("wer_normalized", "wer_raw", "reference"):
            if name in previous:
                record[name] = previous[name]
    (run / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (run / "result.json").chmod(0o600)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


def main(argv: list[str] | None = None, *, fetcher_factory=None,
         transport=minimax.urllib_transport, environ=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    factory = fetcher_factory or (lambda cookies: Fetcher(cookies_from_browser=cookies))
    try:
        if argv[:1] == ["search"]:
            return _search(argv[1:], factory)
        return _transcribe(argv, factory, transport, environ)
    except (VideoError, paths.PrivateDataError) as error:
        _stderr(str(error))
        return EXIT_USAGE
    except minimax.MiniMaxLimit as error:
        _stderr(f"{error}; pieces already transcribed are kept, so a rerun resumes")
        return EXIT_UNAVAILABLE
    except subprocess.CalledProcessError as error:
        _stderr(f"{Path(str(error.cmd[0])).name} failed with exit status {error.returncode}")
        return EXIT_FAILED
    except (minimax.MiniMaxError, FetchError, OSError) as error:
        _stderr(str(error))
        return EXIT_FAILED
