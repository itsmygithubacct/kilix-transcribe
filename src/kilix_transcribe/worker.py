"""Private one-job decoder/inference child; no microphone or network client."""

from __future__ import annotations

import json
from contextlib import nullcontext
import os
from pathlib import Path
import resource
import subprocess
import sys
import wave

# -I removes caller Python paths. This file is in the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kilix_transcribe.runtime import ENGINE_COMMIT, MAX_AUDIO_SECONDS, MAX_RESULT_BYTES
from kilix_transcribe.surface import SegmentUpdate, Transcript, WordTiming, render_transcript as serialize


def _run(command: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
    subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True, pass_fds=pass_fds)


def _words(tokens: list, start: int, end: int) -> tuple[WordTiming, ...]:
    words = []
    pending = None
    for token in tokens:
        text = token.get("text", "")
        offsets = token.get("offsets")
        if not text or text.startswith("[_") or text.startswith("<|") or not offsets:
            continue
        t0, t1 = int(offsets["from"]), int(offsets["to"])
        t0, t1 = max(start, min(t0, end)), max(start, min(t1, end))
        t1 = max(t0, t1)
        if pending is not None and (text[:1].isspace() or t0 > pending[1] + 500):
            words.append(WordTiming(*pending))
            pending = None
        if pending is None:
            pending = [t0, t1, text.strip()]
        else:
            pending[1] = max(pending[1], t1)
            pending[2] += text
    if pending is not None and pending[2]:
        words.append(WordTiming(*pending))
    return tuple(words)


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.umask(0o077)
    request = json.loads(sys.stdin.buffer.read(65_537))
    root = Path(request["runtime"])
    args = request["args"]
    audio_fd = request["audio_fd"]
    manifest = request["manifest"]
    demuxer = {"audio/wav": "wav", "audio/flac": "flac", "audio/ogg": "ogg"}[args["audio"]["media_type"]]
    # The supervisor owns cleanup, including after SIGKILL or decoder failure.
    with nullcontext(request["workspace"]) as temporary:
        directory = Path(temporary)
        decoded = directory / "audio.wav"
        _run([str(root / "ffmpeg"), "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
              "-f", demuxer, "-i", f"/proc/self/fd/{audio_fd}", "-map", "0:a:0",
              "-vn", "-sn", "-dn", "-t", str(MAX_AUDIO_SECONDS + 1), "-ac", "1",
              "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(decoded)],
             pass_fds=(audio_fd,))
        os.close(audio_fd)
        with wave.open(str(decoded), "rb") as audio:
            frames = audio.getnframes()
            if not 0 < frames <= MAX_AUDIO_SECONDS * 16000:
                raise ValueError("audio duration exceeds bound")
            silence = True
            while block := audio.readframes(16000):
                if any(block):
                    silence = False
                    break
        duration_ms = (frames * 1000 + 15999) // 16000
        model = manifest["model"]
        segments = []
        language = args.get("language") or "auto"
        if not silence:
            destination = directory / "transcript"
            command = [str(root / "whisper-cli"), "--model", str(root / "model.bin"),
                       "--file", str(decoded), "--language", language.split("-")[0],
                       "--output-json-full", "--output-file", str(destination),
                       "--threads", "2", "--no-gpu", "--print-progress"]
            if args["task"] == "translate":
                command.append("--translate")
            _run(command)
            payload = destination.with_suffix(".json").read_bytes()
            if len(payload) > MAX_RESULT_BYTES:
                raise ValueError("engine result exceeds bound")
            decoded_result = json.loads(payload)
            language = decoded_result["result"]["language"]
            for index, segment in enumerate(decoded_result["transcription"]):
                start, end = segment["offsets"]["from"], segment["offsets"]["to"]
                # The engine can round the last boundary past the audio end.
                start, end = min(start, duration_ms), min(end, duration_ms)
                text = segment["text"].strip()
                if not text:
                    continue
                segments.append(SegmentUpdate(index, 0, start, max(start, end), text, True,
                                               _words(segment.get("tokens", []), start, max(start, end))))
        transcript = Transcript(args["task"], "whisper.cpp", model["id"], language, tuple(segments))
        result = json.loads(serialize(transcript, "json"))
        result.update({"engine_revision": ENGINE_COMMIT, "model_revision": model["revision"],
                       "duration_ms": duration_ms, "text": serialize(transcript, "text"),
                       "output_format": args["output"], "output": serialize(transcript, args["output"])})
        payload = json.dumps(result, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()
        if len(payload) > MAX_RESULT_BYTES:
            raise ValueError("result exceeds bound")
        sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    main()
