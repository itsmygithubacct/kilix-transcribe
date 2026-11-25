# kilix-transcribe-youtube

`kilix-transcribe-youtube` transcribes a YouTube video and prints the
transcript as text, JSON, WebVTT or SRT. It uses the method that was most
accurate on three reference videos: MiniMax `asr-1.0` supplies the words, and
YouTube's own automatic caption track supplies the word timing.

```sh
kilix-transcribe-youtube https://www.youtube.com/watch?v=zaLQ0AnY9dI --format srt --output talk.srt
kilix-transcribe-youtube zaLQ0AnY9dI --score            # also report accuracy against the uploader track
kilix-transcribe-youtube search "using jev in your agent harness"
```

`VIDEO` is an 11-character video id or a `youtube.com`, `youtu.be`,
`/shorts/` or `/embed/` URL. `--cookies-from-browser chrome` passes a browser
profile to yt-dlp for videos that need a signed-in viewer.

## Requirements

- A current `yt-dlp` at `~/.local/bin/yt-dlp`, on `PATH`, or named by
  `KILIX_YT_DLP`. Distribution packages are often too old for YouTube.
- `ffmpeg` and `ffprobe`.
- A MiniMax API key in the private data directory (below).

## Private data

Everything personal stays in one directory readable only by its owner:
`$XDG_DATA_HOME/kilix-transcribe-youtube`, which is
`~/.local/share/kilix-transcribe-youtube` by default
(`KILIX_TRANSCRIBE_YOUTUBE_HOME` overrides it).

```sh
install -d -m 700 ~/.local/share/kilix-transcribe-youtube
install -m 600 /path/to/minimax-key ~/.local/share/kilix-transcribe-youtube/minimax_api_key
```

The key file must be a regular file owned by you with no group or other
permissions, or the command refuses it. The key is sent only in the MiniMax
request header and is never printed, logged or written anywhere else.

Each video gets `runs/<video-id>/` (mode 700, files mode 600): the audio, the
caption tracks, MiniMax's reply for every piece, and `result.json`, a short
record of the method and, with `--score`, the measured word-error rate. A
second run of the same video reuses all of it and makes no new requests. A run
stopped by a MiniMax plan limit (exit status 69) keeps every finished piece and
resumes from the next one.

## How it works

1. yt-dlp downloads the uploader's English caption track, if there is one, and
   YouTube's original-language automatic track (`en-orig`) in two separate
   requests with a pause between subtitle downloads. Asking for several tracks
   in one burst draws HTTP 429, and asking for `en` with automatic captions
   selects a machine translation. Rate limits are waited out and retried.
2. The audio is cut into mono 16 kHz MP3 pieces of 7 minutes, under MiniMax's
   500-second cap. 16 kHz is deliberate: the same video scored 1.00% word
   errors cut at YouTube's 48 kHz and 0.66% at 16 kHz.
3. Each piece goes to MiniMax with `response_format=json`. `verbose_json` adds
   segment times, but its words are different and worse: it writes "I am"
   where the speaker and the uploader's transcript say "I'm". Each mode returns
   the same words on every call, so this is the mode, not chance.
4. MiniMax's words are aligned with the automatic track's word-timed words.
   Words the two agree on take YouTube's timing, and the words between two
   agreements are spread evenly across the gap. A match more than 15 seconds
   outside its own piece is ignored. If a video has no automatic track,
   MiniMax is also asked for `verbose_json`, and its segment times are used.
5. Words are grouped into cues of at most 84 characters (two subtitle lines)
   and 7 seconds, breaking at sentence ends and pauses, and rendered by
   `kilix_transcribe.surface.render_transcript`.

## Why this method

Every available method was tried on three videos with a word-for-word uploader
transcript as the reference. *Normalized* word-error rate ignores spelling
style ("5" and "five", "etc" and "et cetera", "data sets" and "datasets");
`kilix_transcribe_youtube.score` implements it.

| Method | AREX-2 (16:28) | Jev & OpenJevs (19:19) | Jev in your harness (20:25) |
|---|---|---|---|
| **MiniMax asr-1.0** | **0.5%** | **1.3%** | **0.6%** |
| YouTube automatic captions (`en-orig`) | 1.5% | 1.6% | 1.2% |
| Whisper medium.en (local) | 1.7% | 1.9% | 2.6% |
| Whisper small.en (local) | 1.9% | 2.5% | 1.5% |
| Vosk lgraph-en-us (local) | 11.4% | not run | not run |
| Vosk small-en-us (local) | 13.4% | not run | not run |

Whisper medium's 2.6% on the third video comes from leaving out its final 67
words. Whistle could not finish a 2-minute piece within its time limit on the
test machine, the VibeVoice runtime was not built, and the Whisper tiny
runtime was not prepared, so none of the three was scored. The Vosk models were
dropped after the first video. The bench measured MiniMax with
`response_format=json` on 16 kHz audio, the settings the command uses; the
finished command, run on the same three videos with `--score`, measured 0.59%,
1.26% and 0.61%.

## Limits

- English only. Other languages would need their own reference videos before
  this method is trusted for them.
- The words come from a hosted service: the audio of each video is uploaded to
  MiniMax. Use the local `kilix-transcribe` engines for audio that must not
  leave the machine.
- Word times are YouTube's, so they are as good as YouTube's automatic track;
  a word the two transcripts disagree on is placed between its neighbours.

## Exit status

| Status | Meaning |
|---|---|
| 0 | transcript written |
| 1 | download, ffmpeg or MiniMax failure after retries |
| 2 | bad video id, or the key or data directory is missing or too open |
| 69 | MiniMax refused the request (plan, balance or key); finished pieces are kept |
