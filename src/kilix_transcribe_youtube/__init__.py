"""Transcribe a YouTube video with the method kilix-transcribe selected for it.

The selected method is MiniMax asr-1.0 for the words, with each word timed from
YouTube's own automatic caption track. It scored the lowest word-error rate of
every method tried on three reference videos (see docs/youtube.md).
"""

SCHEMA = "kilix.transcribe.youtube/v1"
ENGINE_ID = "minimax"
MODEL_ID = "asr-1.0"
