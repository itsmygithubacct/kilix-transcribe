"""Engine-neutral candidate mechanics for kilix-transcribe.

This package is intentionally not a provider runtime or a frozen wire contract.
"""

from .surface import (
    COMMANDS,
    OUTPUTS,
    PROVIDER_REFUSAL,
    RUNTIME_COMMANDS,
    SURFACE_SCHEMA,
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

__all__ = [
    "COMMANDS",
    "OUTPUTS",
    "PROVIDER_REFUSAL",
    "RUNTIME_COMMANDS",
    "SURFACE_SCHEMA",
    "JobLifecycle",
    "JobState",
    "RuntimeUnselected",
    "SegmentUpdate",
    "SurfaceError",
    "Transcript",
    "TranscriptAssembler",
    "WordTiming",
    "inspect_command",
    "render_transcript",
]
