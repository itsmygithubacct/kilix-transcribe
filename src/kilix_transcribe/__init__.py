"""Engine-neutral candidate mechanics for kilix-transcribe.

This package is intentionally not a selected engine runtime or a frozen P1
wire contract.  It includes bounded candidate transport and atomic-output
mechanics that remain useful whichever engine is selected later.
"""

from .output import AtomicTranscriptStore
from .protocol import (
    PROTOCOL_SCHEMA,
    ProtocolError,
    ProviderRequest,
    verify_request_descriptors,
)

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
    "AtomicTranscriptStore",
    "OUTPUTS",
    "PROVIDER_REFUSAL",
    "PROTOCOL_SCHEMA",
    "ProtocolError",
    "ProviderRequest",
    "verify_request_descriptors",
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
