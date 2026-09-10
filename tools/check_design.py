#!/usr/bin/env python3
"""Fail-closed checker for the PREP7 transcription design candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "design" / "transcribe-candidate-v1.json"

EXPECTED_COMMANDS = ["record", "file", "serve", "models", "status", "cancel", "unload"]
EXPECTED_TASKS = ["transcribe", "translate", "diarize"]
EXPECTED_OUTPUTS = ["text", "json", "webvtt", "srt"]
EXPECTED_STOPS = ["explicit_cancel", "client_disconnect", "deadline", "screen_lock", "provider_failure"]
EXPECTED_EXCLUSIONS = [
    "cloud_transcription", "always_listening_or_wake_words",
    "autonomous_terminal_execution", "meeting_surveillance",
    "unauthenticated_lan_service",
]
EXPECTED_TRANSPORT = {
    "family": "AF_UNIX",
    "type": "SOCK_SEQPACKET",
    "peer_identity": "SO_PEERCRED_EFFECTIVE_UID",
    "runtime_directory_mode": "0700",
    "socket_mode": "0600",
    "control_frame_bytes": 65536,
    "audio_transfer": "SCM_RIGHTS_DESCRIPTOR",
}
EXPECTED_DECODER_LIMITS = {
    "input_bytes": 2147483648,
    "duration_ms": 14400000,
    "sample_rate_hz": 384000,
    "channels": 32,
    "address_space_bytes": 2147483648,
    "cpu_threads": 4,
}
EXPECTED_SOURCE_OBJECTS = [
    {
        "id": "whisper.cpp",
        "commit": "371b5a7561823ab2bb32142d2751e35e7534727b",
        "tree": "3d7ce4f956997cfa325c7556533aba5604278463",
        "route": "whisper-cpp",
        "selected": False,
    },
    {
        "id": "openai-whisper",
        "commit": "31243bad24cc746f07d4c8bfdd2d974872cb1803",
        "tree": "061ecdd5038d3c824f714bc77b272ad10c1b1dad",
        "route": "openai-pytorch",
        "selected": False,
    },
    {
        "id": "faster-whisper",
        "commit": "65882eee9f5cdbeeb2d877f1131d48cf241b327d",
        "tree": "7f396ce8d3316df36f674183aea9ff00ff946637",
        "route": "faster-whisper",
        "selected": False,
        "required_repair": "THIRD_PARTY_NOTICE",
    },
    {
        "id": "ctranslate2",
        "commit": "0d8bcd362ac75ef860ef161d6f0efad0ae439ff0",
        "tree": "3f2df7ccdec126f6d180367a9906c21221105a26",
        "route": "faster-whisper",
        "selected": False,
    },
    {
        "id": "sherpa-onnx",
        "commit": "1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911",
        "tree": "58c9772e226857cacaa16038bd241e3c3ca77f9a",
        "route": "sherpa-diarization",
        "selected": False,
    },
    {
        "id": "3d-speaker",
        "commit": "065629c313eaf1a01c65c640c46d77e61e9607b4",
        "tree": "fc152f91be157ef32cc52009b4e147dccbcf2a6a",
        "route": "sherpa-diarization",
        "selected": False,
        "required_repair": "REPRODUCIBLE_LOCAL_CAMPP_EXPORT",
    },
    {
        "id": "campplus-model",
        "commit": "032b8131a7ad812f87061955ca974c99060c5a03",
        "tree": "e7a3d915a1b6e7d8ea1eaba68d8472874e1a6145",
        "route": "sherpa-diarization",
        "selected": False,
        "payload_sha256": "5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de",
    },
    {
        "id": "pyannote-segmentation-onnx-model",
        "commit": "9403a6902bb58e3d5ae8c7e77c3422de279db2e0",
        "tree": "5d65ac109661b27dd3daaa16a0e71b2df20b0ea4",
        "route": "sherpa-diarization",
        "selected": False,
        "payload_sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    },
]
EXPECTED_ROUTES = [
    {"id": "whisper-cpp", "selected": False, "runtime_pickle": False},
    {
        "id": "openai-pytorch",
        "selected": False,
        "runtime_pickle": False,
        "note": "Direct pt checkpoint loading remains prohibited",
    },
    {
        "id": "faster-whisper",
        "selected": False,
        "runtime_pickle": False,
        "required_repair": "THIRD_PARTY_NOTICE",
    },
    {
        "id": "sherpa-diarization",
        "selected": False,
        "runtime_pickle": False,
        "required_repair": "REPRODUCIBLE_LOCAL_CAMPP_EXPORT",
    },
]
EXPECTED_FORBIDDEN_FIELDS = [
    "path", "url", "command", "shell", "executable", "environment", "module", "import",
]
EXPECTED_ROOT_KEYS = (
    "schema",
    "status",
    "commands",
    "tasks",
    "outputs",
    "transport",
    "decoder_limits",
    "source_objects",
    "engine_routes",
    "capture_stop_conditions",
    "forbidden_request_fields",
    "excluded_capabilities",
    "retention",
    "requirements",
)
EXPECTED_CATEGORY_COUNTS = {
    "boundary": 6,
    "transport": 6,
    "cli": 7,
    "input": 7,
    "output": 8,
    "engine": 6,
    "lifecycle": 4,
    "privacy": 4,
}
EXPECTED_REQUIREMENTS_SHA256 = "9935d138777fe35e24c8ee16d0dcdf595f0cca211b0590d460c44d824028af88"


class DesignError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def load_design() -> dict[str, Any]:
    with DESIGN_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise DesignError(code, message)


def validate_design(design: dict[str, Any]) -> None:
    require(type(design) is dict, "INVALID_DESIGN", "design root must be an object")
    extra = set(design) - set(EXPECTED_ROOT_KEYS)
    require(not extra, "UNKNOWN_ROOT_KEY",
            "design root contains unknown keys: " + ", ".join(sorted(extra)))
    require(design.get("schema") == "kilix.transcribe.design/candidate-v1",
            "SCHEMA", "design schema identity drifted")
    require(design.get("status") == "PREP7_DESIGN_CANDIDATE_NOT_SELECTED",
            "STATUS", "candidate must not claim selection or acceptance")
    require(design.get("commands") == EXPECTED_COMMANDS,
            "COMMANDS", "command population differs from 7/7")
    require(design.get("tasks") == EXPECTED_TASKS,
            "TASKS", "task population differs from 3/3")
    require(design.get("outputs") == EXPECTED_OUTPUTS,
            "OUTPUTS", "output population differs from 4/4")
    require(design.get("capture_stop_conditions") == EXPECTED_STOPS,
            "CAPTURE_STOPS", "capture stop population differs from 5/5")
    require(design.get("excluded_capabilities") == EXPECTED_EXCLUSIONS,
            "EXCLUSIONS", "exclusion population differs from 5/5")

    transport = design.get("transport", {})
    require(transport == EXPECTED_TRANSPORT,
            "TRANSPORT", "private descriptor transport boundary drifted")

    limits = design.get("decoder_limits", {})
    require(limits == EXPECTED_DECODER_LIMITS, "DECODER_LIMITS",
            "decoder limit population or value drifted")

    sources = design.get("source_objects", [])
    require(len(sources) == 8, "SOURCE_POPULATION",
            "source population differs from 8/8")
    require([row.get("id") for row in sources]
            == [row["id"] for row in EXPECTED_SOURCE_OBJECTS],
            "SOURCE_POPULATION", "source identities differ from accepted 8/8")
    require(all(row.get("selected") is False for row in sources),
            "PREMATURE_SELECTION", "one or more of 8/8 source objects is selected")
    require(sources == EXPECTED_SOURCE_OBJECTS, "SOURCE_IDENTITY",
            "one or more accepted source object values drifted")

    routes = design.get("engine_routes", [])
    require(len(routes) == 4, "ROUTE_POPULATION",
            "architecture route population differs from 4/4")
    require([row.get("id") for row in routes]
            == [row["id"] for row in EXPECTED_ROUTES],
            "ROUTE_POPULATION", "architecture route identities drifted")
    require(all(row.get("selected") is False for row in routes),
            "PREMATURE_SELECTION", "one or more of 4/4 routes is selected")
    require(all(row.get("runtime_pickle") is False for row in routes),
            "RUNTIME_PICKLE", "runtime pickle was admitted")
    repairs = {row.get("required_repair") for row in routes if row.get("required_repair")}
    require(repairs == {"THIRD_PARTY_NOTICE", "REPRODUCIBLE_LOCAL_CAMPP_EXPORT"},
            "REPAIRS", "required repair population differs from 2/2")
    require(routes == EXPECTED_ROUTES, "ROUTE_IDENTITY",
            "one or more architecture route values drifted")

    retention = design.get("retention", {})
    require(retention == {
        "transcripts_by_default": False,
        "audio_by_default": False,
        "logs_contain_content": False,
    }, "RETENTION", "private-by-default retention boundary drifted")

    forbidden = design.get("forbidden_request_fields", [])
    require(forbidden == EXPECTED_FORBIDDEN_FIELDS,
            "FORBIDDEN_FIELDS", "forbidden request-field population differs from 8/8")

    requirements = design.get("requirements", [])
    require(len(requirements) == 48, "REQUIREMENT_POPULATION",
            "requirement population differs from 48/48")
    expected_ids = [f"TR-{number:03d}" for number in range(1, 49)]
    require([row.get("id") for row in requirements] == expected_ids,
            "REQUIREMENT_IDS", "requirement IDs are not the exact TR-001..TR-048 sequence")
    counts = {category: 0 for category in EXPECTED_CATEGORY_COUNTS}
    for row in requirements:
        category = row.get("category")
        require(category in counts, "REQUIREMENT_CATEGORY",
                f"unknown requirement category {category!r}")
        counts[category] += 1
        require(isinstance(row.get("requirement"), str) and row["requirement"].strip(),
                "REQUIREMENT_TEXT", f"{row.get('id')} has no requirement text")
        require(row.get("evidence_phase") in {"P1", "P2", "P3", "P5", "P6", "P7", "P8"},
                "REQUIREMENT_PHASE", f"{row.get('id')} has an invalid evidence phase")
    require(counts == EXPECTED_CATEGORY_COUNTS, "REQUIREMENT_DENOMINATORS",
            "category denominators do not sum to the frozen 48/48 population")
    requirements_bytes = json.dumps(
        requirements, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    require(hashlib.sha256(requirements_bytes).hexdigest()
            == EXPECTED_REQUIREMENTS_SHA256,
            "REQUIREMENT_IDENTITY", "one or more requirement values drifted")


def negative_controls(design: dict[str, Any]) -> None:
    mutations = []

    selected = copy.deepcopy(design)
    selected["engine_routes"][0]["selected"] = True
    mutations.append(("premature-selection", selected, "PREMATURE_SELECTION"))

    missing_command = copy.deepcopy(design)
    missing_command["commands"].remove("cancel")
    mutations.append(("missing-cancel", missing_command, "COMMANDS"))

    pickle = copy.deepcopy(design)
    pickle["engine_routes"][1]["runtime_pickle"] = True
    mutations.append(("runtime-pickle", pickle, "RUNTIME_PICKLE"))

    retained = copy.deepcopy(design)
    retained["retention"]["transcripts_by_default"] = True
    mutations.append(("default-retention", retained, "RETENTION"))

    schema = copy.deepcopy(design)
    schema["schema"] = "kilix.transcribe.design/drifted"
    mutations.append(("schema-drift", schema, "SCHEMA"))

    status = copy.deepcopy(design)
    status["status"] = "SELECTED_FOR_RELEASE"
    mutations.append(("status-drift", status, "STATUS"))

    tasks = copy.deepcopy(design)
    tasks["tasks"].remove("diarize")
    mutations.append(("missing-diarize", tasks, "TASKS"))

    exclusions = copy.deepcopy(design)
    exclusions["excluded_capabilities"].remove("cloud_transcription")
    mutations.append(("missing-exclusion", exclusions, "EXCLUSIONS"))

    transport = copy.deepcopy(design)
    transport["transport"]["socket_mode"] = "0666"
    mutations.append(("open-socket-mode", transport, "TRANSPORT"))

    forbidden = copy.deepcopy(design)
    forbidden["forbidden_request_fields"].remove("shell")
    mutations.append(("missing-forbidden-field", forbidden, "FORBIDDEN_FIELDS"))

    route = copy.deepcopy(design)
    route["engine_routes"][0]["note"] = "unreviewed extra field"
    mutations.append(("route-extra-field", route, "ROUTE_IDENTITY"))

    extra_root = copy.deepcopy(design)
    extra_root["release_profile"] = {"engine": "whisper-cpp", "selected": True}
    mutations.append(("unknown-root-key", extra_root, "UNKNOWN_ROOT_KEY"))

    for name, mutant, expected in mutations:
        try:
            validate_design(mutant)
        except DesignError as error:
            require(error.code == expected, "WRONG_REASON",
                    f"{name} refused as {error.code}, expected {expected}")
        else:
            raise DesignError("MUTATION_ACCEPTED", f"{name} was accepted")


def run() -> None:
    design = load_design()
    validate_design(design)
    negative_controls(design)
    print(
        "TRANSCRIBE_DESIGN_CANDIDATE: PASS "
        "(48/48 requirements; 8/8 P0 source objects; 0/8 selected; "
        "4/4 architecture routes; 0/4 selected; 7/7 commands; "
        "4/4 outputs; 12/12 negative controls)"
    )


if __name__ == "__main__":
    try:
        run()
    except (DesignError, OSError, ValueError, json.JSONDecodeError) as error:
        code = error.code if isinstance(error, DesignError) else type(error).__name__
        print(f"TRANSCRIBE_DESIGN_CANDIDATE: FAIL [{code}] {error}", file=sys.stderr)
        raise SystemExit(1)
