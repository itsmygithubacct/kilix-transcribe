from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "check_design", ROOT / "tools" / "check_design.py"
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
design_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(design_module)


class TranscribeDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = design_module.load_design()

    def refusal(self, mutant, code: str) -> None:
        with self.assertRaises(design_module.DesignError) as caught:
            design_module.validate_design(mutant)
        self.assertEqual(caught.exception.code, code)

    def test_complete_design(self) -> None:
        design_module.validate_design(self.design)

    def test_complete_negative_control_population(self) -> None:
        design_module.negative_controls(self.design)

    def test_missing_capture_stop_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["capture_stop_conditions"].remove("screen_lock")
        self.refusal(mutant, "CAPTURE_STOPS")

    def test_missing_output_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["outputs"].remove("webvtt")
        self.refusal(mutant, "OUTPUTS")

    def test_missing_source_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["source_objects"].pop()
        self.refusal(mutant, "SOURCE_POPULATION")

    def test_missing_notice_repair_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        del mutant["engine_routes"][2]["required_repair"]
        self.refusal(mutant, "REPAIRS")

    def test_unbounded_decoder_field_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["decoder_limits"]["duration_ms"] = 0
        self.refusal(mutant, "DECODER_LIMITS")

    def test_requirement_reordering_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["requirements"][0], mutant["requirements"][1] = (
            mutant["requirements"][1], mutant["requirements"][0]
        )
        self.refusal(mutant, "REQUIREMENT_IDS")

    def test_source_commit_mutation_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["source_objects"][0]["commit"] = "0" * 40
        self.refusal(mutant, "SOURCE_IDENTITY")

    def test_source_payload_mutation_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["source_objects"][6]["payload_sha256"] = "0" * 64
        self.refusal(mutant, "SOURCE_IDENTITY")

    def test_duplicate_command_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["commands"].append("record")
        self.refusal(mutant, "COMMANDS")

    def test_requirement_text_mutation_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["requirements"][0]["requirement"] = "Drifted requirement"
        self.refusal(mutant, "REQUIREMENT_IDENTITY")

    def test_schema_drift_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["schema"] = "kilix.transcribe.design/drifted"
        self.refusal(mutant, "SCHEMA")

    def test_status_drift_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["status"] = "SELECTED_FOR_RELEASE"
        self.refusal(mutant, "STATUS")

    def test_missing_task_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["tasks"].remove("diarize")
        self.refusal(mutant, "TASKS")

    def test_missing_exclusion_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["excluded_capabilities"].remove("cloud_transcription")
        self.refusal(mutant, "EXCLUSIONS")

    def test_open_socket_mode_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["transport"]["socket_mode"] = "0666"
        self.refusal(mutant, "TRANSPORT")

    def test_missing_forbidden_field_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["forbidden_request_fields"].remove("shell")
        self.refusal(mutant, "FORBIDDEN_FIELDS")

    def test_route_identity_extra_field_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["engine_routes"][0]["note"] = "unreviewed extra field"
        self.refusal(mutant, "ROUTE_IDENTITY")

    def test_unknown_root_key_is_refused(self) -> None:
        mutant = copy.deepcopy(self.design)
        mutant["release_profile"] = {"engine": "whisper-cpp", "selected": True}
        self.refusal(mutant, "UNKNOWN_ROOT_KEY")


if __name__ == "__main__":
    unittest.main()
