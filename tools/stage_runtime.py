#!/usr/bin/env python3
"""Stage reviewed local bytes for development; not an F100 installer."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kilix_transcribe.runtime import ENGINE_COMMIT, RUNTIME_SCHEMA, InstalledRuntime, digest_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    for name in ("engine", "decoder"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--installed-asset")
    parser.add_argument("--content-root", type=Path)
    parser.add_argument("--model-snapshot-bytes", type=int)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    args.runtime_root = args.destination.absolute()
    from kilix_transcribe.content import from_options
    model_source = from_options(args, provider="kilix-transcribe",
                                consumer_schema="kilix.transcribe.runtime")
    if (model_source is None) == (args.model is None):
        parser.error("select either --model or the installed Content model options")
    if not re.fullmatch(r"[0-9a-f]{64}", args.model_sha256):
        parser.error("expected model digest must be lowercase SHA-256")
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink():
        parser.error("destination must not already exist")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transcribe-stage-", dir=destination.parent) as temporary:
        root = Path(temporary) / "runtime"
        root.mkdir(mode=0o700)
        files = {}
        sources = [("engine", "whisper-cli"), ("decoder", "ffmpeg")]
        if model_source is None:
            sources.append(("model", "model.bin"))
        for source_name, target_name in sources:
            source = getattr(args, source_name)
            expected = getattr(args, f"{source_name}_sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                parser.error("expected digest must be lowercase SHA-256")
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(descriptor, "rb") as reader:
                info = os.fstat(reader.fileno())
                if not stat.S_ISREG(info.st_mode):
                    parser.error("inputs must be regular files")
                with (root / target_name).open("xb") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
            (root / target_name).chmod(0o600 if source_name == "model" else 0o700)
            if digest_file(root / target_name) != expected:
                parser.error("staged file does not match its expected digest")
            files[target_name] = expected
        files["model.bin"] = args.model_sha256
        manifest = {"schema": RUNTIME_SCHEMA, "engine_revision": ENGINE_COMMIT,
                    "model": {"id": args.model_id, "revision": args.model_revision}, "files": files}
        (root / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (root / "runtime.json").chmod(0o600)
        with model_source if model_source is not None else nullcontext():
            InstalledRuntime(root, model_source=model_source)
            if model_source is not None:
                with model_source.open(lambda: None):
                    pass
        # mkdir reserves the final name without overwriting another staging
        # process; the private parent prevents readers observing partial files.
        destination.mkdir(mode=0o700)
        try:
            for source in root.iterdir():
                os.rename(source, destination / source.name)
        except BaseException:
            shutil.rmtree(destination)
            raise
    print("Development runtime staged; release qualification remains required.")


if __name__ == "__main__":
    main()
