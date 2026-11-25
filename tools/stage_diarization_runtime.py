#!/usr/bin/env python3
"""Stage exact local CPU diarization bytes; never fetch models or mint receipts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kilix_transcribe.diarization import (DiarizationRuntime, SCHEMA, ALGORITHM, SHERPA_REF,
    SEGMENTATION_REF, SEGMENTATION_SHAS, CHECKPOINT_SHA)
from kilix_transcribe.runtime import digest_file

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--segmentation', type=Path, required=True)
    p.add_argument('--embedding', type=Path, required=True)
    p.add_argument('--embedding-sha256', required=True)
    p.add_argument('--asset-id', default='kilix-diarization-cpu')
    p.add_argument('--revision', required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    native = json.loads((args.native/'native.json').read_text())
    if (native.get('schema') != 'kilix.transcribe.diarization-native/v1'
            or native['inputs']['sherpa_revision'] != SHERPA_REF
            or native['inputs']['ort_version'] != '1.27.1'):
        p.error('native source/backend differs from the selected route')
    segmentation_sha = digest_file(args.segmentation)
    if segmentation_sha not in SEGMENTATION_SHAS or digest_file(args.embedding) != args.embedding_sha256:
        p.error('models do not match their selected exact digests')
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink(): p.error('destination must be new')
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.diarization-stage-', dir=destination.parent) as temporary:
        root = Path(temporary)/'runtime';root.mkdir(mode=0o700)
        files = {}
        for name, sha in native['files'].items():
            if name != 'diarize-runner' and not name.startswith('lib/'): continue
            source = args.native/name
            if digest_file(source) != sha: p.error('native population changed')
            target = root/name;target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source,target);target.chmod(0o700 if name=='diarize-runner' else 0o600)
            files[name] = digest_file(target)
        for source,name,sha in ((args.segmentation,'segmentation.onnx',segmentation_sha),
                                (args.embedding,'embedding.onnx',args.embedding_sha256)):
            shutil.copyfile(source,root/name);(root/name).chmod(0o600)
            if digest_file(root/name) != sha: p.error('model changed while staging')
            files[name]=sha
        manifest = {'schema':SCHEMA,'algorithm':ALGORITHM,'asset_id':args.asset_id,'revision':args.revision,
            'engine_revision':SHERPA_REF,'segmentation_revision':SEGMENTATION_REF,
            'embedding_checkpoint_sha256':CHECKPOINT_SHA,'files':files,'qualified':False}
        (root/'diarization.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (root/'diarization.json').chmod(0o600)
        DiarizationRuntime(root)
        destination.mkdir(mode=0o700)
        for source in root.iterdir(): os.rename(source,destination/source.name)
    print('Explicit development diarization staged; installed receipts and qualification are separate.')

if __name__ == '__main__': main()
