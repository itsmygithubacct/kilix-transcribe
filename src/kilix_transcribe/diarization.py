"""Explicit one-request speaker turns using the pinned owned CPU runtime."""
from __future__ import annotations
import array
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
import wave

from .protocol import ProtocolError, decode_payload
from .runtime import InstalledRuntime, private_directory, digest_file, memory_file
from .diarization_core.fd import SEALS, F_ADD_SEALS, Refusal, read_sealed, digest_fd
from .diarization_core.runtime_v2 import (ALGORITHM, MAX_META, MAX_SCORES, MAX_SAMPLES,
                                        strict_json, validate_native, asr_identity)

SCHEMA = 'kilix.transcribe.diarization-runtime/v1'
ANNOTATION_SCHEMA = 'kilix.transcribe.diarization/v1'
SHERPA_REF = '1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911'
SEGMENTATION_REF = '9403a6902bb58e3d5ae8c7e77c3422de279db2e0'
CHECKPOINT_SHA = '5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de'
SEGMENTATION_SHAS = {
    '220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079',
    'd582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d',
}
_SHA = re.compile(r'[0-9a-f]{64}')
_REF = re.compile(r'[0-9a-f]{40}')
_LIB = re.compile(r'lib/[A-Za-z0-9_.+-]+\.so(?:\.[0-9]+)*')

def _require(condition, message):
    if not condition:
        raise ProtocolError('INVALID_RUNTIME', message)

class DiarizationRuntime:
    """A server-selected immutable population; IPC cannot choose model paths.

    A managed selection supplies InstalledModel covering the complete native
    and graph population. Explicit development staging is separate and never
    establishes a qualified profile or an installed model receipt.
    """
    def __init__(self, root: Path, *, model_source=None):
        self.root = private_directory(root)
        manifest = self.root / 'diarization.json'
        info = manifest.lstat()
        _require(info.st_uid == os.geteuid() and not info.st_mode & 0o022
                 and info.st_size <= 65536 and manifest.is_file() and not manifest.is_symlink(),
                 'unsafe diarization manifest')
        payload = manifest.read_bytes()
        value = decode_payload(payload)
        fields = {'schema','algorithm','asset_id','revision','engine_revision','segmentation_revision',
                  'embedding_checkpoint_sha256','files','qualified'}
        _require(type(value) is dict and set(value) == fields and value['schema'] == SCHEMA
                 and value['algorithm'] == ALGORITHM and value['engine_revision'] == SHERPA_REF
                 and value['segmentation_revision'] == SEGMENTATION_REF
                 and value['embedding_checkpoint_sha256'] == CHECKPOINT_SHA
                 and value['qualified'] is False, 'unsupported diarization identity')
        _require(type(value['asset_id']) is str and 0 < len(value['asset_id']) <= 128
                 and type(value['revision']) is str and bool(_REF.fullmatch(value['revision'])),
                 'invalid diarization asset identity')
        files = value['files']
        required = {'diarize-runner','segmentation.onnx','embedding.onnx'}
        _require(type(files) is dict and required <= set(files) and len(files) <= 32
                 and all(name in required or _LIB.fullmatch(name) for name in files)
                 and all(type(sha) is str and _SHA.fullmatch(sha) for sha in files.values())
                 and files['segmentation.onnx'] in SEGMENTATION_SHAS,
                 'invalid diarization population')
        self.manifest = value
        self.manifest_sha256 = hashlib.sha256(payload).hexdigest()
        self.model_source = model_source
        # Reuse the existing trusted directory walk and sealed-copy mechanism.
        self._runtime = InstalledRuntime.__new__(InstalledRuntime)
        self._runtime.root = self.root
        self._runtime.model_source = None
        self._runtime.manifest = {'files':files}
        root_fd, self._runtime._directories = self._runtime._open_root()
        os.close(root_fd)
        if model_source is not None:
            model_source.bind(value['asset_id'], value['revision'],
                              dict(files, **{'diarization.json':self.manifest_sha256}))
        else:
            self.verify_unchanged()
            self._runtime._identities = {name:self._runtime._identity((self.root/name).lstat())
                                         for name in files}

    def verify_unchanged(self, check=lambda: None):
        check()
        _require(digest_file(self.root/'diarization.json', check) == self.manifest_sha256,
                 'diarization manifest changed; restart required')
        if self.model_source is None:
            total = 0
            for name, sha in self.manifest['files'].items():
                check()
                path = self.root / name
                info = path.lstat()
                total += info.st_size
                _require(info.st_uid == os.geteuid() and not info.st_mode & 0o022
                         and info.st_size > 0 and total <= 512*1024**2
                         and digest_file(path, check) == sha, 'diarization runtime changed or exceeds bound')

    @contextmanager
    def snapshots(self, check):
        self.verify_unchanged(check)
        if self.model_source is None:
            with self._runtime.snapshots(check) as descriptors:
                yield descriptors
        else:
            with self.model_source.open(check) as asset:
                descriptors = {}
                try:
                    for name in self.manifest['files']:
                        descriptors[name] = self.model_source.descriptor(asset, name, check, verify_bytes=True)
                    yield descriptors
                finally:
                    for fd in descriptors.values():
                        os.close(fd)

def source_identity(result):
    # Output rendering is checked independently. Keep the original recognized
    # words/timestamps/task/model intact when joining separate speaker turns.
    return asr_identity({k:v for k,v in result.items() if k not in {'output','diarization'}})

def validate_annotation(annotation, result):
    fields = {'schema','algorithm','session','status','duration_ms','source_sha256',
              'runtime_sha256','qualified','turns','unidentified_activity'}
    if (type(annotation) is not dict or set(annotation) != fields
            or annotation['schema'] != ANNOTATION_SCHEMA or annotation['algorithm'] != ALGORITHM
            or annotation['qualified'] is not False
            or annotation['status'] not in {'OK','SILENCE','PARTIAL_UNKNOWN','UNIDENTIFIABLE_ACTIVITY'}
            or type(annotation['session']) is not str
            or not re.fullmatch(r'[0-9a-f]{32}', annotation['session'])
            or annotation['duration_ms'] != result['duration_ms']
            or annotation['source_sha256'] != source_identity(result)
            or type(annotation['runtime_sha256']) is not str or not _SHA.fullmatch(annotation['runtime_sha256'])
            or type(annotation['turns']) is not list or len(annotation['turns']) > 16384
            or type(annotation['unidentified_activity']) is not list
            or len(annotation['unidentified_activity']) > 16384):
        raise ProtocolError('INVALID_RESPONSE', 'invalid explicit-session annotation')
    for population, known in ((annotation['turns'],True),(annotation['unidentified_activity'],False)):
        for row in population:
            expected = {'start_ms','end_ms','speaker','confidence'} if known else {'start_ms','end_ms'}
            if (type(row) is not dict or set(row) != expected
                    or type(row['start_ms']) is not int or type(row['end_ms']) is not int
                    or not 0 <= row['start_ms'] < row['end_ms'] <= result['duration_ms']):
                raise ProtocolError('INVALID_RESPONSE', 'invalid speaker turn bounds')
            if known and (row['confidence'] is not None or type(row['speaker']) is not str
                    or not re.fullmatch(annotation['session'] + r':speaker(?:[0-9]|1[0-5])', row['speaker'])):
                raise ProtocolError('INVALID_RESPONSE', 'invalid session speaker identity')
    return annotation

def run_diarization(decoded: Path, directory: Path, descriptors: dict, manifest: dict,
                    manifest_sha256: str, result: dict, *, deadline: float, guard=()):
    pcm = metadata = scores = None
    try:
        pcm, metadata, scores = memory_file(), memory_file(), memory_file()
        with wave.open(str(decoded), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1,2,16000):
                raise ValueError('unsupported normalized PCM')
            samples = audio.getnframes()
            if not 0 < samples <= MAX_SAMPLES:
                raise ValueError('PCM exceeds the diarization bound')
            while block := audio.readframes(16000):
                integers = array.array('h', block)
                if sys.byteorder != 'little': integers.byteswap()
                floats = array.array('f', (x/32768.0 for x in integers))
                if sys.byteorder != 'little': floats.byteswap()
                view = memoryview(floats).cast('B')
                while view:
                    count = os.write(pcm, view)
                    if count <= 0: raise ValueError('PCM staging stalled')
                    view = view[count:]
        fcntl.fcntl(pcm, F_ADD_SEALS, SEALS)
        native = directory / 'diarization-native'; native.mkdir(mode=0o700)
        (native / 'lib').mkdir(mode=0o700)
        for name, descriptor in descriptors.items():
            if name in {'embedding.onnx','segmentation.onnx'}: continue
            target = native / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as output:
                offset = 0
                while block := os.pread(descriptor, 1048576, offset):
                    output.write(block); offset += len(block)
            if digest_file(target) != manifest['files'][name]:
                raise ValueError('native snapshot changed')
            target.chmod(0o500 if name == 'diarize-runner' else 0o400)
        command = [str(native/'diarize-runner'), str(pcm), str(descriptors['segmentation.onnx']),
                   str(descriptors['embedding.onnx']), str(metadata), str(scores), str(os.getpid()), repr(deadline)]
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, pass_fds=(pcm,metadata,scores,*descriptors.values(),*guard), check=False)
        if completed.returncode:
            raise ValueError('native diarization failed')
        if os.fstat(metadata).st_size > MAX_META or os.fstat(scores).st_size > MAX_SCORES:
            raise ValueError('native diagnostics exceed bound')
        # The native producer seals both outputs, including F_SEAL_SEAL.
        # read_sealed/validate_native require those kernel seals; attempting
        # to add them again would reject the real already-sealed producer.
        record = validate_native(strict_json(read_sealed(metadata,MAX_META)), scores)
        if record['status'] == 'FAILURE' or record['samples'] != samples:
            raise ValueError('native diagnostics do not bind the waveform')
        session = uuid.uuid4().hex
        annotation = {'schema':ANNOTATION_SCHEMA,'algorithm':ALGORITHM,'session':session,
            'status':record['status'],'duration_ms':result['duration_ms'],
            'source_sha256':source_identity(result),'runtime_sha256':manifest_sha256,'qualified':False,
            'turns':[{'speaker':session+':speaker'+str(x['cluster']),'start_ms':x['start_ms'],
                      'end_ms':x['end_ms'],'confidence':None} for x in record['turns']],
            'unidentified_activity':[{'start_ms':x['start_ms'],'end_ms':x['end_ms']}
                for x in record['activity'] if x['cluster'] is None and x['start_ms'] < x['end_ms']]}
        return validate_annotation(annotation,result)
    except (OSError, ValueError, Refusal) as error:
        raise ProtocolError('ENGINE_FAILED','speaker diarization failed') from error
    finally:
        for descriptor in (pcm,metadata,scores):
            if descriptor is not None: os.close(descriptor)
