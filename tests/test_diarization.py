"""Owned explicit-session integration with visibly inert native/model bytes."""
import concurrent.futures
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
import wave
from unittest.mock import patch

import test_runtime as fixtures
from test_streaming import until
from test_streaming_fixes import alive, handle
from kilix_transcribe import diarization
from kilix_transcribe.protocol import ProtocolError
from kilix_transcribe.service import client_request, request_value
from kilix_transcribe.owned import ExecutionPolicy

NATIVE = r'''#!/usr/bin/python3
import fcntl,json,os,pathlib,struct,sys,time,uuid
root=pathlib.Path(FIXTURE_ROOT)
kind=os.pread(int(sys.argv[3]),100,0).decode()
record={'pid':os.getpid(),'start_ticks':int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}
(root/('native-'+uuid.uuid4().hex+'.json')).write_text(json.dumps(record))
if kind=='wait':
    while True:time.sleep(.01)
value=json.loads(RECORD)
if kind=='invalid':value['activity']=[]
scores=b''.join(struct.pack('<7f',-10.,0.,-10.,-10.,-10.,-10.,-10.) for _ in range(589))
os.write(int(sys.argv[4]),json.dumps(value).encode())
os.write(int(sys.argv[5]),scores)
if kind!='unsealed':
    for index in (4,5):fcntl.fcntl(int(sys.argv[index]),1033,15)
'''

@contextmanager
def provider(kind='success'):
    fixture=fixtures.RuntimeTests('runTest');fixture.setUp()
    try:
        with wave.open(str(fixture.audio),'wb') as audio:
            audio.setparams((1,2,16000,0,'NONE','not compressed'))
            audio.writeframes(b'\x01\0'*6400)
        fixture.start()
        root=fixture.root/'diarization';root.mkdir(mode=0o700)
        record=(Path(__file__).parent/'fixtures/diarization-one.json').read_text()
        native=NATIVE.replace('FIXTURE_ROOT',repr(str(root))).replace('RECORD',repr(record))
        files={'diarize-runner':native.encode(),'segmentation.onnx':b'SYNTHETIC NOT A MODEL',
               'embedding.onnx':kind.encode()}
        hashes={}
        for name,data in files.items():
            (root/name).write_bytes(data);(root/name).chmod(0o700 if name=='diarize-runner' else 0o600)
            hashes[name]=hashlib.sha256(data).hexdigest()
        manifest={'schema':diarization.SCHEMA,'algorithm':diarization.ALGORITHM,
            'asset_id':'synthetic-diarization','revision':'a'*40,'engine_revision':diarization.SHERPA_REF,
            'segmentation_revision':diarization.SEGMENTATION_REF,
            'embedding_checkpoint_sha256':diarization.CHECKPOINT_SHA,'files':hashes,'qualified':False}
        (root/'diarization.json').write_text(json.dumps(manifest))
        with patch.object(diarization,'SEGMENTATION_SHAS',{hashes['segmentation.onnx']}):
            selected=diarization.DiarizationRuntime(root)
            fixture.service.runtime.diarization_runtime=selected
            fixture.service.execution_policy=ExecutionPolicy('cpu-development',namespace=str(fixture.root/'lease'))
            fixture.diarization_root=root
            yield fixture
    finally:
        fixture.stop()
        until(lambda:not fixture.service._jobs)
        for path in (fixture.root/'diarization').glob('native-*.json'):
            if alive(json.loads(path.read_text())):
                raise AssertionError('owned native process survived provider teardown')
        fixture.doCleanups()

def value(fixture, *, timeout=5, explicit=True):
    result=fixture.value(timeout=timeout,output='json')
    result['args'].update(task='diarize',diarization_session=explicit)
    return result

def request(fixture, **options):
    with fixture.audio.open('rb') as source:
        return client_request(fixture.ipc,value(fixture,**options),source.fileno())

class SessionTests(unittest.TestCase):
    def test_installed_receipt_covers_manifest_and_all_selected_diarization_bytes(self):
        from test_content import CONTENT_AVAILABLE,PackagedFixture
        if not CONTENT_AVAILABLE:self.skipTest('Content receipt API required')
        from kilix_transcribe.content import InstalledModel
        with provider() as fixture:
            payloads={n:(fixture.diarization_root/n).read_bytes() for n in
                      ('diarize-runner','segmentation.onnx','embedding.onnx','diarization.json')}
            consumer='kilix.transcribe.diarization-runtime'
            for accepted in (True,False):
                with self.subTest(receipt=accepted),PackagedFixture(payloads,
                        asset_id='synthetic-diarization',revision='a'*40,receipt=accepted,
                        consumer_schema=consumer) as installed:
                    source=InstalledModel(installed.asset_id,installed.data,
                        maximum_bytes=512*1024**2,provider='kilix-transcribe',consumer_schema=consumer)
                    with patch.object(diarization,'SEGMENTATION_SHAS',
                            {hashlib.sha256(payloads['segmentation.onnx']).hexdigest()}):
                        selected=diarization.DiarizationRuntime(installed.selected,model_source=source)
                    fixture.service.runtime.diarization_runtime=selected
                    if accepted:
                        self.assertEqual(request(fixture)['diarization']['status'],'OK')
                        (installed.selected/'diarization.json').write_bytes(b'changed')
                    with self.assertRaises(ProtocolError) as error:request(fixture)
                    self.assertEqual(error.exception.code,'INVALID_RUNTIME')

    def test_actual_owned_route_returns_turns_bound_to_unchanged_transcript(self):
        with provider() as fixture:
            result=request(fixture)
            annotation=result['diarization']
            self.assertEqual((annotation['status'],annotation['qualified']),('OK',False))
            self.assertEqual(annotation['source_sha256'],diarization.source_identity(result))
            self.assertEqual([(x['start_ms'],x['end_ms']) for x in annotation['turns']],[(0,400)])
            self.assertIsNone(annotation['turns'][0]['confidence'])
            self.assertEqual(json.loads(result['output'])['diarization'],annotation)
            self.assertEqual(result['text'],'Test transcript.\n')
            rows=client_request(fixture.ipc,request_value('models'))['models']
            self.assertIn('diarize',rows[0]['capabilities'])

    def test_each_request_uses_fresh_unlinkable_session_and_plain_jobs_keep_their_route(self):
        with provider() as fixture:
            first,second=request(fixture),request(fixture)
            self.assertNotEqual(first['diarization']['session'],second['diarization']['session'])
            self.assertNotEqual(first['diarization']['turns'][0]['speaker'],second['diarization']['turns'][0]['speaker'])
            self.assertEqual(first['segments'],second['segments'])
            with fixture.audio.open('rb') as source:
                plain=client_request(fixture.ipc,fixture.value(),source.fileno())
            self.assertNotIn('diarization',plain)
            self.assertEqual(len(list(fixture.diarization_root.glob('native-*.json'))),2)

    def test_declined_session_refuses_before_native_execution(self):
        with provider() as fixture:
            with self.assertRaises(ProtocolError) as error:request(fixture,explicit=False)
            self.assertEqual(error.exception.code,'INVALID_REQUEST')
            self.assertFalse(list(fixture.diarization_root.glob('native-*.json')))

    def test_native_inconsistent_known_identity_fails_without_commit(self):
        with provider('invalid') as fixture:
            with self.assertRaises(ProtocolError) as error:request(fixture)
            self.assertEqual(error.exception.code,'ENGINE_FAILED')
            self.assertFalse(client_request(fixture.ipc,request_value('status'))['worker_active'])

    def test_unsealed_native_outputs_cannot_be_committed_or_resealed_by_consumer(self):
        with provider('unsealed') as fixture:
            with self.assertRaises(ProtocolError) as error:request(fixture)
            self.assertEqual(error.exception.code,'ENGINE_FAILED')
            self.assertFalse(client_request(fixture.ipc,request_value('status'))['worker_active'])

    def test_changed_runtime_refuses_before_native_execution(self):
        with provider() as fixture:
            (fixture.diarization_root/'embedding.onnx').write_bytes(b'changed')
            with self.assertRaises(ProtocolError) as error:request(fixture)
            self.assertEqual(error.exception.code,'INVALID_RUNTIME')
            self.assertFalse(list(fixture.diarization_root.glob('native-*.json')))

    def test_cancel_and_deadline_reap_native_before_successor_grant(self):
        import kilix_device_lease
        for action in ('cancel','deadline'):
            with self.subTest(action=action),provider('wait') as fixture,concurrent.futures.ThreadPoolExecutor() as pool:
                future=pool.submit(request,fixture,timeout=1 if action=='deadline' else 5)
                until(lambda:bool(list(fixture.diarization_root.glob('native-*.json'))))
                record=json.loads(next(fixture.diarization_root.glob('native-*.json')).read_text())
                self.assertTrue(alive(record))
                if action=='cancel':client_request(fixture.ipc,request_value('cancel',job_id='test-job'))
                with self.assertRaises(ProtocolError) as error:future.result(7)
                self.assertEqual(error.exception.code,'CANCELED' if action=='cancel' else 'DEADLINE_EXCEEDED')
                until(lambda:not fixture.service._jobs)
                self.assertFalse(alive(record))
                grant=kilix_device_lease.acquire(job_id='successor',workload='tts-utterance',device='cpu-development',
                    deadline=time.monotonic()+1,namespace=str(fixture.root/'lease'))
                grant.release(cleanup_complete=True)

    def test_annotation_rejects_wrong_source_cross_session_and_invented_confidence(self):
        with provider() as fixture:
            result=request(fixture)
            for damage in ('source','session','confidence','time','qualification'):
                annotation=copy.deepcopy(result['diarization'])
                if damage=='source':annotation['source_sha256']='0'*64
                elif damage=='session':annotation['turns'][0]['speaker']='0'*32+':speaker0'
                elif damage=='confidence':annotation['turns'][0]['confidence']=1
                elif damage=='time':annotation['turns'][0]['end_ms']=401
                else:annotation['qualified']=True
                with self.subTest(damage=damage),self.assertRaises(ProtocolError):
                    diarization.validate_annotation(annotation,result)

if __name__=='__main__':unittest.main()
