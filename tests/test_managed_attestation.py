"""Managed evidence on synthetic bytes/tools and scratch real receipt authority."""
from contextlib import contextmanager
from dataclasses import replace
import copy
import json
import os
import threading
import time
import socket
import unittest
from unittest.mock import patch

from kilix_transcribe import service, runtime
from kilix_transcribe.attestation import FEATURE
from kilix_transcribe.protocol import ProtocolError, ProviderRequest
from kilix_transcribe.results import decode_result
import test_content
import test_content_runtime as content_runtime_fixtures
import test_runtime as runtime_fixtures


@unittest.skipUnless(test_content.CONTENT_AVAILABLE, 'Content/License packages required')
class ManagedAttestationTests(unittest.TestCase):
    setup_runtime = content_runtime_fixtures.ContentRuntimeTests.setup_runtime

    def submit(self, fixture, peer, job='attested'):
        value = fixture.value(job=job)
        with fixture.audio.open('rb') as audio:
            return service.client_request(fixture.ipc, value, audio.fileno(), managed=peer), value

    def revoke(self, installed):
        for path in installed.receipts.iterdir():
            path.unlink()

    def test_negotiated_result_and_legacy_population(self):
        f, _installed, _source, rt = self.setup_runtime()
        self.assertEqual(service.client_request(f.ipc, service.request_value('hello',args={
            'protocol_major':1,'protocol_minor':0})), {'protocol_major':1,'protocol_minor':0})
        legacy = service.client_request(f.ipc,service.request_value('models'))
        self.assertNotIn('model_sha256',legacy['models'][0])
        self.assertNotIn('attestation',f.submit())
        peer=service.negotiate_managed(f.ipc)
        record=service.client_request(f.ipc,service.request_value('models'),managed=peer)['models'][0]
        self.assertEqual(record['model_sha256'],rt.manifest['files']['model.bin'])
        self.assertIs(record['receipt_enforced'],True)
        result, value=self.submit(f,peer)
        self.assertEqual(result['model_sha256'],rt.manifest['files']['model.bin'])
        self.assertIs(result['receipt_enforced'],True)
        self.assertEqual(result['attestation'],dict(feature=FEATURE,service_instance=peer.service_instance,
            request_id=value['request_id'],job_id=value['job_id'],audio_sha256=value['args']['audio']['sha256'],
            job_instance=result['attestation']['job_instance']))
        self.assertEqual(len(result['attestation']['job_instance']),32)
        self.assertNotIn('attestation',value['args']) # no mutation of caller request

    def test_revoked_receipt_before_job(self):
        f, installed, _source, _rt = self.setup_runtime()
        peer=service.negotiate_managed(f.ipc)
        self.revoke(installed)
        with self.assertRaises(ProtocolError) as e:self.submit(f,peer)
        self.assertEqual(e.exception.code,'INVALID_RUNTIME')
        self.assertFalse((f.installation/'engine.pid').exists())

    def test_revocation_after_snapshot_refuses_result(self):
        f, installed, source, _rt = self.setup_runtime()
        peer=service.negotiate_managed(f.ipc);original=source.open
        @contextmanager
        def revoked(check):
            with original(check) as asset:
                self.revoke(installed)
                yield asset
        with patch.object(source,'open',revoked):
            with self.assertRaises(ProtocolError) as e:self.submit(f,peer)
        self.assertEqual(e.exception.code,'INVALID_RUNTIME')
        self.assertTrue((f.installation/'engine.pid').exists())
        self.assertEqual(service.client_request(f.ipc,service.request_value('status'))['provider_state'],'ready')

    def test_final_delivery_rechecks_same_authority(self):
        f, installed, _source, _rt = self.setup_runtime()
        peer=service.negotiate_managed(f.ipc);original=service.run_job
        def revoke_after_job(*a,**kw):
            result=original(*a,**kw);self.revoke(installed);return result
        with patch.object(service,'run_job',revoke_after_job):
            with self.assertRaises(ProtocolError) as e:self.submit(f,peer)
        self.assertEqual(e.exception.code,'INVALID_RUNTIME')

    def test_snapshot_bytes_and_metadata_survive_replacement(self):
        f, installed, source, rt = self.setup_runtime()
        peer=service.negotiate_managed(f.ipc);original=source.open;digest=rt.manifest['files']['model.bin']
        @contextmanager
        def replace_bytes(check):
            with original(check) as asset:
                (installed.selected/'model.bin').write_bytes(b'fail')
                yield asset
        original_run=service.run_job
        def mutate_metadata(*a,**kw):
            result=original_run(*a,**kw)
            rt.model_revision='post-job-drift'
            rt.manifest['engine_revision']='post-job-drift'
            return result
        with patch.object(source,'open',replace_bytes),patch.object(service,'run_job',mutate_metadata):
            result,_=self.submit(f,peer)
        self.assertIn('Test transcript',result['text'])
        self.assertEqual(result['model_sha256'],digest)
        self.assertEqual(result['model_revision'],'5359861c739e955e79d9a303bcbc70fb988958b1')
        self.assertEqual(result['engine_revision'],runtime.ENGINE_COMMIT)
        with self.assertRaises(ProtocolError):self.submit(f,peer,'replaced-model')

    def test_endpoint_instance_and_peer_replacement_refuse(self):
        f,_installed,_source,rt=self.setup_runtime();old=service.negotiate_managed(f.ipc)
        bad=replace(old,credentials=(old.credentials[0]+1,*old.credentials[1:]))
        with self.assertRaises(ProtocolError) as e:self.submit(f,bad)
        self.assertEqual(e.exception.code,'UNAUTHORIZED_PEER')
        f.stop();f.service=service.Service(rt,f.ipc);f.thread=threading.Thread(target=f.service.serve)
        f.thread.start();self.assertTrue(f.service.ready.wait(3))
        with self.assertRaises(ProtocolError) as e:self.submit(f,old)
        self.assertEqual(e.exception.code,'UNAUTHORIZED_PEER')
        current=service.negotiate_managed(f.ipc)
        with self.assertRaises(ProtocolError) as e:self.submit(f,replace(current,service_instance=old.service_instance))
        self.assertEqual(e.exception.code,'INCOMPATIBLE_PROTOCOL')
        self.assertFalse((f.installation/'engine.pid').exists())

    def test_managed_cancel_reaps_escaped_child_and_recovers(self):
        f,_installed,_source,_rt=self.setup_runtime('escape');peer=service.negotiate_managed(f.ipc)
        canceled=threading.Event();errors=[]
        def run():
            with f.audio.open('rb') as audio:
                try:service.client_request(f.ipc,f.value(job='cancel-attested',timeout=8),audio.fileno(),
                                           managed=peer,cancelled=canceled.is_set)
                except ProtocolError as e:errors.append(e.code)
        thread=threading.Thread(target=run);thread.start()
        try:
            marker=f.installation/'escaped.pid';deadline=time.monotonic()+4
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(marker.exists());pid=int(marker.read_text());canceled.set();thread.join(3)
            self.assertFalse(thread.is_alive());self.assertEqual(errors,['CANCELED'])
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                if service.client_request(f.ipc,service.request_value('status'))['provider_state']=='ready':break
                time.sleep(.01)
            self.assertEqual(service.client_request(f.ipc,service.request_value('status'))['provider_state'],'ready')
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        finally:canceled.set();f.service.stop();thread.join(3)

    def test_managed_deadline_reaps_before_provider_terminal(self):
        from kilix_transcribe.protocol import send_packet, receive_packet
        f,_installed,_source,_rt=self.setup_runtime('escape');peer=service.negotiate_managed(f.ipc)
        value=f.value(job='deadline-attested',timeout=3)
        value['args']['attestation']=dict(feature=FEATURE,service_instance=peer.service_instance)
        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as channel,f.audio.open('rb') as audio:
            channel.settimeout(6);channel.connect(str(f.ipc/service.SOCKET_NAME));send_packet(channel,value,audio.fileno())
            marker=f.installation/'escaped.pid';deadline=time.monotonic()+2.5
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(marker.exists());pid=int(marker.read_text())
            for _ in range(10):
                event,fds=receive_packet(channel);self.assertEqual(fds,())
                if event['type']=='error':break
            self.assertEqual(event['error']['code'],'DEADLINE_EXCEEDED')
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_managed_disconnect_reaps_escaped_child(self):
        from kilix_transcribe.protocol import send_packet
        f,_installed,_source,_rt=self.setup_runtime('escape');peer=service.negotiate_managed(f.ipc)
        value=f.value(job='disconnect-attested',timeout=8)
        value['args']['attestation']=dict(feature=FEATURE,service_instance=peer.service_instance)
        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as channel,f.audio.open('rb') as audio:
            channel.connect(str(f.ipc/service.SOCKET_NAME));send_packet(channel,value,audio.fileno())
            marker=f.installation/'escaped.pid';deadline=time.monotonic()+4
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(marker.exists());pid=int(marker.read_text())
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if service.client_request(f.ipc,service.request_value('status'))['provider_state']=='ready':break
            time.sleep(.01)
        self.assertEqual(service.client_request(f.ipc,service.request_value('status'))['provider_state'],'ready')
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_strict_bound_decoder_rejects_evidence_tampering(self):
        f,_installed,_source,_rt=self.setup_runtime();peer=service.negotiate_managed(f.ipc)
        result,value=self.submit(f,peer)
        args=dict(value['args'],attestation=dict(feature=FEATURE,service_instance=peer.service_instance))
        metadata=dict(engine_id=result['engine']['id'],model_id=result['model']['id'],
            **{k:result[k] for k in ('engine_revision','model_revision','duration_ms','model_sha256','receipt_enforced','attestation')})
        def decode(doc,meta=metadata,args=args):
            return decode_result(json.dumps(doc).encode(),meta,args,request_id=value['request_id'],job_id=value['job_id'],job_instance=result['attestation']['job_instance'])
        self.assertEqual(decode(result),result)
        for mutate in [lambda d:d.pop('attestation'),lambda d:d.update(receipt_enforced=1),
            lambda d:d.update(model_sha256='z'*64),lambda d:d['attestation'].update(request_id='wrong'),
            lambda d:d['attestation'].update(job_id='wrong'),lambda d:d['attestation'].update(audio_sha256='0'*64),
            lambda d:d['attestation'].update(service_instance='0'*32),
            lambda d:d['attestation'].update(job_instance='0'*32),lambda d:d.update(extra=True)]:
            doc=copy.deepcopy(result);mutate(doc)
            with self.assertRaises(ProtocolError):decode(doc)
        with self.assertRaises(ProtocolError):decode(result,args=value['args']) # legacy remains strict
        forged=copy.deepcopy(metadata);forged['model_sha256']='0'*64
        with self.assertRaises(ProtocolError):decode(result,meta=forged)

    def test_fd_cleanup_success_and_refusal(self):
        f,installed,_source,_rt=self.setup_runtime();peer=service.negotiate_managed(f.ipc)
        baseline=len(os.listdir('/proc/self/fd'));self.submit(f,peer)
        self.revoke(installed)
        with self.assertRaises(ProtocolError):self.submit(f,peer,'revoked')
        f.stop();self.assertLessEqual(len(os.listdir('/proc/self/fd')),baseline)


class UnmanagedNegotiationTests(unittest.TestCase):
    def test_unmanaged_runtime_refuses_without_execution(self):
        f=runtime_fixtures.RuntimeTests('runTest');f.setUp();self.addCleanup(f.doCleanups);f.start()
        with self.assertRaises(ProtocolError) as e:service.negotiate_managed(f.ipc)
        self.assertEqual(e.exception.code,'INVALID_RUNTIME')
        self.assertFalse((f.installation/'engine.pid').exists())

    def test_old_peer_negotiation_refuses(self):
        for response in [{'protocol_major':1,'protocol_minor':0},
                         {'protocol_major':1,'protocol_minor':1,'features':['segments-v1']}]:
            with patch.object(service,'client_request',return_value=response):
                with self.assertRaises(ProtocolError) as e:service.negotiate_managed('/synthetic')
            self.assertEqual(e.exception.code,'INCOMPATIBLE_PROTOCOL')

    def test_feature_version_and_task_validation(self):
        for features,minor in [([FEATURE],1),([FEATURE,'unknown'],2),(['unknown'],2)]:
            with self.assertRaises(ProtocolError):service.request_value('hello',args={
                'protocol_major':1,'protocol_minor':minor,'features':features})
        valid=service.request_value('hello',args={'protocol_major':1,'protocol_minor':3,'features':[FEATURE]})
        self.assertEqual(ProviderRequest.from_payload(valid).operation,'hello')
