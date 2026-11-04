"""Regression for103-T061A-CANCEL-PEER and same-ID job-generation isolation."""
from dataclasses import replace
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch

from kilix_transcribe import service
from kilix_transcribe.attestation import FEATURE, ManagedPeer
from kilix_transcribe.protocol import ProtocolError, ProviderRequest, receive_packet, send_packet
import test_content
import test_content_runtime


@unittest.skipUnless(test_content.CONTENT_AVAILABLE,'Content/License packages required')
class ManagedCancelTests(unittest.TestCase):
    setup_runtime=test_content_runtime.ContentRuntimeTests.setup_runtime

    def test_reviewer_endpoint_replacement_gets_no_cancel(self):
        self.replacement_case(False)

    def test_replacement_between_endpoint_check_and_connect_gets_no_cancel(self):
        self.replacement_case(True)

    def replacement_case(self, race):
        f,_installed,_source,_rt=self.setup_runtime('escape');peer=service.negotiate_managed(f.ipc)
        seen=[];replacement=None;server_thread=None;connected=[];armed=False;replaced=False
        original_connect=socket.socket.connect
        def replace_path():
            nonlocal replacement,server_thread,replaced
            replaced=True;(f.ipc/service.SOCKET_NAME).unlink()
            replacement=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)
            replacement.bind(str(f.ipc/service.SOCKET_NAME));(f.ipc/service.SOCKET_NAME).chmod(0o600)
            replacement.listen(1);replacement.settimeout(1)
            def server():
                try:
                    with replacement.accept()[0] as channel:
                        connected.append(True);channel.settimeout(1)
                        try:
                            value,fds=receive_packet(channel)
                            for fd in fds:os.close(fd)
                            seen.append(value['op'])
                        except ProtocolError:pass
                except socket.timeout:pass
            server_thread=threading.Thread(target=server);server_thread.start()
        def connect(channel,address):
            if race and armed and not replaced:replace_path()
            return original_connect(channel,address)
        def canceled():
            nonlocal armed
            if not (f.installation/'escaped.pid').exists():return False
            if not replaced:
                if race:armed=True
                else:replace_path()
            return True
        baseline=len(os.listdir('/proc/self/fd'))
        try:
            with patch.object(socket.socket,'connect',connect),f.audio.open('rb') as audio,self.assertRaises(ProtocolError) as e:
                service.client_request(f.ipc,f.value(job='same-id',timeout=8),audio.fileno(),managed=peer,cancelled=canceled)
            self.assertEqual(e.exception.code,'UNAUTHORIZED_PEER')
        finally:
            if server_thread:server_thread.join(2)
            if replacement:replacement.close()
            f.stop()
        self.assertEqual(seen,[],'replacement must receive no cancellation packet')
        if race:self.assertEqual(connected,[True]) # actual new connection refused before sending
        else:self.assertEqual(connected,[])        # refused before connect
        pid=int((f.installation/'escaped.pid').read_text())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        self.assertLessEqual(len(os.listdir('/proc/self/fd')),baseline)

    def test_same_ids_need_actual_job_instance_and_managed_cancel(self):
        f,_installed,_source,_rt=self.setup_runtime('escape');peer=service.negotiate_managed(f.ipc)
        marker=f.installation/'escaped.pid';errors=[];threads=[];targets={};original_send=service.send_packet
        def send(channel,event,descriptor=None):
            if event.get('type')=='accepted' and 'job_instance' in event['result']:
                targets[event['request_id']]=dict(request_id=event['request_id'],job_instance=event['result']['job_instance'])
            return original_send(channel,event,descriptor)
        def start(value):
            if marker.exists():marker.unlink()
            def submit():
                try:
                    with f.audio.open('rb') as audio:service.client_request(f.ipc,value,audio.fileno(),managed=peer)
                except ProtocolError as e:errors.append(e.code)
            thread=threading.Thread(target=submit);threads.append(thread);thread.start()
            deadline=time.monotonic()+4
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(marker.exists());return thread,int(marker.read_text()),dict(targets[value['request_id']])
        def cancel(target,peer=peer):
            return service.client_request(f.ipc,service.request_value('cancel',job_id='reused',timeout=2),
                                          managed=peer,cancel_target=target)
        value=f.value(job='reused',timeout=12)
        try:
            with patch.object(service,'send_packet',send):
                t1,pid1,first=start(value);ack=cancel(first);self.assertIs(ack['cancel_requested'],True)
                self.assertEqual(ack['target_job_instance'],first['job_instance']);t1.join(3);self.assertFalse(t1.is_alive())
                with self.assertRaises(ProcessLookupError):os.kill(pid1,0)
                # Deliberately reuse BOTH caller-controlled IDs and all arguments.
                t2,pid2,second=start(value);self.assertNotEqual(first['job_instance'],second['job_instance'])
                with self.assertRaises(ProtocolError) as e:cancel(first)
                self.assertEqual(e.exception.code,'STALE_JOB');self.assertTrue(t2.is_alive());os.kill(pid2,0)
                with self.assertRaises(ProtocolError) as e:
                    service.client_request(f.ipc,service.request_value('cancel',job_id='reused',timeout=2))
                self.assertEqual(e.exception.code,'INCOMPATIBLE_PROTOCOL');self.assertTrue(t2.is_alive());os.kill(pid2,0)
                with self.assertRaises(ProtocolError) as e:cancel(second,replace(peer,service_instance='0'*32))
                self.assertEqual(e.exception.code,'INCOMPATIBLE_PROTOCOL');self.assertTrue(t2.is_alive());os.kill(pid2,0)
                self.assertIs(cancel(second)['cancel_requested'],True);t2.join(3);self.assertFalse(t2.is_alive())
                with self.assertRaises(ProcessLookupError):os.kill(pid2,0)
                self.assertEqual(errors,['CANCELED','CANCELED']);self.assertEqual(f.service._job_requests,{})
        finally:
            f.stop()
            for thread in threads:thread.join(3)

    def test_correct_peer_callback_sends_one_bound_cancel(self):
        f,_installed,_source,_rt=self.setup_runtime('escape');peer=service.negotiate_managed(f.ipc)
        original=service.send_packet;seen=[];accepted=[];value=f.value(job='bound-cancel',timeout=8)
        def send(channel,event,descriptor=None):
            if event.get('op')=='cancel':seen.append(event)
            if event.get('type')=='accepted':accepted.append(event['result'])
            return original(channel,event,descriptor)
        with patch.object(service,'send_packet',send),f.audio.open('rb') as audio,self.assertRaises(ProtocolError) as e:
            service.client_request(f.ipc,value,audio.fileno(),managed=peer,
                                   cancelled=lambda:(f.installation/'escaped.pid').exists())
        self.assertEqual(e.exception.code,'CANCELED');self.assertEqual(len(seen),1)
        self.assertEqual(seen[0]['args'],dict(attestation=dict(feature=FEATURE,service_instance=peer.service_instance),
                                            target_request_id=value['request_id'],
                                            target_job_instance=accepted[0]['job_instance']))
        f.stop();pid=int((f.installation/'escaped.pid').read_text())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)


class ManagedCancelValidationTests(unittest.TestCase):
    def test_managed_cancel_needs_explicit_valid_target(self):
        peer=ManagedPeer('/no-such-directory',(),(),'0'*32)
        value=service.request_value('cancel',job_id='job')
        for target in [None,True,{},'x',dict(request_id='',job_instance='0'*32),
                       dict(request_id='target',job_instance='x'*32),dict(request_id='x'*65,job_instance='0'*32)]:
            with self.assertRaises(ProtocolError):
                service.client_request('/no-such-directory',value,managed=peer,cancel_target=target)
        with self.assertRaises(ProtocolError):service.client_request('/no-such-directory',value,cancel_target='target')
        self.assertEqual(ProviderRequest.from_payload(value).arguments,{})
        wire=dict(value,args=dict(attestation=dict(feature=FEATURE,service_instance='0'*32),target_request_id='target',target_job_instance='0'*32))
        self.assertEqual(ProviderRequest.from_payload(wire).arguments['target_request_id'],'target')
