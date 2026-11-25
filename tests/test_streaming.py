"""Actual inert engines and private sockets; no weights or physical devices."""
from contextlib import contextmanager
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
import wave

from kilix_transcribe.protocol import ProtocolError, ProviderRequest, receive_packet, send_packet
from kilix_transcribe.service import client_request, negotiate_stream, request_value
from kilix_transcribe.streaming import (STREAM_FEATURE, STREAM_LIMITS, SegmentStream,
                                        WorkerFrames, MAX_UPDATES)
import test_runtime as fixtures


ENGINE = r'''#!/usr/bin/python3
import json, os, pathlib, signal, subprocess, sys, time, wave
root = pathlib.Path(FIXTURE_ROOT)
mode = pathlib.Path(sys.argv[sys.argv.index('--model')+1]).read_text()
calls = root / 'calls'
index = int(calls.read_text()) if calls.exists() else 0
calls.write_text(str(index + 1))
def handle(pid):
    data = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(data[19])}
(root / f'engine-{index}.json').write_text(json.dumps(handle(os.getpid())))
with wave.open(sys.argv[sys.argv.index('--file')+1], 'rb') as audio:
    duration = audio.getnframes()*1000//audio.getframerate()
if mode == 'escape':
    child = subprocess.Popen(['/usr/bin/python3','-c',
        'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'], start_new_session=True)
    (root/'escaped.json').write_text(json.dumps(handle(child.pid)))
if mode == 'malformed':
    print('progress = 50%', flush=True)
elif mode == 'oversized':
    print('x'*1048577, flush=True)
elif mode == 'flood':
    for i in range(8):
        print('[00:00:00.000 --> 00:00:01.000]   '+'x'*4000, flush=True)
        time.sleep(.06)
    time.sleep(120)
elif mode != 'no-callbacks':
    hint = 'Wrong draft.' if index == 0 and mode in ('gated', 'escape') else 'Correct transcript.'
    print('[00:00:00.000 --> 00:00:01.000]   '+hint, flush=True)
if mode in ('gated', 'escape', 'no-callbacks'):
    (root/f'blocked-{index}').touch()
    while not (root/f'gate-{index}').exists(): time.sleep(.005)
if mode == 'fail': raise SystemExit(3)
if mode == 'oversized-json':
    pathlib.Path(sys.argv[sys.argv.index('--output-file')+1]+'.json').write_text('x'*1048577)
else:
    pathlib.Path(sys.argv[sys.argv.index('--output-file')+1]+'.json').write_text(json.dumps({
        'result': {'language':'en'}, 'transcription': [{
        'offsets':{'from':0,'to':duration}, 'text':'Correct transcript.',
        'tokens':[{'text':' Correct','offsets':{'from':0,'to':min(400,duration)}},
                  {'text':' transcript.','offsets':{'from':min(400,duration),'to':duration}}]}]}))
'''
DECODER = r'''#!/usr/bin/python3
import json, os, pathlib, shutil, sys, time
root = pathlib.Path(FIXTURE_ROOT)
(root/'decoder.json').write_text(json.dumps({'pid':os.getpid(), 'start_ticks':int(
    pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])}))
if (root/'hold-decoder').exists():
    while not (root/'decoder-gate').exists(): time.sleep(.005)
shutil.copyfile(sys.argv[sys.argv.index('-i')+1], sys.argv[-1])
'''


def until(predicate, timeout=3):
    expires = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= expires:
            raise AssertionError('inert engine did not reach its explicit gate')
        time.sleep(.005)


@contextmanager
def provider(mode='quick', seconds=1):
    fixture = fixtures.RuntimeTests('runTest')
    fixture.setUp()
    try:
        with wave.open(str(fixture.audio), 'wb') as audio:
            audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\1\0' * (seconds * 16000))
        with patch.object(fixtures, 'ENGINE', ENGINE), patch.object(fixtures, 'DECODER', DECODER):
            fixture.start(mode)
        yield fixture
    finally:
        fixture.doCleanups()


def value(fixture, *, output='text', timeout=5, job='stream-job', stream=True):
    request = fixture.value(job=job, timeout=timeout, output=output)
    if stream:
        request['args']['stream'] = STREAM_FEATURE
    return request


def submit(fixture, *, on_update=None, **kwargs):
    with fixture.audio.open('rb') as audio:
        return client_request(fixture.ipc, value(fixture, **kwargs), audio.fileno(), on_update=on_update)


def raw(fixture, **kwargs):
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    channel.settimeout(3)
    channel.connect(str(fixture.service.path))
    try:
        with fixture.audio.open('rb') as audio:
            send_packet(channel, value(fixture, **kwargs), audio.fileno())
        accepted, descriptors = receive_packet(channel)
        assert accepted['type'] == 'accepted' and not descriptors
        return channel
    except BaseException:
        channel.close()
        raise


def assert_reaped(test, path):
    record = json.loads(path.read_text())
    current = Path(f"/proc/{record['pid']}/stat")
    if current.exists():
        ticks = int(current.read_text().rsplit(')', 1)[1].split()[19])
        test.assertNotEqual(ticks, record['start_ticks'], 'owned incarnation survived teardown')


class StreamingTests(unittest.TestCase):
    def test_partials_precede_completion_then_correct_stabilize_and_render(self):
        with provider('gated', seconds=8) as fixture, concurrent.futures.ThreadPoolExecutor(1) as pool:
            updates = []
            future = pool.submit(submit, fixture, on_update=updates.append)
            try:
                until(lambda: len(updates) >= 1)
                self.assertFalse(future.done())
                self.assertFalse(updates[0].stable)
                self.assertEqual(updates[0].text, 'Wrong draft.')
                self.assertTrue((fixture.installation/'blocked-0').exists())
                self.assertFalse((fixture.installation/'gate-0').exists())
                (fixture.installation/'gate-0').touch()
                until(lambda: len(updates) >= 2)
                self.assertEqual(updates[-1].text, 'Correct transcript.')
                self.assertFalse(updates[-1].stable)
                self.assertGreater(updates[-1].revision, updates[0].revision)
                self.assertFalse(future.done())
                (fixture.installation/'gate-1').touch()
                result = future.result(3)
                self.assertTrue(updates[-1].stable)
                self.assertEqual(result['output'], 'Correct transcript.\n')
                self.assertEqual([u.revision for u in updates], list(range(len(updates))))
                self.assertEqual(result['segments'][0]['revision'], updates[-1].revision)
                for record in fixture.installation.glob('engine-*.json'):
                    assert_reaped(self, record)
            finally:
                fixture.service.stop()

    def test_all_final_formats_and_legacy_path(self):
        for output in ('text', 'json', 'srt', 'webvtt'):
            with self.subTest(output=output), provider() as fixture:
                updates = []
                result = submit(fixture, output=output, on_update=updates.append)
                legacy = submit(fixture, output=output, stream=False)
                self.assertTrue(updates and not updates[0].stable and updates[-1].stable)
                self.assertEqual(result['text'], legacy['text'])
                if output != 'json': self.assertEqual(result['output'], legacy['output'])
                self.assertNotIn('Wrong draft', result['output'])

    def test_old_and_new_hello_and_incompatible_negotiation(self):
        with provider() as fixture:
            old = client_request(fixture.ipc, request_value('hello', args={'protocol_major':1,'protocol_minor':0}))
            self.assertEqual(old, {'protocol_major':1,'protocol_minor':0})
            self.assertEqual(negotiate_stream(fixture.ipc)['limits'], STREAM_LIMITS)
            models = client_request(fixture.ipc, request_value('models'))
            self.assertIn('unstable_partials', models['models'][0]['capabilities'])
            with patch('kilix_transcribe.service.client_request', return_value=old):
                with self.assertRaises(ProtocolError) as error: negotiate_stream(fixture.ipc)
                self.assertEqual(error.exception.code, 'INCOMPATIBLE_PROTOCOL')
            request = value(fixture)
            request['args']['stream'] = 'future-v2'
            with self.assertRaises(ProtocolError): ProviderRequest.from_payload(request)
            self.assertFalse((fixture.installation/'calls').exists())

    def test_cancel_decoder_inference_and_updates_reaps_actual_children(self):
        for phase in ('decoder', 'inference', 'update'):
            with self.subTest(phase=phase), provider('escape') as fixture:
                if phase == 'decoder': (fixture.installation/'hold-decoder').touch()
                with raw(fixture) as channel:
                    marker = fixture.installation / ('decoder.json' if phase == 'decoder' else 'escaped.json')
                    until(marker.exists)
                    if phase == 'update':
                        event, descriptors = receive_packet(channel)
                        self.assertEqual(event['type'], 'update')
                        self.assertFalse(descriptors)
                    client_request(fixture.ipc, request_value('cancel', job_id='stream-job'))
                    while True:
                        event, descriptors = receive_packet(channel)
                        self.assertFalse(descriptors)
                        if event['type'] == 'error': break
                        self.assertEqual(event['type'], 'update')
                    self.assertEqual(event['error']['code'], 'CANCELED')
                for record in fixture.installation.glob('*.json'):
                    if record.name.startswith(('decoder', 'engine-', 'escaped')): assert_reaped(self, record)
                self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')

    def test_deadline_and_disconnect_do_not_deliver_final(self):
        for ending in ('deadline', 'disconnect'):
            with self.subTest(ending=ending), provider('escape') as fixture:
                with raw(fixture, timeout=.8 if ending == 'deadline' else 5) as channel:
                    until((fixture.installation/'escaped.json').exists)
                    if ending == 'disconnect': channel.close()
                    else:
                        while True:
                            event, descriptors = receive_packet(channel)
                            self.assertFalse(descriptors)
                            if event['type'] == 'error': break
                            self.assertEqual(event['type'], 'update')
                        self.assertEqual(event['error']['code'], 'DEADLINE_EXCEEDED')
                until(lambda: not fixture.service._jobs)
                assert_reaped(self, fixture.installation/'escaped.json')

    def test_other_client_disconnect_does_not_cancel_authorized_job(self):
        with provider('gated') as fixture, raw(fixture) as active:
            until((fixture.installation/'blocked-0').exists)
            other = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            other.connect(str(fixture.service.path))
            other.close()
            with self.assertRaises(ProtocolError) as error: submit(fixture, job='other-client')
            self.assertEqual(error.exception.code, 'BUSY')
            self.assertEqual(len(fixture.service._jobs), 1)
            (fixture.installation/'gate-0').touch()
            terminal = None
            while terminal is None:
                event, descriptors = receive_packet(active)
                if event['type'] == 'result': terminal = event
                for descriptor in descriptors: os.close(descriptor)
            self.assertEqual(terminal['job_id'], 'stream-job')

    def test_bad_callback_output_and_window_json_fail_without_result(self):
        for mode in ('malformed', 'oversized', 'oversized-json', 'fail'):
            with self.subTest(mode=mode), provider(mode) as fixture:
                with self.assertRaises(ProtocolError): submit(fixture, on_update=lambda u: None)
                self.assertEqual(client_request(fixture.ipc, request_value('status'))['provider_state'], 'ready')
                assert_reaped(self, fixture.installation/'engine-0.json')

    def test_silence_has_no_invented_partial(self):
        with provider() as fixture:
            with wave.open(str(fixture.audio), 'wb') as audio:
                audio.setparams((1,2,16000,0,'NONE','not compressed'))
                audio.writeframes(bytes(32000))
            updates = []
            result = submit(fixture, on_update=updates.append)
            self.assertEqual((updates, result['text']), ([], ''))
            self.assertFalse((fixture.installation/'calls').exists())

    def test_wire_stable_update_must_match_corrected_final(self):
        import kilix_transcribe.service as service
        original = service.send_packet
        def conflicting(channel, event, descriptor=None):
            if event['type'] == 'update' and event['result']['segment']['stable']:
                event = copy.deepcopy(event)
                event['result']['segment']['text'] = 'Other stable text.'
            return original(channel, event, descriptor)
        with provider() as fixture, patch.object(service, 'send_packet', conflicting):
            with self.assertRaises(ProtocolError) as error: submit(fixture, on_update=lambda update: None)
            self.assertEqual(error.exception.code, 'INVALID_RESPONSE')

    def test_callback_failure_closes_only_owned_job_and_reaps(self):
        with provider('escape') as fixture:
            def failed(update): raise RuntimeError('observer failed')
            with self.assertRaisesRegex(RuntimeError, 'observer failed'):
                submit(fixture, on_update=failed)
            until(lambda: not fixture.service._jobs)
            assert_reaped(self, fixture.installation/'escaped.json')

    def test_cli_shows_provisional_revisions_and_only_correct_final_stdout(self):
        with provider() as fixture:
            environment = dict(os.environ, XDG_RUNTIME_DIR=str(fixture.ipc))
            process = subprocess.run([sys.executable, '-B', '-m', 'kilix_transcribe', 'file',
                str(fixture.audio), '--stream', '--format', 'srt'], env=environment,
                capture_output=True, timeout=5)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn(b'"stable":false', process.stderr)
            self.assertIn(b'"stable":true', process.stderr)
            self.assertEqual(process.stdout, b'1\n00:00:00,000 --> 00:00:01,000\nCorrect transcript.\n')

    def test_actual_slow_socket_bounds_queue_and_reaps_engine(self):
        from kilix_transcribe.service import Service
        original = Service._handle
        def small_buffer(service, channel):
            channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
            return original(service, channel)
        with patch.object(Service, '_handle', small_buffer), provider('flood') as fixture:
            with raw(fixture) as channel:
                until((fixture.installation/'engine-0.json').exists)
                started = time.monotonic()
                until(lambda: not fixture.service._jobs, timeout=2)
                self.assertLess(time.monotonic()-started, 1.5)
                assert_reaped(self, fixture.installation/'engine-0.json')
                # No final is delivered, and buffered updates stay finite.
                count = 0
                while True:
                    try: event, descriptors = receive_packet(channel)
                    except ProtocolError: break
                    self.assertFalse(descriptors)
                    self.assertNotEqual(event['type'], 'result')
                    count += 1
                    if event['type'] == 'error':
                        self.assertEqual(event['error']['code'], 'SLOW_CONSUMER')
                self.assertLessEqual(count, 8)

    def test_recording_route_observes_partials_after_proved_microphone_close(self):
        try:
            from voicelib.microphone import CaptureControl
        except ImportError:
            self.skipTest('managed microphone API absent')
        from kilix_transcribe.recording import record_and_transcribe
        with provider() as fixture:
            phases, updates = [], []
            def observed(update):
                self.assertEqual(phases[-1], 'inactive')
                updates.append(update)
            result = record_and_transcribe(fixture.ipc, seconds=1, indicator=phases.append,
                timeout=5, locked=lambda: False, namespace=str(fixture.root/'microphone'),
                config={'audio':{'capture_cmd':[sys.executable,'-B','-c',
                    "import os;os.write(1,b'\\1\\0'*6400)"]}}, on_update=observed)
            self.assertTrue(updates and not updates[0].stable and updates[-1].stable)
            self.assertEqual(result['text'], 'Correct transcript.\n')
            self.assertEqual(phases[0], 'recording')

    def test_actual_shared_grant_survives_updates_and_releases_after_reap(self):
        try:
            import kilix_device_lease as device_leases
        except ImportError:
            self.skipTest('shared device lease API absent')
        from kilix_transcribe.owned import ExecutionPolicy
        with provider('escape') as fixture:
            namespace = str(fixture.root/'execution')
            fixture.service.execution_policy = ExecutionPolicy('cpu-development', namespace=namespace)
            with raw(fixture) as channel:
                until((fixture.installation/'escaped.json').exists)
                event, descriptors = receive_packet(channel)
                self.assertEqual(event['type'], 'update')
                self.assertFalse(descriptors)
                with self.assertRaises(device_leases.LeaseError) as error:
                    device_leases.acquire(job_id='too-early', workload='llm-turn', device='cpu-development',
                        deadline=time.monotonic()+.1, namespace=namespace)
                self.assertEqual(error.exception.code, 'deadline')
                client_request(fixture.ipc, request_value('cancel', job_id='stream-job'))
                while True:
                    event, descriptors = receive_packet(channel)
                    self.assertFalse(descriptors)
                    if event['type'] == 'error': break
                self.assertEqual(event['error']['code'], 'CANCELED')
                assert_reaped(self, fixture.installation/'escaped.json')
                successor = device_leases.acquire(job_id='after-reap', workload='llm-turn', device='cpu-development',
                    deadline=time.monotonic()+1, namespace=namespace)
                successor.release(cleanup_complete=True)

    def test_streaming_supervisor_loss_quarantines_after_all_handles_close(self):
        try:
            import kilix_device_lease as device_leases
        except ImportError:
            self.skipTest('shared device lease API absent')
        process = subprocess.run([sys.executable, '-B',
            str(Path(__file__).with_name('streaming_loss_probe.py'))],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        evidence = json.loads(process.stdout)
        print('STREAMING_OWNED_LOSS_EVIDENCE '+process.stdout.strip())
        self.assertEqual(evidence['terminal'], 'SUPERVISOR_FAILED')
        self.assertEqual(evidence['successor_after_complete_observer_cleanup'], 'unavailable')
        self.assertTrue(evidence['unrelated_preserved'])
        self.assertEqual(evidence['children_after_probe_cleanup'], [])


class StreamValidationTests(unittest.TestCase):
    def update(self, revision=0, stable=False):
        return {'sequence':revision, 'segment':{'segment_id':0, 'revision':revision,
            'start_ms':0, 'end_ms':1000, 'text':'real hypothesis', 'stable':stable, 'words':[]}}

    def test_malformed_oversized_and_truncated_worker_frames(self):
        for length in (0, 101):
            with self.assertRaises(ProtocolError): WorkerFrames(100).feed(struct.pack('!I',length))
        for payload in (b'{', b'\xff', b'{"x":1,"x":2}'):
            decoder = WorkerFrames(100)
            decoder.feed(struct.pack('!I',len(payload)))
            with self.assertRaises(ProtocolError): decoder.feed(payload)
        for prefix, body in ((b'\0', b''), (struct.pack('!I', 20), b'{}')):
            decoder = WorkerFrames(100)
            decoder.feed(prefix)
            if body: decoder.feed(body)
            with self.assertRaises(ProtocolError): decoder.eof()

    def test_revision_stability_sequence_and_population_bounds(self):
        state = SegmentStream('transcribe')
        state.apply(self.update())
        for value in (self.update(), self.update(2)):
            with self.assertRaises(ProtocolError): state.apply(value)
        state.apply(self.update(1, True))
        with self.assertRaises(ProtocolError): state.apply(self.update(2))
        for change in ({'text':'x'*32769}, {'segment_id':180}, {'end_ms':3600001}, {'stable':1}):
            value = self.update()
            value['segment'].update(change)
            with self.assertRaises(ProtocolError): SegmentStream('transcribe').apply(value)
        state = SegmentStream('transcribe')
        state.sequence = MAX_UPDATES
        value = self.update()
        value['sequence'] = MAX_UPDATES
        with self.assertRaises(ProtocolError) as error: state.apply(value)
        self.assertEqual(error.exception.code, 'LIMIT_EXCEEDED')

    def test_overlap_midpoints_assign_each_token_once(self):
        from kilix_transcribe.engine_stream import _final_window
        from kilix_transcribe.worker import _words
        payload = json.dumps({'result':{'language':'en'},'transcription':[{'text':'one two three',
            'offsets':{'from':19000,'to':21000},'tokens':[
            {'text':' one','offsets':{'from':19000,'to':19500}},
            {'text':' two','offsets':{'from':19500,'to':20500}},
            {'text':' three','offsets':{'from':20500,'to':21000}}]}]}).encode()
        left = _final_window(payload, base=0, start=0, end=20000, words=_words)
        right = _final_window(payload, base=0, start=20000, end=40000, words=_words)
        self.assertEqual((left[1], right[1]), ('one', 'two three'))
        self.assertEqual((right[2][0].start_ms, right[2][0].end_ms), (19500, 20500))

    def test_unstable_or_conflicting_final_never_commits(self):
        for stable in (False, True):
            state = SegmentStream('transcribe')
            update = self.update(stable=stable)
            state.apply(update)
            result = {'engine':{'id':'whisper.cpp'},'model':{'id':'fixture'},'language':'en',
                      'segments':[dict(update['segment'], stable=True, text='mismatched')]}
            with self.assertRaises(ProtocolError): state.finish(result)

    def test_end_boundary_and_empty_retraction_render_correctly(self):
        from kilix_transcribe.engine_stream import _final_window
        from kilix_transcribe.worker import _words
        from kilix_transcribe.surface import Transcript, SegmentUpdate, render_transcript
        payload = json.dumps({'result':{'language':'en'},'transcription':[{'text':'last word',
            'offsets':{'from':0,'to':400},'tokens':[
            {'text':' last','offsets':{'from':0,'to':400}},
            {'text':' word','offsets':{'from':400,'to':400}}]}]}).encode()
        result = _final_window(payload, base=0, start=0, end=400, words=_words,
                               include_end=True, window_end=400)
        self.assertEqual(result[1], 'last word')
        transcript = Transcript('transcribe','whisper.cpp','fixture','en',
            (SegmentUpdate(0,1,0,1000,'',True), SegmentUpdate(1,1,1000,2000,'kept',True)))
        self.assertEqual(render_transcript(transcript,'text'),'kept\n')
        self.assertEqual(render_transcript(transcript,'srt'),
                         '1\n00:00:01,000 --> 00:00:02,000\nkept\n')


if __name__ == '__main__': unittest.main()
