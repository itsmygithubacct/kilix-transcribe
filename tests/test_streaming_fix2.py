"""Review2 regressions using public CLI and owned inert service/worker routes."""
import copy
import ctypes
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_streaming import until
from test_streaming_fixes import alive, handle, word_provider, word_submit, update
from kilix_transcribe.protocol import ProtocolError
from kilix_transcribe.service import client_request
from kilix_transcribe.streaming import SegmentStream
from kilix_transcribe.surface import WordTiming, _segment_payload


class SharedDiagnosticTests(unittest.TestCase):
    def test_prewarmed_public_clis_preserve_shared_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader, writer = os.pipe()
            duplicate = os.dup(writer)
            capacity = fcntl.fcntl(writer, fcntl.F_GETPIPE_SZ)
            os.set_blocking(writer, False)
            self.assertEqual(os.write(writer, b'x' * capacity), capacity)
            os.set_blocking(writer, True)
            processes, handles, pidfds = [], [], []
            libc = ctypes.CDLL(None,use_errno=True)
            flags = []
            try:
                for index in range(2):
                    script = ('from pathlib import Path\nimport time\n'
                              'import kilix_transcribe.runtime,kilix_transcribe.service\n'
                              'from kilix_transcribe.cli import main\n'
                              'import kilix_transcribe.diagnostic_writer as diagnostics\n'
                              'original=diagnostics.write_deadline\n'
                              f'def entered(*args):\n Path({str(root / f"entered-{index}")!r}).touch()\n return original(*args)\n'
                              'diagnostics.write_deadline=entered\n'
                              f'Path({str(root / f"ready-{index}")!r}).touch()\n'
                              f'while not Path({str(root / f"gate-{index}")!r}).exists():time.sleep(.001)\n'
                              f'raise SystemExit(main(["file",{str(root / "absent.wav")!r}]))\n')
                    process = subprocess.Popen([sys.executable, '-B', '-c', script],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=writer)
                    pidfd = libc.syscall(434,process.pid,0)
                    if pidfd < 0:
                        process.kill(); process.wait(3); process.stdout.close()
                        raise OSError(ctypes.get_errno(),'owned CLI pidfd')
                    processes.append(process); handles.append(handle(process.pid)); pidfds.append(pidfd)
                until(lambda: all((root / f'ready-{index}').exists() for index in range(2)), timeout=3)
                (root / 'gate-0').touch()
                # Observe the actual bounded writer while its receiver is full.
                until((root / 'entered-0').exists, timeout=2)
                flags.append(fcntl.fcntl(writer, fcntl.F_GETFL))
                (root / 'gate-1').touch()
                end = time.monotonic() + 2
                while any(process.poll() is None for process in processes) and time.monotonic() < end:
                    flags.append(fcntl.fcntl(writer, fcntl.F_GETFL))
                    time.sleep(.002)
                outputs = [process.communicate(timeout=1)[0] for process in processes]
                final_flags = [fcntl.fcntl(fd, fcntl.F_GETFL) for fd in (writer, duplicate)]
                print(json.dumps(dict(shared_cli_handles=handles, exits=[p.returncode for p in processes],
                    stdout_lengths=list(map(len, outputs)), observed_flags=flags, final_flags=final_flags,
                    connected_reader_never_drained=True)), flush=True)
                self.assertEqual([p.returncode for p in processes], [69, 69])
                self.assertEqual(outputs, [b'', b''])
                self.assertTrue(all(not value & os.O_NONBLOCK for value in flags + final_flags))
            finally:
                for process, identity, pidfd in zip(processes, handles, pidfds):
                    if process.poll() is None:
                        self.assertEqual(libc.syscall(424,pidfd,signal.SIGKILL,0,0),0)
                    process.wait(3); process.stdout.close(); os.close(pidfd)
                    self.assertFalse(alive(identity))
                print(json.dumps(dict(shared_cli_cleanup=handles, owned_reaped=True)), flush=True)
                os.set_blocking(writer, True)
                for descriptor in (reader, writer, duplicate): os.close(descriptor)

    def test_normal_unicode_delivery_preserves_blocking_and_nonblocking_flags(self):
        from kilix_transcribe.cli import _stderr_write
        for blocking in (True, False):
            with self.subTest(blocking=blocking):
                reader, writer = os.pipe()
                os.set_blocking(writer, blocking)
                stream = SimpleNamespace(fileno=lambda: writer)
                before = fcntl.fcntl(writer, fcntl.F_GETFL)
                try:
                    with patch.object(sys, 'stderr', stream): _stderr_write('日本語 é\n', required=True)
                    self.assertEqual(os.read(reader, 100), '日本語 é\n'.encode())
                    self.assertEqual(fcntl.fcntl(writer, fcntl.F_GETFL), before)
                finally: os.close(reader); os.close(writer)


class CommittedTokenTests(unittest.TestCase):
    def test_long_word_tail_punctuation_unicode_and_eof_are_owned_once(self):
        cases = [
            ([(' electro',17000,19500),('magnetic',19500,21000)], 'electromagnetic', (17000,21000)),
            ([(' electro',17000,19500),('magnetic',19500,21000),('!',21000,21000)], 'electromagnetic!', (17000,21000)),
            ([('日本',17000,19500),('語',19500,21000),('。',21000,21000)], '日本語。', (17000,21000)),
            ([(' e',17000,19500),('\u0301',19500,21000)], 'e\u0301', (17000,21000)),
        ]
        for pieces, expected, timing in cases:
            tokens = [dict(text=text, **{'from':start,'to':end}) for text,start,end in pieces]
            with self.subTest(expected=expected), word_provider(tokens,23000) as fixture:
                history = []
                value = fixture.value(timeout=5,output='json'); value['args']['stream']='segments-v1'
                with fixture.audio.open('rb') as audio:
                    result = client_request(fixture.ipc,value,audio.fileno(),on_update=history.append)
                legacy = word_submit(fixture,stream=False)
                self.assertEqual(result['text'],expected+'\n')
                self.assertEqual(result['text'],legacy['text'])
                words = [word for segment in result['segments'] for word in segment['words']]
                self.assertEqual([(w['text'],w['start_ms'],w['end_ms']) for w in words],[(expected,*timing)])
                self.assertEqual([segment['text'] for segment in result['segments']],[expected,''])
                self.assertEqual([_segment_payload(segment) for segment in history if segment.stable],result['segments'])
        tokens = [dict(text=' electro',**{'from':17000,'to':19500}),
                  dict(text='magnetic',**{'from':19500,'to':21000}),
                  dict(text='.',**{'from':23000,'to':23000})]
        with word_provider(tokens,23000) as fixture:
            result = word_submit(fixture)
            self.assertEqual(result['text'],'electromagnetic\n.\n')
            self.assertEqual(result['segments'][1]['words'][0]['start_ms'],23000)

    def test_identical_neighbors_at_distinct_times_are_retained(self):
        tokens = [dict(text=' electro',**{'from':17000,'to':19500}),
                  dict(text='magnetic',**{'from':19500,'to':21000}),
                  dict(text=' magnetic',**{'from':21100,'to':21500}),
                  dict(text=' magnetic',**{'from':21600,'to':22000}),
                  dict(text='.',**{'from':22000,'to':22000})]
        with word_provider(tokens,23000) as fixture:
            result = word_submit(fixture)
            legacy = word_submit(fixture,stream=False)
            words = [word for segment in result['segments'] for word in segment['words']]
            reference = [word for segment in legacy['segments'] for word in segment['words']]
            self.assertEqual(words,reference)
            self.assertEqual([w['text'] for w in words],['electromagnetic','magnetic','magnetic.'])
            self.assertEqual(result['text'],'electromagnetic\nmagnetic magnetic.\n')

    def test_committed_occurrence_multiset_preserves_an_extra_identical_atom(self):
        from kilix_transcribe.engine_stream import _final_window
        from kilix_transcribe.worker import _words
        tokens = [{'text':'.','offsets':{'from':20000,'to':20000}} for _ in range(3)]
        payload = json.dumps(dict(result={'language':'en'},transcription=[dict(tokens=tokens)])).encode()
        assigned = []
        language,text,words = _final_window(payload,base=0,start=20000,end=21000,words=_words,
            committed=[('.',20000,20000)] * 2, assigned=assigned)
        self.assertEqual((language,text),('en','.'))
        self.assertEqual(len(words),1)
        self.assertEqual(assigned,[('.',20000,20000)])


def eof_history():
    return [update(stable=True,end=20000,text='.',words=[dict(text='.',start_ms=20000,end_ms=20000)]),
            update(1,1,stable=True,end=21000,text='next')]


class EofAdvanceTests(unittest.TestCase):
    def test_revalidate_before_state_mutation_and_keep_actual_eof(self):
        stream = SegmentStream('transcribe')
        first,advance = eof_history(); stream.apply(first)
        before = (stream.sequence,stream.text_bytes,stream.word_count,dict(stream.segments))
        with self.assertRaises(ProtocolError) as caught: stream.apply(advance)
        self.assertEqual(caught.exception.code,'INVALID_RESPONSE')
        self.assertEqual((stream.sequence,stream.text_bytes,stream.word_count,stream.segments),before)
        stream.finish(dict(duration_ms=20000,engine={'id':'inert'},model={'id':'bytes'},language='en',
                           segments=[first['segment']]))
        empty = SegmentStream('transcribe')
        empty.apply(update(stable=True,end=20000,text=''))
        empty.apply(update(1,1,stable=True,end=21000,text=''))
        empty.finish(dict(duration_ms=21000,engine={'id':'inert'},model={'id':'bytes'},language='en',
                          segments=[_segment_payload(s) for s in empty.segments.values()]))

    def test_emitter_refuses_before_writing_the_advance(self):
        from kilix_transcribe.engine_stream import Emitter
        emitter = Emitter('transcribe'); output = io.BytesIO()
        with patch.object(sys,'stdout',SimpleNamespace(buffer=output)):
            emitter.emit(0,0,20000,'.',True,(WordTiming(20000,20000,'.'),))
            before = output.getvalue()
            with self.assertRaises(ProtocolError): emitter.emit(1,20000,21000,'next',True)
        self.assertEqual(output.getvalue(),before)

    def check_route(self, route):
        import kilix_transcribe.service as service
        import kilix_transcribe.runtime as runtime
        history = eof_history(); observed = []; caught = None
        with word_provider([],21000) as fixture:
            if route == 'peer':
                original = service.send_packet; sent = False
                def send(channel,value,descriptor=None):
                    nonlocal sent
                    if value['type'] != 'update': return original(channel,value,descriptor)
                    if not sent:
                        sent = True
                        for event in history:
                            modified = copy.deepcopy(value); modified['result'] = event
                            original(channel,modified,descriptor)
                guard = patch.object(service,'send_packet',send)
            else:
                private = fixture.root/'owned-eof-worker'; private.mkdir()
                source = Path(runtime.__file__)
                (private/'supervisor.py').write_bytes(source.with_name('supervisor.py').read_bytes())
                marker = private/'identity.json'
                script = ('import json,os,pathlib,sys,time\n'
                          f'sys.path.insert(0,{str(source.parents[1])!r})\n'
                          'from kilix_transcribe.streaming import worker_frame\n'
                          'sys.stdin.buffer.read(65537)\n'
                          f'pathlib.Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),start_ticks=int(pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]))))\n'
                          f'for value in {history!r}:\n sys.stdout.buffer.write(worker_frame(dict(type="update",result=value)));sys.stdout.buffer.flush()\n'
                          'time.sleep(120)\n')
                (private/'worker.py').write_text(script)
                guard = patch.object(runtime,'__file__',str(private/'runtime.py'))
            value = fixture.value(timeout=3); value['args']['stream']='segments-v1'
            started = time.monotonic()
            try:
                with guard,fixture.audio.open('rb') as audio:
                    client_request(fixture.ipc,value,audio.fileno(),on_update=observed.append)
            except Exception as error: caught = error
            finally:
                until(lambda: not fixture.service._jobs)
                identities = [json.loads(p.read_text()) for p in fixture.installation.glob('word-engine-*.json')]
                if route == 'worker': identities.append(json.loads(marker.read_text()))
                self.assertTrue(all(not alive(identity) for identity in identities))
                print(json.dumps(dict(eof_route=route,matching_audio_ms=21000,code=getattr(caught,'code',None),
                    callbacks=len(observed),seconds=time.monotonic()-started,identities=identities,owned_reaped=True)),flush=True)
            self.assertIsInstance(caught,ProtocolError)
            self.assertEqual(caught.code,'INVALID_RESPONSE')
            self.assertEqual(len(observed),1)

    def test_actual_peer_refuses_before_callback(self): self.check_route('peer')
    def test_actual_worker_refuses_before_callback_and_timeout(self): self.check_route('worker')
