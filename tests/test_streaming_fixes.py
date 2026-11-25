"""Streaming review regressions with inert tools and exact owned processes."""
import ctypes
import copy
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import wave

from test_streaming import provider, until
import test_runtime as fixtures
from kilix_transcribe.service import client_request
from kilix_transcribe.protocol import ProtocolError
from kilix_transcribe.streaming import SegmentStream, MAX_STREAM_TEXT_BYTES
from kilix_transcribe.surface import SegmentUpdate, TranscriptAssembler, _segment_payload


def handle(pid):
    return {'pid': pid, 'start_ticks': int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])}


def alive(identity):
    try:
        return handle(identity['pid']) == identity
    except FileNotFoundError:
        return False


class DiagnosticDeliveryTests(unittest.TestCase):
    def run_cli(self, fixture, arguments, *, closed=False, expected=69):
        reader, writer = os.pipe()
        if closed:
            os.close(reader)
            reader = None
        else:
            capacity = fcntl.fcntl(writer, fcntl.F_GETPIPE_SZ)
            os.set_blocking(writer, False)
            self.assertEqual(os.write(writer, b'x' * capacity), capacity)
            os.set_blocking(writer, True)
        child = subprocess.Popen([sys.executable, '-B', '-m', 'kilix_transcribe', *arguments],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=writer,
                                 env=dict(os.environ, XDG_RUNTIME_DIR=str(fixture.ipc)))
        identity = handle(child.pid)
        libc = ctypes.CDLL(None, use_errno=True)
        pidfd = libc.syscall(434, child.pid, 0)
        self.assertGreaterEqual(pidfd, 0)
        started = time.monotonic()
        try:
            output, _ = child.communicate(timeout=2)
            self.assertEqual(child.returncode, expected)
            self.assertEqual(output, b'')
            self.assertFalse(alive(identity))
            self.assertFalse(fcntl.fcntl(writer, fcntl.F_GETFL) & os.O_NONBLOCK)
            until(lambda: not fixture.service._jobs)
            engines = [json.loads(p.read_text()) for p in fixture.installation.glob('engine-*.json')]
            self.assertTrue(all(not alive(engine) for engine in engines))
            print(json.dumps(dict(case=arguments, closed=closed, seconds=time.monotonic()-started,
                                  exit=child.returncode, cli=identity, engines=engines,
                                  service_jobs=0, owned_reaped=True)), flush=True)
        finally:
            if child.poll() is None:
                self.assertTrue(alive(identity))
                self.assertEqual(libc.syscall(424, pidfd, signal.SIGKILL, 0, 0), 0)
            child.wait(3)
            until(lambda: not fixture.service._jobs)
            engines = [json.loads(p.read_text()) for p in fixture.installation.glob('engine-*.json')]
            self.assertFalse(alive(identity))
            self.assertTrue(all(not alive(engine) for engine in engines))
            print(json.dumps(dict(cli_cleanup=identity, exit=child.returncode,
                                  engines=engines, owned_reaped=True)), flush=True)
            os.close(pidfd)
            child.stdout.close()
            os.close(writer)
            if reader is not None:
                os.close(reader)

    def test_connected_nonreader_cli_exits_without_draining_pipe(self):
        with provider() as fixture:
            self.run_cli(fixture, ['file', str(fixture.audio), '--stream', '--timeout', '1'])

    def test_closed_preview_pipe_exits_and_reaps(self):
        with provider() as fixture:
            self.run_cli(fixture, ['file', str(fixture.audio), '--stream', '--timeout', '1'], closed=True)

    def test_outer_error_delivery_is_bounded(self):
        with provider() as fixture:
            self.run_cli(fixture, ['file', str(fixture.root / 'missing.wav')])
            self.assertFalse(list(fixture.installation.glob('engine-*.json')))

    def test_argparse_diagnostics_are_bounded(self):
        with provider() as fixture:
            self.run_cli(fixture, ['file', '--unknown-option'], expected=2)
            self.assertFalse(list(fixture.installation.glob('engine-*.json')))


WORD_ENGINE = r'''#!/usr/bin/python3
import json,os,pathlib,struct,sys,wave
root=pathlib.Path(FIXTURE_ROOT)
spec=json.loads(pathlib.Path(sys.argv[sys.argv.index('--model')+1]).read_text())
index=len(list(root.glob('word-engine-*.json')))
with wave.open(sys.argv[sys.argv.index('--file')+1],'rb') as w:
    frames=w.getnframes(); first=struct.unpack('<h',w.readframes(1))[0]
base=(first-1)*100; stop=base+(frames*1000+15999)//16000
chosen=[t for t in spec if base<=t['from']<=t['to']<=stop]
record={'pid':os.getpid(),'start_ticks':int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        'base':base,'stop':stop,'tokens':chosen}
(root/f'word-engine-{index:03}.json').write_text(json.dumps(record))
text=''.join(t['text'] for t in chosen)
if text:print('[00:00:00.000 --> 00:00:01.000]  '+text,flush=True)
tokens=[{'text':t['text'],'offsets':{'from':t['from']-base,'to':t['to']-base}} for t in chosen]
out=pathlib.Path(sys.argv[sys.argv.index('--output-file')+1]+'.json')
out.write_text(json.dumps({'result':{'language':'en'},'transcription':[
    {'text':text,'offsets':{'from':0,'to':stop-base},'tokens':tokens}]}))
'''


@contextmanager
def word_provider(tokens, milliseconds=21000):
    fixture = fixtures.RuntimeTests('runTest')
    fixture.setUp()
    try:
        import struct
        count = milliseconds * 16
        pcm = b''.join(struct.pack('<h', i + 1) * min(1600, count-i*1600)
                       for i in range((count+1599)//1600))
        with wave.open(str(fixture.audio), 'wb') as output:
            output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            output.writeframes(pcm)
        with patch.object(fixtures, 'ENGINE', WORD_ENGINE):
            fixture.start(json.dumps(tokens))
        yield fixture
    finally:
        fixture.stop()
        records = [json.loads(p.read_text()) for p in fixture.installation.glob('word-engine-*.json')]
        assert all(not alive({'pid': r['pid'], 'start_ticks': r['start_ticks']}) for r in records)
        print(json.dumps(dict(word_engine_calls=records, owned_reaped=True)), flush=True)
        fixture.doCleanups()


def word_submit(fixture, *, output='text', stream=True):
    value = fixture.value(timeout=5, output=output)
    if stream:
        value['args']['stream'] = 'segments-v1'
    with fixture.audio.open('rb') as audio:
        return client_request(fixture.ipc, value, audio.fileno(),
                              on_update=(lambda _: None) if stream else None)


class WholeUnitOwnershipTests(unittest.TestCase):
    def test_crossing_continuations_preserve_exact_text_and_full_timing(self):
        for pieces, expected in [((' micro', 'phone'), 'microphone'),
                                 ((' 日本', '語'), '日本語'), ((' 😀', '\ufe0f'), '😀\ufe0f')]:
            tokens = [{'text': pieces[0], 'from': 19800, 'to': 20000},
                      {'text': pieces[1], 'from': 20000, 'to': 20200}]
            with self.subTest(expected=expected), word_provider(tokens) as fixture:
                streamed = word_submit(fixture)
                legacy = word_submit(fixture, stream=False)
                self.assertEqual(streamed['text'], legacy['text'])
                self.assertEqual(streamed['text'], expected+'\n')
                self.assertEqual([s['text'] for s in streamed['segments']], ['', expected])
                word = streamed['segments'][1]['words'][0]
                self.assertEqual((word['text'], word['start_ms'], word['end_ms']), (expected, 19800, 20200))
                self.assertEqual(streamed['segments'][1]['start_ms'], 19800)
                records = [json.loads(p.read_text()) for p in fixture.installation.glob('word-engine-*.json')]
                full = [r for r in records if r['stop'] == 21000]
                self.assertEqual(len(full), 3)  # two overlapping full passes and the legacy pass
                self.assertTrue(all(r['tokens'] == tokens for r in full))

    def test_boundary_word_and_eof_punctuation_are_owned_once(self):
        tokens = [{'text':' Left','from':19000,'to':19900},
                  {'text':' Edge','from':19900,'to':20100},
                  {'text':' Right','from':20100,'to':20400},
                  {'text':' Last','from':44900,'to':45000},
                  {'text':'.','from':45000,'to':45000}]
        with word_provider(tokens, 45000) as fixture:
            result = word_submit(fixture)
            self.assertEqual([s['text'] for s in result['segments']], ['Left','Edge Right','Last.'])
            words = [w for s in result['segments'] for w in s['words']]
            self.assertEqual([(w['text'],w['start_ms'],w['end_ms']) for w in words],
                             [('Left',19000,19900),('Edge',19900,20100),('Right',20100,20400),('Last.',44900,45000)])
            records = [json.loads(p.read_text()) for p in sorted(fixture.installation.glob('word-engine-*.json'))]
            self.assertEqual([(r['base'],r['stop']) for r in records],
                             [(0,5000),(0,22000),(18000,25000),(18000,42000),(38000,45000)])

    def test_crossing_word_final_formats_preserve_punctuation(self):
        tokens = [{'text':' micro','from':19800,'to':20000},
                  {'text':'phone','from':20000,'to':20200}, {'text':'!','from':20200,'to':20200}]
        with word_provider(tokens) as fixture:
            for output in ('text', 'json', 'srt', 'webvtt'):
                with self.subTest(output=output):
                    result = word_submit(fixture, output=output)
                    self.assertEqual(result['text'], 'microphone!\n')
                    if output == 'text':
                        self.assertEqual(result['output'], 'microphone!\n')
                    elif output == 'json':
                        self.assertEqual(json.loads(result['output'])['segments'][1]['words'][0]['text'], 'microphone!')
                    else:
                        self.assertEqual(result['output'].count('microphone!'), 1)
                        self.assertIn('00:00:19', result['output'])


def update(sequence=0, identity=0, *, revision=0, stable=False, end=None, text='a', words=None):
    return {'sequence': sequence, 'segment': dict(segment_id=identity, revision=revision,
        start_ms=identity*20000, end_ms=end if end is not None else identity*20000+1000,
        text=text, stable=stable, words=words or [])}


def bad_history(case):
    if case in ('reversed', 'skipped'):
        return [update(identity=1 if case == 'reversed' else 2)]
    if case == 'premature':
        return [update(), update(1, 1)]
    if case == 'terminal':
        return [update(stable=True), update(1, 1)]
    bad = update()
    bad['segment'].update(start_ms=1000, end_ms=2000)
    return [bad]


class RollingCoreValidationTests(unittest.TestCase):
    def test_reversed_skipped_premature_and_terminal_advance_refuse(self):
        for case in ('reversed', 'skipped', 'premature', 'terminal', 'time'):
            with self.subTest(case=case):
                stream = SegmentStream('transcribe')
                history = bad_history(case)
                for event in history[:-1]:
                    stream.apply(event)
                with self.assertRaises(ProtocolError) as caught:
                    stream.apply(history[-1])
                self.assertEqual(caught.exception.code, 'INVALID_RESPONSE')
                self.assertEqual(stream.sequence, len(history)-1)
        stream = SegmentStream('transcribe')
        stream.apply(update(stable=True, end=20000))
        with self.assertRaises(ProtocolError):
            stream.apply(update(1, 2))

    def test_wrong_word_ownership_and_excess_context_refuse(self):
        for event in (update(stable=True, end=20100, words=[{'text':'wrong','start_ms':20000,'end_ms':20100}]),
                      update(identity=0, stable=True, end=20000)):
            if not event['segment']['words']:
                event['segment']['start_ms'] = 1
            with self.subTest(event=event), self.assertRaises(ProtocolError):
                SegmentStream('transcribe').apply(event)
        stream = SegmentStream('transcribe')
        stream.apply(update(stable=True, end=20000))
        event = update(1, 1, stable=True, end=22500, words=[{'text':'wide','start_ms':17500,'end_ms':22500}])
        event['segment']['start_ms'] = 17500
        with self.assertRaises(ProtocolError):
            stream.apply(event)

    def test_revisions_retraction_word_spill_and_final_duration(self):
        stream = SegmentStream('transcribe')
        stream.apply(update(text='draft'))
        stream.apply(update(1, revision=1, text='replacement'))
        stream.apply(update(2, revision=2, stable=True, end=20000, text=''))
        last = update(3, 1, stable=True, end=21000, text='microphone',
                      words=[{'text':'microphone','start_ms':19800,'end_ms':20200}])
        last['segment']['start_ms'] = 19800
        stream.apply(last)
        result = dict(engine={'id':'whisper.cpp'}, model={'id':'inert'}, language='en', duration_ms=21000,
                      segments=[_segment_payload(s) for s in stream.segments.values()])
        stream.finish(result)
        SegmentStream('transcribe').finish(dict(result, segments=[]))  # silent audio
        for duration in (1000, 40001, True):
            with self.subTest(duration=duration):
                state = SegmentStream('transcribe')
                state.apply(update(stable=True, end=20000))
                with self.assertRaises(ProtocolError):
                    state.finish(dict(result, duration_ms=duration, segments=[_segment_payload(state.segments[0])]))

    def test_legal_stable_core_population_retains_aggregate_bound(self):
        stream = SegmentStream('transcribe')
        for identity in range(32):
            stream.apply(update(identity, identity, stable=True, end=(identity+1)*20000, text='x'*32768))
        self.assertEqual(stream.text_bytes, MAX_STREAM_TEXT_BYTES)
        with self.assertRaises(ProtocolError) as caught:
            stream.apply(update(32, 32, stable=True, end=660000, text='x'))
        self.assertEqual(caught.exception.code, 'LIMIT_EXCEEDED')

    def test_worker_emitter_refuses_illegal_core_before_writing(self):
        import io
        from types import SimpleNamespace
        from kilix_transcribe.engine_stream import Emitter
        for case in ('reversed', 'skipped', 'premature', 'terminal', 'time'):
            with self.subTest(case=case):
                emitter = Emitter('transcribe')
                output = io.BytesIO()
                history = bad_history(case)
                with patch.object(sys, 'stdout', SimpleNamespace(buffer=output)):
                    for event in history[:-1]:
                        segment = event['segment']
                        emitter.emit(segment['segment_id'], segment['start_ms'], segment['end_ms'],
                                     segment['text'], segment['stable'])
                    before = output.getvalue()
                    segment = history[-1]['segment']
                    with self.assertRaises(ProtocolError):
                        emitter.emit(segment['segment_id'], segment['start_ms'], segment['end_ms'],
                                     segment['text'], segment['stable'])
                    self.assertEqual(output.getvalue(), before)

    def test_legacy_final_only_ids_remain_arbitrary(self):
        assembler = TranscriptAssembler('transcribe')
        assembler.apply(SegmentUpdate(47, 0, 1000, 2000, 'second', True))
        assembler.apply(SegmentUpdate(3, 0, 0, 1000, 'first', True))
        transcript = assembler.finish(engine_id='inert', model_id='inert', language='en')
        self.assertEqual([segment.segment_id for segment in transcript.segments], [3, 47])


class MalformedRollingRoutesTests(unittest.TestCase):
    def test_actual_client_refuses_illegal_peer_history_before_callback(self):
        import kilix_transcribe.service as service
        from test_streaming import submit
        for case in ('reversed', 'skipped', 'premature', 'terminal', 'time'):
            with self.subTest(case=case), provider() as fixture:
                history = bad_history(case)
                original = service.send_packet
                sent = False
                def malformed(channel, event, descriptor=None):
                    nonlocal sent
                    if event['type'] != 'update':
                        return original(channel, event, descriptor)
                    if not sent:
                        sent = True
                        for item in history:
                            modified = copy.deepcopy(event)
                            modified['result'] = item
                            original(channel, modified, descriptor)
                observed = []
                with patch.object(service, 'send_packet', malformed):
                    with self.assertRaises(ProtocolError) as caught:
                        submit(fixture, on_update=observed.append)
                until(lambda: not fixture.service._jobs)
                engines = [json.loads(p.read_text()) for p in fixture.installation.glob('engine-*.json')]
                self.assertTrue(all(not alive(engine) for engine in engines))
                print(json.dumps(dict(peer_fault=case, code=caught.exception.code, observed_callbacks=len(observed), engines=engines, owned_reaped=True)), flush=True)
                self.assertEqual(caught.exception.code, 'INVALID_RESPONSE')
                self.assertEqual(len(observed), len(history)-1)

    def test_actual_runtime_refuses_owned_malformed_worker_and_reaps(self):
        import kilix_transcribe.runtime as runtime
        from test_streaming import submit
        for case in ('reversed', 'skipped', 'premature', 'terminal', 'time'):
            with self.subTest(case=case), provider() as fixture:
                history = bad_history(case)
                private = fixture.root / 'malformed-worker'
                private.mkdir()
                # Run the exact trusted supervisor with a deliberately bad
                # private worker; never modify the product supervisor.
                source = Path(runtime.__file__)
                (private / 'supervisor.py').write_bytes(source.with_name('supervisor.py').read_bytes())
                marker = private / 'worker.json'
                script = ('import json,os,pathlib,sys,time\n'
                          f'sys.path.insert(0,{str(source.parents[1])!r})\n'
                          'from kilix_transcribe.streaming import worker_frame\n'
                          'sys.stdin.buffer.read(65537)\n'
                          f'pathlib.Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),start_ticks=int(pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]))))\n'
                          f'for value in {history!r}:\n'
                          ' sys.stdout.buffer.write(worker_frame(dict(type="update",result=value)));sys.stdout.buffer.flush()\n'
                          'time.sleep(120)\n')
                (private / 'worker.py').write_text(script)
                observed = []
                with patch.object(runtime, '__file__', str(private / 'runtime.py')):
                    with self.assertRaises(ProtocolError) as caught:
                        submit(fixture, on_update=observed.append, timeout=3)
                until(lambda: not fixture.service._jobs)
                identity = json.loads(marker.read_text())
                self.assertFalse(alive(identity))
                print(json.dumps(dict(worker_fault=case, code=caught.exception.code, observed_callbacks=len(observed), worker=identity, owned_reaped=True)), flush=True)
                self.assertEqual(caught.exception.code, 'INVALID_RESPONSE')
                self.assertEqual(len(observed), len(history)-1)
