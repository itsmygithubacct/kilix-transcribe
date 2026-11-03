"""Actual adapter, immutable stable events and original-timing regressions."""
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from kilix_transcribe import engine_stream
from kilix_transcribe.streaming import SegmentStream
from kilix_transcribe.surface import _segment_payload
from kilix_transcribe.worker import _words
from ownership_cases import cases


def replay(case):
    captured = io.BytesIO()
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        audio = root/'input.wav'
        with wave.open(str(audio), 'wb') as writer:
            writer.setparams((1,2,16000,0,'NONE','not compressed'))
            writer.writeframes(b'\x01\x00' * (case['duration_ms']*16))
        def infer(command, pass_fds, **options):
            identity = options['segment_id']
            calls.append((options['base'],options['limit']))
            options['emitter'].emit(identity,options['start'],options['end'],'preview')
            payload = dict(result=dict(language='en'),transcription=[dict(tokens=[
                dict(text=text,offsets={'from':start-options['base'],'to':end-options['base']})
                for text,start,end in case['windows'][identity]])])
            if not case['windows'][identity]:
                payload['transcription'] = []
            path = Path(command[command.index('--output-file')+1]).with_suffix('.json')
            path.write_text(json.dumps(payload))
        with patch.object(engine_stream,'_infer',infer), \
                patch.object(engine_stream.sys,'stdout',SimpleNamespace(buffer=captured)):
            _,segments = engine_stream.rolling_segments(audio,root,case['duration_ms']*16,
                ['inert','--file',str(audio),'--output-file',str(root/'result')],(),'transcribe',_words)
    frames=[];data=captured.getvalue();cursor=0
    state=SegmentStream('transcribe')
    while cursor<len(data):
        size,=struct.unpack('!I',data[cursor:cursor+4]);cursor+=4
        frame=json.loads(data[cursor:cursor+size]);cursor+=size
        assert frame['type']=='update'
        state.apply(frame['result']);frames.append(frame['result'])
    assert cursor==len(data)
    state.finish(dict(duration_ms=case['duration_ms'],engine={'id':'inert'},model={'id':'inert'},
                      language='en',segments=[_segment_payload(s) for s in segments]))
    return segments,frames,calls


class WordReconciliationTests(unittest.TestCase):
    def test_frozen_boundary_matrix_and_semantic_cases(self):
        for case in cases():
            with self.subTest(case=case['name']):
                segments,frames,calls=replay(case)
                self.assertEqual(' '.join(s.text for s in segments if s.text),case['expected'])
                self.assertEqual(calls,[(0,5000),(0,22000),(18000,24000)])
                stable={}
                for frame in frames:
                    segment=frame['segment'];identity=segment['segment_id']
                    self.assertNotIn(identity,stable,'a stable segment was revised')
                    if segment['stable']:stable[identity]=segment
                self.assertEqual(list(stable.values()),[_segment_payload(s) for s in segments])
                # Every public word timing must come from a complete observed
                # unit in one of the full windows, after physical-end clipping.
                observed=set()
                for index,tokens in enumerate(case['windows']):
                    limit=min(case['duration_ms'],(index+1)*20000+2000)
                    native=[dict(text=t,offsets={'from':min(a,limit),'to':min(b,limit)}) for t,a,b in tokens]
                    for word in _words(native,0,case['duration_ms']):
                        observed.add((word.text,word.start_ms,word.end_ms))
                for segment in segments:
                    for word in segment.words:
                        self.assertIn((word.text,word.start_ms,word.end_ms),observed)


if __name__=='__main__':unittest.main()
