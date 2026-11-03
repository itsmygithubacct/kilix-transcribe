"""Actual rolling adapter regressions with inert, drifting decoder windows."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from kilix_transcribe import engine_stream
from kilix_transcribe.surface import _segment_payload
from kilix_transcribe.worker import _words
from kilix_transcribe.streaming import SegmentStream


def payload(tokens, base):
    return json.dumps(dict(result={'language': 'en'}, transcription=[dict(tokens=[
        {'text': text, 'offsets': {'from': start-base, 'to': end-base}}
        for text, start, end in tokens])])).encode()


class OverlapJitterTests(unittest.TestCase):
    def replay(self, previous, following):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root/'input.wav'
            with wave.open(str(audio), 'wb') as writer:
                writer.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                writer.writeframes(b'\x01\x00' * (24*16000))
            def infer(command, pass_fds, **options):
                base = options['base']
                value = previous if base == 0 else following
                destination = Path(command[command.index('--output-file')+1]).with_suffix('.json')
                destination.write_bytes(payload(value, base))
            with patch.object(engine_stream, '_infer', infer), \
                    patch.object(engine_stream.sys, 'stdout', SimpleNamespace(buffer=io.BytesIO())):
                _, segments = engine_stream.rolling_segments(audio, root, 24*16000,
                    ['inert', '--file', str(audio), '--output-file', str(root/'result')], (), 'transcribe', _words)
            # Preserve the unmodified public ownership and final-binding rules.
            state = SegmentStream('transcribe')
            for index, segment in enumerate(segments):
                state.apply({'sequence': index, 'segment': _segment_payload(segment)})
            state.finish(dict(duration_ms=24000, engine={'id': 'inert'}, model={'id': 'inert'},
                              language='en', segments=[_segment_payload(s) for s in segments]))
            return segments

    def inputs(self, deferred=' red', recovered=' red'):
        previous = [(' we',17500,17800), (' saw',18300,18600), (' a',18700,18800),
                    (' bright',18900,19800), (deferred,20100,20400)]
        following = [(' saw',18350,18650), (' a',18700,18800), (' bright',18800,19600),
                     (recovered,19600,19900), (' bird',20200,20500)]
        return previous, following

    def test_agreed_word_is_not_lost_when_both_windows_defer_it(self):
        segments = self.replay(*self.inputs())
        self.assertEqual([s.text for s in segments], ['we saw a bright', 'red bird'])
        words = [w for s in segments for w in s.words]
        self.assertEqual([w.text for w in words], ['we','saw','a','bright','red','bird'])
        self.assertEqual((words[-2].start_ms, words[-2].end_ms), (20000,20000))
        self.assertEqual((words[-1].start_ms, words[-1].end_ms), (20200,20500))

    def test_changed_lookahead_does_not_add_unagreed_words(self):
        segments = self.replay(*self.inputs(deferred=' blue'))
        self.assertEqual([s.text for s in segments], ['we saw a bright', 'bird'])

    def test_unicode_and_changed_subword_boundaries_keep_whole_text(self):
        previous, following = self.inputs(deferred=' café', recovered=' café')
        previous[-1:] = [(' ca',20100,20200), ('fé',20200,20400)]
        segments = self.replay(previous, following)
        self.assertEqual(segments[-1].text, 'café bird')
        self.assertEqual([w.text for w in segments[-1].words], ['café','bird'])

    def test_ambiguous_repeated_anchor_does_not_guess_an_occurrence(self):
        previous = [(' near',18100,18200), (' the',18300,18400), (' garden',18500,19000), (' red',20100,20400)]
        following = [(' near',18000,18100), (' the',18100,18200), (' garden',18200,18300),
                     (' near',18300,18400), (' the',18400,18500), (' garden',18500,19000),
                     (' red',19600,19900), (' bird',20200,20500)]
        self.assertEqual(self.replay(previous, following)[-1].text, 'bird')

    def test_word_already_owned_by_next_window_is_not_duplicated(self):
        previous, following = self.inputs()
        following[-2] = (' red',20100,20400)
        self.assertEqual(self.replay(previous, following)[-1].text, 'red bird')

if __name__ == '__main__': unittest.main()
