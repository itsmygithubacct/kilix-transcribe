"""Owned native text without tokens must survive mixed streaming assembly."""
import json
import unittest

from kilix_transcribe.engine_stream import _final_window
from kilix_transcribe.worker import _words


def timed(text, start, end):
    return {'offsets': {'from': start, 'to': end}, 'text': text.strip(),
            'tokens': [{'text': text, 'offsets': {'from': start, 'to': end}}]}


def untimed(text, start, end):
    return {'offsets': {'from': start, 'to': end}, 'text': text}


def assemble(segments, **options):
    values = dict(base=0, start=0, end=20000, words=_words, window_end=22000)
    values.update(options)
    return _final_window(json.dumps({'result': {'language': 'en'},
                                    'transcription': segments}).encode(), **values)


class MixedNativeSegmentsTests(unittest.TestCase):
    def test_text_only_after_selected_tokens_is_not_lost(self):
        _, text, words = assemble([timed(' Issued.', 100, 300),
                                   untimed('Store the record.', 600, 900)])
        self.assertEqual(text, 'Issued. Store the record.')
        self.assertEqual([(w.text, w.start_ms, w.end_ms) for w in words], [('Issued.', 100, 300)])

    def test_leading_middle_and_trailing_text_keep_source_order(self):
        _, text, words = assemble([
            untimed('Heading.', 0, 100), timed(' First.', 200, 300),
            untimed('An aside.', 400, 500), timed(' Last.', 600, 700),
            untimed('Footer.', 800, 900)])
        self.assertEqual(text, 'Heading. First. An aside. Last. Footer.')
        self.assertEqual([w.text for w in words], ['First.', 'Last.'])

    def test_content_is_not_sorted_by_approximate_segment_times(self):
        _, text, words = assemble([timed(' Foreword.', 3000, 3200),
                                   untimed('Then the note.', 500, 800)])
        self.assertEqual(text, 'Foreword. Then the note.')
        self.assertEqual((words[0].start_ms, words[0].end_ms), (3000, 3200))

    def test_untimed_text_prevents_false_continuation_word(self):
        _, text, words = assemble([timed(' poly', 100, 200),
                                   untimed('interlude', 300, 400), timed('mer', 450, 500)])
        self.assertEqual(text, 'poly interlude mer')
        self.assertEqual([(w.text, w.start_ms, w.end_ms) for w in words],
                         [('poly', 100, 200), ('mer', 450, 500)])

    def test_unowned_text_is_not_admitted_to_current_core(self):
        _, text, words = assemble([timed(' Retained.', 100, 300),
                                   untimed('Next core.', 20500, 21000)])
        self.assertEqual(text, 'Retained.')
        self.assertEqual([w.text for w in words], ['Retained.'])

    def test_genuine_repeated_untimed_text_is_not_deduplicated(self):
        _, text, words = assemble([untimed('echo', 0, 100), untimed('echo', 200, 300)])
        self.assertEqual(text, 'echo echo')
        self.assertEqual(words, ())

    def test_recovery_keeps_current_content_position_and_original_time(self):
        previous = []
        assemble([timed(' Gentle', 18200, 18300), timed(' breezes', 18500, 18600),
                  timed(' arrive.', 18800, 18900), timed(' restored', 20100, 20300)],
                 observations=previous)
        _, text, words = assemble([
            timed(' Gentle', 300, 400), timed(' breezes', 600, 700),
            timed(' arrive.', 900, 1000), timed(' restored', 1800, 1900),
            untimed('Untimed note.', 2500, 2700), timed(' Finish.', 3000, 3200)],
            base=18000, start=20000, end=40000, window_end=40000, previous=previous)
        self.assertEqual(text, 'restored Untimed note. Finish.')
        self.assertEqual([(w.text, w.start_ms, w.end_ms) for w in words],
                         [('restored', 20100, 20300), ('Finish.', 21000, 21200)])

    def test_empty_token_lists_use_the_same_text_path(self):
        native = untimed('A note.', 500, 800)
        native['tokens'] = []
        _, text, words = assemble([timed(' Start.', 0, 100), native])
        self.assertEqual(text, 'Start. A note.')
        self.assertEqual([w.text for w in words], ['Start.'])
