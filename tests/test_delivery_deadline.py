"""Real transcript validation must finish within the original client budget."""
import os
import threading
import time
import unittest
from unittest.mock import patch

from kilix_transcribe import protocol, service
from test_client_control import runtime


class DeliveryDeadlineTests(unittest.TestCase):
    def test_final_delivery_checks_time_after_real_validation(self):
        for mode in ('normal', 'deadline', 'cancel'):
            with self.subTest(mode=mode), runtime() as fixture, fixture.audio.open('rb') as source:
                cancel = threading.Event()
                original = service.decode_result
                budget = 1.0
                began = time.monotonic()
                def validated(*args):
                    value = original(*args)
                    self.assertLess(time.monotonic() - began, budget)
                    if mode == 'deadline':
                        time.sleep(max(0, began + budget + .08 - time.monotonic()))
                    if mode == 'cancel':
                        cancel.set()
                    return value
                request = service.request_value('submit', job_id='delivery-budget',
                                                args=fixture.value()['args'], timeout=budget)
                with patch.object(service, 'decode_result', validated):
                    if mode == 'normal':
                        result = service.client_request(fixture.ipc, request, source.fileno(), cancelled=cancel.is_set)
                        self.assertTrue(result)
                    else:
                        with self.assertRaises(protocol.ProtocolError) as error:
                            service.client_request(fixture.ipc, request, source.fileno(), cancelled=cancel.is_set)
                        self.assertEqual(error.exception.code, 'CANCELED' if mode == 'cancel' else 'DEADLINE_EXCEEDED')
                os.fstat(source.fileno())
                self.assertEqual(service.client_request(fixture.ipc, service.request_value('status'))['provider_state'], 'ready')

    def test_short_control_delivery_retains_its_original_deadline(self):
        with runtime() as fixture:
            original = service.receive_packet
            def received(*args):
                value = original(*args)
                if value[0].get('type') != 'request':
                    time.sleep(.16)
                return value
            with patch.object(service, 'receive_packet', received), self.assertRaises(protocol.ProtocolError) as error:
                service.client_request(fixture.ipc, service.request_value('status', timeout=.1))
            self.assertEqual(error.exception.code, 'DEADLINE_EXCEEDED')
            self.assertEqual(service.client_request(fixture.ipc, service.request_value('status'))['provider_state'], 'ready')


if __name__ == '__main__':
    unittest.main()
