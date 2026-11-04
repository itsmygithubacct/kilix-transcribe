"""Explicit managed-job extension; not a remote executable measurement."""
from dataclasses import dataclass
from .protocol import ProtocolError

FEATURE = 'managed-job-v1'
MINOR = 2


@dataclass(frozen=True)
class ManagedPeer:
    directory: str
    endpoint: tuple
    credentials: tuple
    service_instance: str


def selection(value):
    if (type(value) is not dict or set(value) != {'feature', 'service_instance'}
            or value['feature'] != FEATURE or type(value['service_instance']) is not str
            or len(value['service_instance']) != 32
            or any(c not in '0123456789abcdef' for c in value['service_instance'])):
        raise ProtocolError('INCOMPATIBLE_PROTOCOL', 'invalid managed-job selection')
    return value


def validate(value, args, request_id, job_id):
    expected = {'feature': FEATURE, 'service_instance': args['attestation']['service_instance'],
                'request_id': request_id, 'job_id': job_id,
                'audio_sha256': args['audio']['sha256']}
    if type(value) is not dict or value != expected:
        raise ProtocolError('INVALID_RESPONSE', 'unbound managed-job evidence')


class ManagedResult(dict):
    """Internal result retains the exact job authority through wire delivery."""
    def __init__(self, value, recheck):
        super().__init__(value)
        self.recheck = recheck
