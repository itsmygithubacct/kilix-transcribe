"""Versioned FD-only diagnostic/annotation seam. No launcher or model resolver.
The production entry refuses until a separately accepted exact profile exists.
The development consumer is useful for inert/source review only.
"""
import array
import hashlib
import json
import math
import os
import struct
import sys
from .fd import Refusal, digest_fd, read_sealed, require, sealed_info
from . import diagnostic_policy
from .numeric_limits import MIN_INTEGER, MAX_INTEGER, parse_integer, parse_float

ALGORITHM = 'f104-diarization-algorithm/2.1'
SCHEMA = 'f104.diarization.diagnostics/2.1'
MAX_SAMPLES = 57600000
MAX_META = 16 << 20
MAX_SCORES = 128 << 20
STATUSES = {'OK','SILENCE','PARTIAL_UNKNOWN','UNIDENTIFIABLE_ACTIVITY','FAILURE'}

def integer(x, low, high):
    require(type(x) is int and low <= x <= high, 'BOUNDED_INTEGER_REQUIRED')

def fields(obj, keys):
    require(type(obj) is dict and set(obj) == set(keys), 'EXACT_SCHEMA_FIELDS_REQUIRED')

def strict_json(data):
    require(len(data) <= MAX_META, 'JSON_BYTE_LIMIT')
    # Bound nesting before the stdlib parser allocates a deeply recursive tree.
    depth = structural_tokens = 0
    quoted = escaped = False
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (123,91):
            depth += 1
            structural_tokens += 1
            require(depth <= 12, 'JSON_DEPTH_LIMIT')
        elif byte in (125,93):
            depth -= 1
            structural_tokens += 1
            require(depth >= 0, 'JSON_DEPTH_INVALID')
        elif byte in (44,58):
            structural_tokens += 1
        require(structural_tokens <= 1000000, 'JSON_TOKEN_LIMIT')
    require(depth == 0 and not quoted, 'JSON_INCOMPLETE')
    def pairs(items):
        obj = {}
        for key,value in items:
            require(key not in obj, 'DUPLICATE_JSON_FIELD')
            obj[key] = value
        return obj
    def constant(_):
        raise Refusal('NONFINITE_JSON_VALUE')
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant,
                          parse_int=parse_integer, parse_float=parse_float)
    except (UnicodeDecodeError,json.JSONDecodeError,RecursionError) as e:
        raise Refusal('MALFORMED_JSON') from e

def cluster(value):
    if value is not None:
        integer(value,0,15)

def span(row, samples):
    for key in ('start2','end2'):
        integer(row[key],0,2*samples)
    require(row['start2'] < row['end2'], 'EMPTY_OR_REVERSED_ACTIVITY')
    duration_ms = samples*1000//16000
    integer(row['start_ms'],0,duration_ms)
    integer(row['end_ms'],0,duration_ms)
    projected_start=min(duration_ms,(row['start2']*1000+31999)//32000)
    projected_end=max(projected_start,min(duration_ms,row['end2']*1000//32000))
    require(row['start_ms'] == projected_start and row['end_ms'] == projected_end, 'INTEGER_TIME_BINDING_MISMATCH')

def validate_native(record, scores_fd):
    fields(record, ('schema','algorithm','state','sample_rate','samples','status','error_code',
        'score_domain','scores','score_bytes','mappings','embedding_rejections','assignments',
        'activity','turns','confidence','calibration_binding'))
    require(record['schema'] == SCHEMA and record['algorithm'] == ALGORITHM and
        record['state'] == 'DEVELOPMENT' and record['sample_rate'] == 16000 and
        record['score_domain'] == 'FINITE_NEURAL_SCORES_UNVERIFIED_DOMAIN', 'UNREVIEWED_NATIVE_CONTRACT')
    integer(record['samples'],1,MAX_SAMPLES)
    integer(record['sample_rate'],16000,16000)
    require(type(record['status']) is str and record['status'] in STATUSES and record['confidence'] is None and
        record['calibration_binding'] is None, 'UNADMITTED_CONFIDENCE_OR_STATUS')
    require(record['error_code'] in (None,'POPULATION_LIMIT','MALFORMED_INPUT','MODEL_ERROR'), 'ERROR_CODE_INVALID')
    require((record['status']=='FAILURE') == (record['error_code'] is not None), 'FAILURE_CODE_REQUIRED')
    for name,cap in (('scores',4096),('mappings',12000),('embedding_rejections',12000),
                     ('assignments',12000),('activity',16384),('turns',16384)):
        require(type(record[name]) is list and len(record[name]) <= cap, 'DIAGNOSTIC_POPULATION_LIMIT')
    integer(record['score_bytes'],0,MAX_SCORES)
    sealed_info(scores_fd, MAX_SCORES, record['score_bytes'])
    offset = 0
    masks = []
    for index,row in enumerate(record['scores']):
        fields(row, ('chunk','offset_samples','frames','classes','byte_offset','byte_count'))
        integer(row['chunk'],0,4095)
        for field in ('offset_samples','frames','classes','byte_offset','byte_count'):
            integer(row[field],0,MAX_SCORES)
        require(row == {'chunk':index,'offset_samples':index*16000,'frames':589,'classes':7,
                       'byte_offset':offset,'byte_count':589*7*4}, 'SCORE_DESCRIPTOR_MISMATCH')
        data = os.pread(scores_fd,row['byte_count'],offset)
        require(len(data) == row['byte_count'] and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',data)),
                'NONFINITE_OR_TRUNCATED_SCORE_PLANE')
        masks.append(diagnostic_policy.decode(data, index, record['samples']))
        offset += row['byte_count']
    require(offset == record['score_bytes'], 'SCORE_POPULATION_MISMATCH')
    if record['status'] != 'FAILURE':
        chunks = 1 if record['samples'] <= 160000 else 1+(record['samples']-160000+15999)//16000
        require(len(record['scores']) == chunks, 'ACTUAL_SEGMENTATION_EVIDENCE_REQUIRED')
    mapping = {}
    for row in record['mappings']:
        fields(row, ('chunk','slot','cluster','reason'))
        integer(row['chunk'],0,len(record['scores'])-1)
        integer(row['slot'],0,2)
        cluster(row['cluster'])
        key = row['chunk'],row['slot']
        require(key not in mapping, 'DUPLICATE_LOCAL_MAPPING')
        require(row['reason'] == ('MAPPED' if row['cluster'] is not None else 'NO_VALID_ISOLATED_EMBEDDING'),
                'MAPPING_REASON_MISMATCH')
        mapping[key] = row['cluster']
    eligible = diagnostic_policy.eligible_embeddings(masks, record['samples'])
    require(all(value is None or key in eligible for key,value in mapping.items()),
            'KNOWN_MAPPING_WITHOUT_ELIGIBLE_EMBEDDING')
    rejected = set()
    for row in record['embedding_rejections']:
        fields(row, ('chunk','slot','reason'))
        integer(row['chunk'],0,4095);integer(row['slot'],0,2)
        require((row['chunk'],row['slot']) in mapping and mapping[row['chunk'],row['slot']] is None,
                'REJECTION_MAPPING_MISMATCH')
        require((row['chunk'],row['slot']) not in rejected, 'DUPLICATE_EMBEDDING_REJECTION')
        rejected.add((row['chunk'],row['slot']))
        require(row['reason'] in ('MALFORMED_EMBEDDING_SHAPE','NONFINITE_EMBEDDING',
            'ZERO_OR_DEGENERATE_EMBEDDING','INSUFFICIENT_ISOLATED_SAMPLES'), 'REJECTION_REASON_INVALID')
    assignments = set()
    for row in record['assignments']:
        fields(row, ('chunk','slot','cluster','assigned_cosine_distance','alternative_cosine_distance','margin'))
        integer(row['chunk'],0,4095);integer(row['slot'],0,2);integer(row['cluster'],0,15)
        key = row['chunk'],row['slot']
        require(key in mapping and mapping[key] is not None and mapping[key] == row['cluster'] and
                key not in assignments, 'ASSIGNMENT_MAPPING_MISMATCH')
        assignments.add(key)
        assigned,alternative,margin = (row[k] for k in ('assigned_cosine_distance','alternative_cosine_distance','margin'))
        for value in (assigned,alternative):
            require(value is None or type(value) in (int,float) and 0 <= value <= 2 and math.isfinite(value),
                    'INVALID_COSINE_DISTANCE')
        require((alternative is None) == (margin is None), 'MISSING_ALTERNATIVE_MARGIN_BINDING')
        if margin is not None:
            require(assigned is not None and type(margin) in (int,float) and -2 <= margin <= 2 and math.isfinite(margin) and
                    abs(margin-(alternative-assigned)) <= 1e-12, 'INVALID_ASSIGNMENT_MARGIN')
    require(assignments == {key for key,value in mapping.items() if value is not None}, 'ASSIGNMENTS_REQUIRED')
    for row in record['activity']:
        fields(row, ('chunk','slot','cluster','start2','end2','start_ms','end_ms'))
        integer(row['chunk'],0,4095);integer(row['slot'],0,2);cluster(row['cluster'])
        require((row['chunk'],row['slot']) in mapping and mapping[row['chunk'],row['slot']] == row['cluster'],
                'ACTIVITY_MAPPING_MISMATCH')
        span(row,record['samples'])
    for row in record['turns']:
        fields(row, ('cluster','start2','end2','start_ms','end_ms'))
        integer(row['cluster'],0,15)
        require(row['cluster'] is not None and row['cluster'] in set(mapping.values()), 'TURN_IDENTITY_MISMATCH')
        span(row,record['samples'])
        require(row['start_ms'] < row['end_ms'], 'EMPTY_MILLISECOND_TURN')
    unknown = any(value is None for value in mapping.values())
    if record['status'] == 'SILENCE':
        require(not mapping and not record['activity'] and not record['turns'], 'FALSE_SILENCE')
    elif record['status'] == 'OK':
        require(bool(mapping) and not unknown and bool(record['activity']), 'FALSE_COMPLETE_IDENTITY')
    elif record['status'] == 'UNIDENTIFIABLE_ACTIVITY':
        require(bool(mapping) and all(x is None for x in mapping.values()) and bool(record['activity']) and
                not record['turns'], 'FALSE_UNIDENTIFIABLE_ACTIVITY')
    elif record['status'] == 'PARTIAL_UNKNOWN':
        require(unknown and bool(assignments) and bool(record['activity']), 'FALSE_PARTIAL_IDENTITY')
    active_keys, expected_activity = diagnostic_policy.activity(masks, record['samples'], mapping)
    require(record['status'] != 'SILENCE' or not active_keys, 'FALSE_SILENCE')
    require(set(mapping) == active_keys, 'ACTIVE_MAPPING_POPULATION_MISMATCH')
    supplied_activity = [(row['chunk'],row['slot'],row['cluster'],row['start2'],row['end2'],
                          row['start_ms'],row['end_ms']) for row in record['activity']]
    require(sorted(supplied_activity) == sorted(expected_activity), 'ACTIVITY_POPULATION_MISMATCH')
    diagnostic_policy.validate_turns(record['turns'], masks, record['samples'], mapping)
    return record

def pcm_identity(fd, samples):
    integer(samples,1,MAX_SAMPLES)
    sealed_info(fd,4*MAX_SAMPLES,4*samples)
    for offset in range(0,4*samples,65536):
        data = os.pread(fd,min(65536,4*samples-offset),offset)
        require(all(math.isfinite(x[0]) and abs(x[0]) <= 1 for x in struct.iter_unpack('<f',data)),
                'NORMALIZED_FINITE_PCM_REQUIRED')
    return digest_fd(fd,4*MAX_SAMPLES,4*samples)

def _asr_json_string_size(value):
    require(len(value) <= MAX_META, 'ASR_BYTE_LIMIT')
    size = 2  # JSON quotes
    for char in value:
        code = ord(char)
        if char in ('"', '\\'):
            size += 2
        elif code < 0x20:
            size += 2 if char in ('\b','\t','\n','\f','\r') else 6
        elif code < 0x80:
            size += 1
        elif code <= 0xffff:
            size += 6
        else:
            size += 12  # ensure_ascii emits a surrogate pair
        require(size <= MAX_META, 'ASR_BYTE_LIMIT')
    return size

def _preflight_asr_json_tree(value):
    """Bound every value before json.dumps allocates an encoded copy."""
    encoded = 0
    structural = 0
    nodes = 0

    def add_bytes(count):
        nonlocal encoded
        require(0 <= count <= MAX_META - encoded, 'ASR_BYTE_LIMIT')
        encoded += count

    def visit(item, depth):
        nonlocal structural, nodes
        require(depth <= 12, 'ASR_JSON_DEPTH_LIMIT')
        nodes += 1
        require(nodes <= 1000000, 'ASR_JSON_TOKEN_LIMIT')
        kind = type(item)
        if kind is dict:
            require(depth < 12, 'ASR_JSON_DEPTH_LIMIT')
            size = len(item)
            tokens = 2 + size + max(0, size - 1)  # braces, colons and commas
            structural += tokens
            require(structural <= 1000000, 'ASR_JSON_TOKEN_LIMIT')
            add_bytes(2 + max(0, size - 1) + size)  # braces, commas, colons
            for key, child in item.items():
                require(type(key) is str, 'ASR_JSON_KEY_INVALID')
                add_bytes(_asr_json_string_size(key))
                visit(child, depth + 1)
        elif kind is list:
            require(depth < 12, 'ASR_JSON_DEPTH_LIMIT')
            size = len(item)
            structural += 2 + max(0, size - 1)  # brackets and commas
            require(structural <= 1000000, 'ASR_JSON_TOKEN_LIMIT')
            add_bytes(2 + max(0, size - 1))
            for child in item:
                visit(child, depth + 1)
        elif kind is str:
            add_bytes(_asr_json_string_size(item))
        elif kind is int:
            require(MIN_INTEGER <= item <= MAX_INTEGER, 'ASR_INTEGER_RANGE')
            add_bytes(len(str(item)))
        elif kind is float:
            require(math.isfinite(item), 'ASR_NONFINITE_NUMBER')
            add_bytes(len(repr(item)))
        elif kind is bool:
            add_bytes(4 if item else 5)
        elif item is None:
            add_bytes(4)
        else:
            raise Refusal('ASR_JSON_VALUE_INVALID')

    visit(value, 0)
    return encoded

def asr_identity(asr):
    # Wire source is an immutable JSON snapshot supplied by the accepted ASR
    # owner. The adapter copies annotations separately and never rewrites times.
    require(type(asr) is dict and type(asr.get('segments')) is list and len(asr['segments']) <= 16384,
            'IMMUTABLE_ASR_SCHEMA_REQUIRED')
    for row in asr['segments']:
        require(type(row) is dict and type(row.get('start_ms')) is int and
                type(row.get('end_ms')) is int and 0 <= row['start_ms'] <= row['end_ms'] <= MAX_INTEGER,
                'ASR_TIME_INVALID')
    _preflight_asr_json_tree(asr)
    data = json.dumps(asr,sort_keys=True,separators=(',',':'),allow_nan=False,
                      ensure_ascii=True).encode('ascii')
    require(len(data) <= MAX_META, 'ASR_BYTE_LIMIT')
    return hashlib.sha256(data).hexdigest()

class ProductionAdapter:
    def run(self, *args, **kwargs):
        raise Refusal('NO_ADMITTED_EXACT_DIARIZATION_PROFILE')

class DevelopmentFdAdapter:
    """Explicit-session diagnostic consumer; no inference or child execution.
    Owner object/capability is an existing authenticated broker seam, not a
    client-supplied licence/profile/receipt flag. Inputs here are synthetic only.
    """
    def __init__(self):
        self._sessions = {}

    def open_session(self, owner, session):
        require(type(session) is str and 1 <= len(session) <= 128 and session not in self._sessions,
                'FRESH_EXPLICIT_SESSION_REQUIRED')
        self._sessions[session] = owner

    def close_session(self, owner, session):
        require(session in self._sessions and self._sessions[session] is owner, 'SESSION_OWNER_MISMATCH')
        del self._sessions[session]

    def consume_synthetic(self, owner, session, pcm_fd, metadata_fd, scores_fd, immutable_asr):
        require(session in self._sessions and self._sessions[session] is owner, 'SESSION_OWNER_MISMATCH')
        require(len({pcm_fd,metadata_fd,scores_fd}) == 3, 'DISTINCT_INPUT_FDS_REQUIRED')
        record = validate_native(strict_json(read_sealed(metadata_fd,MAX_META)),scores_fd)
        pcm_sha = pcm_identity(pcm_fd,record['samples'])
        asr_sha = asr_identity(immutable_asr)
        # Private session labels never become persistent speaker identities.
        turns = [{'speaker':f'{session}:speaker{row["cluster"]}',
                  'start2':row['start2'],'end2':row['end2'],
                  'start_ms':row['start_ms'],'end_ms':row['end_ms'],
                  'confidence':None,'state':'DEVELOPMENT'} for row in record['turns']]
        unknown = [row.copy() for row in record['activity'] if row['cluster'] is None]
        require(asr_identity(immutable_asr) == asr_sha, 'ASR_CHANGED')
        return {'schema':'f104.diarization.annotation/2.1','session':session,'qualified':False,
                'status':record['status'],'sample_rate':16000,'samples':record['samples'],
                'pcm_sha256':pcm_sha,'diagnostics_sha256':digest_fd(metadata_fd,MAX_META),
                'scores_sha256':digest_fd(scores_fd,MAX_SCORES),'immutable_asr_sha256':asr_sha,
                'turns':turns,'unidentified_activity':unknown,'calibration_binding':None}
