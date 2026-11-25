"""Frozen 2.1 score/RF and global aggregation policy, without ML/runtime effects.
Top-k ties retain the inherited native policy: any tied highest-count choice is
permitted; cardinality, support and maximal turn population are mandatory.
"""
import array
import itertools
import math
import struct
from .fd import require

POWERSET = (0, 1, 2, 4, 3, 5, 6)
FRAMES, SHIFT, WINDOW = 589, 16000, 160000
MAX_GLOBAL, MAX_ACTIVITY = 262144, 16384
MIN_ISOLATED_FRAMES, MIN_FRONTEND_SAMPLES = 10, 400

def boundary(frame):
    return 0 if frame == 0 else 991 + (2 * frame - 1) * 270

def keep(available, samples):
    return min(available, max(1, (2 * samples - 722) // 540 + 1))

def projection(start, end, samples):
    duration = samples * 1000 // 16000
    a = min(duration, (start * 1000 + 31999) // 32000)
    return a, max(a, min(duration, end * 1000 // 32000))

def decode(data, chunk, samples):
    local = min(WINDOW, samples - chunk * SHIFT)
    require(local > 0, 'SCORE_DESCRIPTOR_MISMATCH')
    observed = keep(FRAMES, local)
    masks = bytearray(FRAMES)
    for frame, scores in enumerate(struct.iter_unpack('<7f', data)):
        require(all(math.isfinite(value) for value in scores), 'NONFINITE_OR_TRUNCATED_SCORE_PLANE')
        # First maximal class, matching Eigen's existing maxCoeff scan.
        cls = max(range(7), key=scores.__getitem__)
        if frame < observed:
            masks[frame] = POWERSET[cls]
    return masks

def activity(masks, samples, mapping):
    rows, active_keys = [], set()
    for chunk, mask in enumerate(masks):
        local = min(WINDOW, samples - chunk * SHIFT)
        observed = keep(FRAMES, local)
        for slot in range(3):
            start = None
            for frame in range(observed + 1):
                on = frame < observed and mask[frame] & (1 << slot)
                if on and start is None:
                    start = frame
                    active_keys.add((chunk, slot))
                elif not on and start is not None:
                    a = boundary(start) + 2 * chunk * SHIFT
                    b = (2 * local if frame == observed else boundary(frame)) + 2 * chunk * SHIFT
                    require(len(rows) < MAX_ACTIVITY, 'DIAGNOSTIC_POPULATION_LIMIT')
                    rows.append((chunk, slot, mapping.get((chunk, slot)), a, b, *projection(a, b, samples)))
                    start = None
    return active_keys, rows

def eligible_embeddings(masks, samples):
    """Return local pairs the production extractor can actually submit.

    Match ExcludeOverlap/GetChunkSpeakerSampleIndexes: discard frames with
    multiple active slots, require ten isolated RF frames, project each
    contiguous run outward to observed sample bounds, and require enough
    concatenated samples for one snip-edges 400-sample Fbank frame.
    """
    eligible = set()
    for chunk, mask in enumerate(masks):
        local = min(WINDOW, samples - chunk * SHIFT)
        require(local > 0, 'SCORE_DESCRIPTOR_MISMATCH')
        observed = keep(FRAMES, local)
        for slot in range(3):
            isolated = 0
            collected = 0
            start = None
            for frame in range(observed + 1):
                bits = mask[frame] if frame < observed else 0
                on = frame < observed and (bits & (1 << slot)) and bits.bit_count() == 1
                if on:
                    isolated += 1
                    if start is None:
                        start = frame
                elif start is not None:
                    start2 = boundary(start)
                    end2 = 2 * local if frame == observed else boundary(frame)
                    # The extractor rounds half-sample boundaries outward and
                    # clips the resulting span to the observed PCM window.
                    a = chunk * SHIFT + start2 // 2
                    b = chunk * SHIFT + (end2 + 1) // 2
                    a = min(samples, max(0, a))
                    b = min(samples, max(0, b))
                    if a < b:
                        collected += b - a
                    start = None
            if isolated >= MIN_ISOLATED_FRAMES and collected >= MIN_FRONTEND_SAMPLES:
                eligible.add((chunk, slot))
    return eligible

def aggregation(masks, samples, mapping):
    if not masks:
        return 0, 0, (), (), ()
    available = ((len(masks) - 1) * SHIFT + 135) // 270 + FRAMES
    require(available <= MAX_GLOBAL, 'DIAGNOSTIC_POPULATION_LIMIT')
    frames = keep(available, samples)
    known = [value for value in mapping.values() if value is not None]
    columns = max(known) + 1 if known else 0
    # Checked finite dimensions precede allocation; this is bounded arithmetic,
    # not a measured whole-process/native/model resource-fit claim.
    require(frames * columns * 2 <= 256 << 20, 'DIAGNOSTIC_POPULATION_LIMIT')
    counts = array.array('H', [0]) * (frames * columns)
    weights = array.array('H', [0]) * frames
    totals = array.array('H', [0]) * frames
    for chunk, mask in enumerate(masks):
        offset = (chunk * SHIFT + 135) // 270
        for local in range(min(FRAMES, frames - offset)):
            frame = offset + local
            bits = mask[local]
            weights[frame] += 1
            totals[frame] += bits.bit_count()
            # ReLabel is binary per cluster in a chunk: co-mapped local slots
            # do not add multiple votes for that same identity.
            clusters = {mapping[(chunk, slot)] for slot in range(3)
                        if bits & (1 << slot) and mapping[(chunk, slot)] is not None}
            for identity in clusters:
                counts[frame * columns + identity] += 1
    return frames, columns, counts, weights, totals

def _frame_index(edge, frames, samples):
    if edge == 2 * samples:
        return frames
    if edge == 0:
        return 0
    delta = edge - 721
    require(delta > 0 and delta % 540 == 0 and delta // 540 < frames, 'TURN_POPULATION_MISMATCH')
    return delta // 540

def validate_turns(turns, masks, samples, mapping):
    frames, columns, counts, weights, totals = aggregation(masks, samples, mapping)
    chosen = bytearray(frames * columns)
    for row in turns:
        identity = row['cluster']
        require(identity < columns, 'TURN_POPULATION_MISMATCH')
        start = _frame_index(row['start2'], frames, samples)
        end = _frame_index(row['end2'], frames, samples)
        require(start < end, 'TURN_POPULATION_MISMATCH')
        for frame in range(start, end):
            index = frame * columns + identity
            require(not chosen[index], 'TURN_POPULATION_MISMATCH')
            chosen[index] = 1
    canonical = []
    for identity in range(columns):
        start = None
        for frame in range(frames + 1):
            on = frame < frames and chosen[frame * columns + identity]
            if on and start is None:
                start = frame
            elif not on and start is not None:
                a, b = boundary(start), 2 * samples if frame == frames else boundary(frame)
                ms = projection(a, b, samples)
                if ms[0] < ms[1]:
                    canonical.append((identity, a, b, *ms))
                start = None
    supplied = [(row['cluster'], row['start2'], row['end2'], row['start_ms'], row['end_ms']) for row in turns]
    require(sorted(supplied) == sorted(canonical), 'TURN_POPULATION_MISMATCH')
    for frame in range(frames):
        votes = counts[frame * columns:(frame + 1) * columns]
        require(weights[frame] > 0, 'GLOBAL_RF_COVERAGE_REQUIRED')
        k = min((2 * totals[frame] + weights[frame]) // (2 * weights[frame]), columns)
        wanted = min(k, sum(value > 0 for value in votes))
        selected = {identity for identity in range(columns) if chosen[frame * columns + identity]}
        def valid(population):
            return len(population) == wanted and all(votes[i] > 0 for i in population) and (
                not population or min(votes[i] for i in population) >=
                max((votes[i] for i in range(columns) if i not in population), default=0))
        if valid(selected):
            continue
        a, b = boundary(frame), 2 * samples if frame + 1 == frames else boundary(frame + 1)
        ms = projection(a, b, samples)
        # Only an isolated exact sub-ms selection can disappear from known
        # millisecond turns. It cannot shorten an adjacent renderable turn.
        missing = wanted - len(selected)
        require(ms[0] == ms[1] and 0 < missing <= 2, 'TURN_AGGREGATION_MISMATCH')
        options = [i for i in range(columns) if votes[i] > 0 and i not in selected and
                   (frame == 0 or not chosen[(frame - 1) * columns + i]) and
                   (frame + 1 == frames or not chosen[(frame + 1) * columns + i])]
        require(any(valid(selected | set(extra)) for extra in itertools.combinations(options, missing)),
                'TURN_AGGREGATION_MISMATCH')
