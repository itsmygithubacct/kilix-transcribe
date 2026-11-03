"""Pure occurrence reconciliation using adjacent observed whole-unit timings."""
from .streaming import CONTEXT_MS

MAX_ALIGNMENT_UNITS = 64


def text(unit):
    return ''.join(token['text'] for token in unit[0]).strip()


def owned(unit, start, end, include_end):
    midpoint = (unit[1] + unit[2]) // 2
    return start <= midpoint < end or include_end and midpoint == end


def _mapping(previous, current, start):
    left = [(i, row['unit']) for i, row in enumerate(previous)
            if row['state'] in ('committed', 'deferred')
            and row['unit'][2] >= start-CONTEXT_MS and row['unit'][1] <= start+CONTEXT_MS]
    right = [(i, unit) for i, unit, eligible in current if eligible
             and unit[2] >= start-CONTEXT_MS and unit[1] <= start+CONTEXT_MS]
    n, m = len(left), len(right)
    if not n or not m or max(n, m) > MAX_ALIGNMENT_UNITS:
        return None
    matches = [[text(a) == text(b) and abs(a[1]-b[1]) <= CONTEXT_MS
                and abs(a[2]-b[2]) <= CONTEXT_MS for _, b in right] for _, a in left]
    forward = [[0]*(m+1) for _ in range(n+1)]
    backward = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            forward[i+1][j+1] = max(forward[i][j+1], forward[i+1][j],
                                     forward[i][j]+int(matches[i][j]))
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            backward[i][j] = max(backward[i+1][j], backward[i][j+1],
                                  backward[i+1][j+1]+int(matches[i][j]))
    length = forward[n][m]
    # Every edge on any optimum is enumerated. Exactly L such edges means
    # one occurrence mapping; do not choose a convenient tie through repeats.
    edges = [(i, j) for i in range(n) for j in range(m) if matches[i][j]
             and forward[i][j]+1+backward[i+1][j+1] == length]
    if length < 3 or len(edges) != length:
        return None
    run = []
    substantial = False
    for i, j in edges:
        a, b = left[i][0], right[j][0]
        if run and (a != run[-1][0]+1 or b != run[-1][1]+1):
            run = []
        run.append((a, b))
        names = [text(previous[x]['unit']) for x, _ in run]
        if len(run) >= 3 and len(set(names)) >= 2 and sum(map(len, names)) >= 8:
            substantial = True
    if not substantial:
        return None
    return {right[j][0]: previous[left[i][0]] for i, j in edges}


def _ordered_path(options):
    paths = [(0, (), [])]
    for variants in options:
        following = []
        for alternate, unit in variants:
            candidates = []
            for cost, preference, chosen in paths:
                if chosen:
                    before = chosen[-1]
                    if unit[1] < before[1]:
                        continue
                    # Changing timing sources must not glue distinct units.
                    if not unit[0][0]['text'][:1].isspace() and unit[1] <= before[2]+500:
                        continue
                candidates.append((cost+alternate, preference+(alternate,), chosen+[unit]))
            if candidates:
                following.append(min(candidates, key=lambda p: p[:2]))
        if not following:
            return None
        paths = following
    return min(paths, key=lambda p: p[:2])[2]


def reconcile(previous, units, observed, removed, start, end, include_end=False,
              selected_indices=None):
    """Return selected units and represented current occurrence indices.

    Whole previous observations provide alternate timings, never invented
    ones. Stable output is only read. Ambiguous or infeasible reconciliation
    returns ordinary midpoint ownership unchanged.
    Optional selected_indices records current content positions even when
    the chosen tokens come from a previous observation; it never guides matching.
    """
    normal = {i for i, unit in enumerate(units) if owned(unit, start, end, include_end)}
    fallback = ([unit for i, unit in enumerate(units) if i in normal], normal)
    def ordinary():
        if selected_indices is not None:
            selected_indices.extend(i for i in range(len(units)) if i in normal)
        return fallback
    if not previous:
        return ordinary()
    visible = []
    whole = {}
    for i, unit in enumerate(observed):
        identities = {id(token) for token in unit[0]}
        eligible = bool(identities) and (not identities & removed or identities <= removed)
        visible.append((i, unit, eligible))
        whole[tuple(id(token) for token in unit[0])] = i
    matched = _mapping(previous, visible, start)
    if matched is None:
        return ordinary()
    mapping = {i: matched[whole[key]] for i, unit in enumerate(units)
               if (key := tuple(id(token) for token in unit[0])) in whole
               and whole[key] in matched}
    represented, options, positions = set(), [], []
    for i, unit in enumerate(units):
        prior = mapping.get(i)
        if prior is not None and prior['state'] == 'committed':
            represented.add(i)
            continue
        variants = [(0, unit)] if i in normal else []
        if (prior is not None and prior['state'] == 'deferred'
                and owned(prior['unit'], start, end, include_end)
                and prior['unit'] != unit):
            variants.append((1, prior['unit']))
        if variants:
            options.append(variants)
            positions.append(i)
            represented.add(i)
    chosen = _ordered_path(options)
    if chosen is None:
        return ordinary()
    if selected_indices is not None:
        selected_indices.extend(positions)
    return chosen, represented
