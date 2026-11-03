"""Inert whole-word observations, independent of any transcript corpus."""
from copy import deepcopy


def cases():
    rows = []
    shifts = (-600, -200, 0, 200, 600)
    for left in shifts:
        for right in shifts:
            previous = [(' we',17500,17800),(' saw',18300,18600),(' a',18600,18700),
                        (' bright',18800,19200),(' red',19900+left,20100+left),
                        (' bird',20800,21000),(' outside',21500,21600)]
            current = previous[1:4] + [(' red',19900+right,20100+right)] + previous[-2:]
            rows.append(dict(name=f'shift_{left}_{right}', windows=[previous,current],
                             expected='we saw a bright red bird outside', duration_ms=24000))
            previous = [(' we',18050,18200),(' heard',18300,18600),(' a',18600,18800),
                        (' very',19600+left,19700+left),(' very',20050+left,20150+left),
                        (' loud',21000,21200),(' sound',21400,21600)]
            current = previous[:3] + [(' very',19600+right,19700+right),
                                     (' very',20050+right,20150+right)] + previous[-2:]
            rows.append(dict(name=f'double_{left}_{right}', windows=[previous,current],
                             expected='we heard a very very loud sound', duration_ms=24000))
    anchor = [(' we',17500,17800),(' saw',18300,18600),(' a',18700,18800),(' bright',18900,19200)]
    for pieces, text in [((' ca','fé'),'café'),((' 日本','語'),'日本語'),
                         ((' e','\u0301'),'e\u0301'),((' 😀','\ufe0f'),'😀\ufe0f'),
                         ((' micro','phone','!'),'microphone!')]:
        previous = anchor + [(piece,20100+i*30,20130+i*30) for i,piece in enumerate(pieces)] + [(' bird',20300,20500)]
        current = anchor[1:] + [(' '+text,19600,19900),(' bird',20300,20500)]
        rows.append(dict(name='whole_'+text,windows=[previous,current],
                         expected='we saw a bright '+text+' bird',duration_ms=24000))
    rows.append(dict(name='crossing_next_word',windows=[
        anchor+[(' red',20100,20200),(' bird',20200,20500),(' outside',20800,21000)],
        anchor[1:]+[(' red',19600,19900),(' bird',19990,20210),(' outside',20800,21000)]],
        expected='we saw a bright red bird outside',duration_ms=24000))
    rows.append(dict(name='unavailable_ordered_timing',windows=[
        anchor+[(' red',20100,20200),(' fox',20200,20500)],
        anchor[1:]+[(' red',19600,19900),(' bird',19990,20210)]],
        expected='we saw a bright bird',duration_ms=24000,ordinary_fallback=True))
    rows.append(dict(name='changed_lookahead',windows=[
        anchor+[(' blue',20100,20400)],anchor[1:]+[(' red',19600,19900),(' bird',20300,20500)]],
        expected='we saw a bright bird',duration_ms=24000,ordinary_fallback=True))
    rows.append(dict(name='ambiguous_phrase',windows=[
        [(' near',18100,18200),(' the',18300,18400),(' garden',18500,19000),(' red',20100,20400)],
        [(' near',18000,18100),(' the',18100,18200),(' garden',18200,18300),
         (' near',18300,18400),(' the',18400,18500),(' garden',18500,19000),
         (' red',19600,19900),(' bird',20300,20500)]],
        expected='near the garden bird',duration_ms=24000,ordinary_fallback=True))
    rows.append(dict(name='unequal_double_word',windows=[
        anchor+[(' very',19300,19400),(' very',19500,19600),(' red',20100,20400)],
        anchor[1:]+[(' very',19400,19500),(' red',19600,19900),(' bird',20300,20500)]],
        expected='we saw a bright very very bird',duration_ms=24000,ordinary_fallback=True))
    rows.append(dict(name='committed_partial_continuation',windows=[
        [(' a',18200,18300),(' micro',19750,19850),('phone',19850,19950)],
        [('phone',19850,19950),(' next',20100,20300)]],
        expected='a microphone next',duration_ms=24000))
    rows.append(dict(name='final_punctuation',windows=[
        [(' left',19000,19400)], [(' last',23800,24000),('.',24000,24000)]],
        expected='left last.',duration_ms=24000))
    rows.append(dict(name='empty_final_core',windows=[[(' left',19000,19400)],[]],
                     expected='left',duration_ms=24000))
    rows.append(dict(name='separate_equal_words',windows=[
        [(' echo',10000,10500),(' echo',19000,19500)], [(' echo',23000,23500)]],
        expected='echo echo echo',duration_ms=24000))
    rows.append(dict(name='long_indivisible_unit',windows=[
        [(' super',18000,19900),('long',19900,22100)], [(' super',18000,19900),('long',19900,22100)]],
        expected='superlong',duration_ms=24000))
    return deepcopy(rows)
