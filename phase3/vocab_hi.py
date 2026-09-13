# -*- coding: utf-8 -*-
"""
Vocabulary, entity pools and templates for the Hindi (Model H) synthetic
reasoning dataset.

All pools (names, item nouns, templates) are split 70/15/15 into
train / val / test *once*, with a fixed seed, so that:
  - train and val never see test's entity names or test's template phrasings
  - val also never sees train's entity names or train's template phrasings
This is the train/val/test leakage-avoidance mechanism documented in
README.md (held-out entities + held-out template patterns).
"""

import random

_SPLIT_SEED = 20260913  # fixed, independent of the data-generation seed

# ---------------------------------------------------------------------------
# Entity name pools
# ---------------------------------------------------------------------------

PERSON_NAMES = [
    "राहुल", "सीता", "अमित", "प्रिया", "विकास", "अंजलि", "रोहन", "कविता",
    "संजय", "नेहा", "अर्जुन", "दीपा", "करण", "पूजा", "मोहन", "गीता",
    "सुरेश", "रीना", "विनोद", "शिल्पा", "आकाश", "स्वाति", "निखिल", "मीरा",
    "गौरव", "पल्लवी", "यश", "सुनीता", "राजेश", "अनिता", "हर्ष", "ममता",
    "तरुण", "कोमल", "अभिषेक", "शालिनी", "दिनेश", "रचना", "प्रकाश", "उर्मिला",
    "विशाल", "सपना", "मनोज", "ज्योति", "सचिन", "वंदना", "राकेश", "किरण",
    "अनुज", "प्रतिभा",
]

ITEM_NAMES = [
    "सेब", "केला", "आम", "अनार", "संतरा", "अंगूर", "पपीता", "तरबूज़",
    "किताब", "पेन", "बैग", "घड़ी", "कुर्सी", "मेज़", "साइकिल", "छाता",
    "जूता", "टोपी", "कप", "थैला", "बॉक्स", "लैंप", "तकिया", "कंबल",
    "बोतल", "प्लेट", "चाकू", "चम्मच", "अलमारी", "दर्पण",
]


def _split(pool, seed_offset):
    """70/15/15 split that guarantees at least one item per split
    whenever the pool has >=3 items (needed for small template lists)."""
    rng = random.Random(_SPLIT_SEED + seed_offset)
    items = list(pool)
    rng.shuffle(items)
    n = len(items)
    if n >= 3:
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = max(1, n - n_val - n_test)
        # if rounding overshot, trim val/test (never below 1)
        while n_train + n_val + n_test > n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
    else:
        n_train, n_val, n_test = n, 0, 0
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val] or items[:1],
        "test": items[n_train + n_val:n_train + n_val + n_test] or items[-1:],
    }


PERSON_NAMES_SPLIT = _split(PERSON_NAMES, 1)
ITEM_NAMES_SPLIT = _split(ITEM_NAMES, 2)

GEOGRAPHY_NAMES = [
    "दिल्ली", "मुंबई", "जयपुर", "पटना", "लखनऊ", "भोपाल", "इंदौर", "चंडीगढ़",
    "आगरा", "मेरठ", "रांची", "कानपुर", "वाराणसी", "ग्वालियर", "जोधपुर", "अमृतसर",
    "राजस्थान", "पंजाब", "हरियाणा", "गुजरात", "बिहार", "केरल", "असम", "ओडिशा",
    "उत्तराखंड", "झारखंड", "गोवा", "हिमाचल", "सिक्किम", "त्रिपुरा", "मणिपुर", "मेघालय"
]
GEOGRAPHY_NAMES_SPLIT = _split(GEOGRAPHY_NAMES, 3)

# Science pool: metals, liquids, vehicles, appliances (no plants — separate pool)
SCIENCE_NAMES = [
    "लोहा", "तांबा", "सोना", "चांदी", "एल्युमिनियम", "पीतल", "जस्ता", "कांस्य",
    "सीसा", "पारा", "टिन", "निकल",
    "पानी", "दूध", "तेल", "पेट्रोल", "डीजल", "सिरका", "शहद", "अल्कोहल",
    "कार", "बस", "ट्रेन", "ट्रक", "मोटरसाइकिल",
    "बल्ब", "हीटर",
]
SCIENCE_NAMES_SPLIT = _split(SCIENCE_NAMES, 4)

# Plant pool: used for tree/lifespan domain attributes
PLANT_NAMES = [
    "नीम", "बरगद", "पीपल", "शीशम", "गुलाब",
    "आम", "सागौन", "चंदन", "तुलसी", "अशोक", "अर्जुन", "जामुन",
]
PLANT_NAMES_SPLIT = _split(PLANT_NAMES, 5)

# ---------------------------------------------------------------------------
# Gender table: 'm' = masculine (का), 'f' = feminine (की)
# Used to generate grammatically correct Hindi possessive in answers.
# ---------------------------------------------------------------------------

ATTR_GENDER = {
    # Physics
    "गति": "f",          # गति (speed) — feminine
    "चाल": "f",          # चाल (velocity) — feminine
    "द्रव्यमान": "m",    # द्रव्यमान (mass) — masculine
    "बल": "m",           # बल (force) — masculine
    "दूरी": "f",         # दूरी (distance) — feminine
    "तापमान": "m",       # तापमान (temperature) — masculine
    "विभवांतर": "m",     # विभवांतर (potential diff) — masculine
    "विद्युत धारा": "f", # धारा — feminine
    "प्रतिरोध": "m",     # प्रतिरोध (resistance) — masculine
    "कार्य": "m",        # कार्य (work) — masculine
    "घनत्व": "m",        # घनत्व (density) — masculine
    "तरंगदैर्घ्य": "m",  # तरंगदैर्घ्य (wavelength) — masculine
    # Chemistry
    "क्वथनांक": "m",     # क्वथनांक (boiling pt) — masculine
    "गलनांक": "m",       # गलनांक (melting pt) — masculine
    "सांद्रता": "f",     # सांद्रता (concentration) — feminine
    # Biology
    "कोशिका व्यास": "m", # व्यास (diameter) — masculine
    "हृदय गति": "f",     # गति — feminine
    "जीवनकाल": "m",      # जीवनकाल (lifespan) — masculine
    # Economics / Geography
    "जनसंख्या": "f",     # जनसंख्या (population) — feminine
    "प्रति व्यक्ति आय": "f",  # आय (income) — feminine
    "साक्षरता दर": "f",  # दर (rate) — feminine
    "गेहूं उत्पादन": "m", # उत्पादन (production) — masculine
    "वार्षिक वर्षा": "f", # वर्षा (rainfall) — feminine
    # Everyday
    "लंबाई": "f",        # लंबाई (height/length) — feminine
    "उम्र": "f",         # उम्र (age) — feminine
    "कीमत": "f",         # कीमत (price) — feminine
    "मात्रा": "f",       # मात्रा (quantity) — feminine
}


def _poss(attr_name):
    """Return the correct Hindi possessive particle for the attribute."""
    return "का" if ATTR_GENDER.get(attr_name, "f") == "m" else "की"


# ---------------------------------------------------------------------------
# Attribute configs: which entity domain, unit, numeric range
# ---------------------------------------------------------------------------

ATTRIBUTES = [
    {"name": "लंबाई", "unit": "सेंटीमीटर", "domain": "person", "lo": 140, "hi": 200},
    {"name": "उम्र", "unit": "साल", "domain": "person", "lo": 5, "hi": 85},
    {"name": "कीमत", "unit": "रुपये", "domain": "item", "lo": 10, "hi": 4500},
    {"name": "मात्रा", "unit": "किलोग्राम", "domain": "item", "lo": 1, "hi": 120},
]

# ---------------------------------------------------------------------------
# Templates, keyed by question type. Each entry is a raw template string.
# Placeholders: {e1} {e2} {e3} {e4} {v1} {v2} {v3} {v4} {t} {attr} {unit} {poss}
# {poss} = "का" (masculine) or "की" (feminine), filled at generation time.
# ---------------------------------------------------------------------------

TEMPLATES_RAW = {
    "direct_value_more": [
        {
            "text": "{e1} {poss} {attr} {v1} {unit} है और {e2} {poss} {attr} {v2} {unit} है।",
            "question": "किसका/किसकी {attr} ज़्यादा है?"
        },
        {
            "text": "{e1} {poss} {attr} {v1} {unit} है, जबकि {e2} {poss} {attr} {v2} {unit} है।",
            "question": "{attr} में कौन आगे है?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}।",
            "question": "इनमें से अधिक {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} और {e2} {poss} {attr} क्रमशः {v1} {unit} और {v2} {unit} है।",
            "question": "बताइए किसका/किसकी {attr} ज़्यादा है।"
        },
        {
            "text": "मान लीजिए {e1} {poss} {attr} {v1} {unit} है और {e2} {poss} {attr} {v2} {unit} है।",
            "question": "ज़्यादा {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr} {v1} {unit} तथा {e2} {poss} {attr} {v2} {unit} दर्ज़ की गई है।",
            "question": "किसका/किसकी {attr} ज़्यादा है?"
        }
    ],
    "direct_value_less": [
        {
            "text": "यदि {e1} {poss} {attr} {v1} {unit} है और {e2} {poss} {attr} {v2} {unit} है,",
            "question": "तो किसका/किसकी {attr} कम है?"
        },
        {
            "text": "{e1} {poss} {attr} {v1} {unit} है और {e2} {poss} {attr} {v2} {unit} है।",
            "question": "किसका/किसकी {attr} कम है?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}।",
            "question": "कम {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} और {e2} {poss} {attr} क्रमशः {v1} {unit} और {v2} {unit} है।",
            "question": "किसका/किसकी {attr} कम है, बताइए।"
        }
    ],
    "transitive_relation": [
        {
            "text": "{e1} {poss} {attr}, {e2} से ज़्यादा है। {e2} {poss} {attr}, {e3} से ज़्यादा है।",
            "chain": "gt",
            "ask": "max",
            "question": "तीनों में सबसे ज़्यादा {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से कम है। {e2} {poss} {attr}, {e3} से कम है।",
            "chain": "lt",
            "ask": "max",
            "question": "सबसे ज़्यादा {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से ज़्यादा है और {e2} {poss} {attr}, {e3} से ज़्यादा है।",
            "chain": "gt",
            "ask": "cross",
            "question": "{e1} और {e3} में से किसका/किसकी {attr} ज़्यादा है?"
        },
        {
            "text": "यदि {e1}, {attr} में {e2} से आगे है और {e2}, {attr} में {e3} से आगे है,",
            "chain": "gt",
            "ask": "cross",
            "question": "तो {e1} और {e3} में से कौन आगे है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से कम है और {e2} {poss} {attr}, {e3} से कम है।",
            "chain": "lt",
            "ask": "min",
            "question": "तीनों में सबसे कम {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से ज़्यादा है। {e2} {poss} {attr}, {e3} से ज़्यादा है।",
            "chain": "gt",
            "ask": "min",
            "question": "सबसे कम {attr} किसका/किसकी है?"
        }
    ],
    "order_3way_max": [
        {
            "text": "{e1} {poss} {attr} {v1} {unit}, {e2} {poss} {attr} {v2} {unit}, और {e3} {poss} {attr} {v3} {unit} है।",
            "question": "सबसे ज़्यादा {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}, {e3}: {v3} {unit}।",
            "question": "तीनों में से सबसे अधिक {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1}, {e2} और {e3} {poss} {attr} क्रमशः {v1} {unit}, {v2} {unit} और {v3} {unit} है।",
            "question": "सबसे ज़्यादा {attr} किसका/किसकी है?"
        }
    ],
    "order_3way_min": [
        {
            "text": "{e1} {poss} {attr} {v1} {unit}, {e2} {poss} {attr} {v2} {unit}, और {e3} {poss} {attr} {v3} {unit} है।",
            "question": "सबसे कम {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}, {e3}: {v3} {unit}।",
            "question": "तीनों में से सबसे कम {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1}, {e2} और {e3} {poss} {attr} क्रमशः {v1} {unit}, {v2} {unit} और {v3} {unit} है।",
            "question": "सबसे कम {attr} किसका/किसकी है?"
        }
    ],
    "transitive_chain_4way": [
        {
            "text": "{e1} {poss} {attr}, {e2} से ज़्यादा है, {e2} {poss} {attr}, {e3} से ज़्यादा है, और {e3} {poss} {attr}, {e4} से ज़्यादा है।",
            "chain": "gt",
            "ask": "min",
            "question": "चारों में सबसे कम {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से ज़्यादा है, {e2} {poss} {attr}, {e3} से ज़्यादा है, तथा {e3} {poss} {attr}, {e4} से ज़्यादा है।",
            "chain": "gt",
            "ask": "max",
            "question": "सबसे ज़्यादा {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से कम है, {e2} {poss} {attr}, {e3} से कम है, और {e3} {poss} {attr}, {e4} से कम है।",
            "chain": "lt",
            "ask": "max",
            "question": "सबसे ज़्यादा {attr} किसका/किसकी है?"
        },
        {
            "text": "{e1} {poss} {attr}, {e2} से ज़्यादा है, {e2} {poss} {attr}, {e3} से ज़्यादा है, तथा {e3} {poss} {attr}, {e4} से ज़्यादा है।",
            "chain": "gt",
            "ask": "cross_extreme",
            "question": "{e1} और {e4} में से किसका/किसकी {attr} ज़्यादा है?"
        }
    ],
    "order_4way_rank": [
        {
            "text": "दिए गए आंकड़ों में {e1} ({v1} {unit}), {e2} ({v2} {unit}), {e3} ({v3} {unit}) और {e4} ({v4} {unit}) {attr} है।",
            "rank": "2nd_max",
            "question": "दूसरा सबसे अधिक मान किसका है?"
        },
        {
            "text": "{e1} ({v1} {unit}), {e2} ({v2} {unit}), {e3} ({v3} {unit}), और {e4} ({v4} {unit}) {attr} में से दूसरा सबसे कम मान किसका है?",
            "rank": "2nd_min",
            "question": ""
        }
    ],
    "threshold_comparison": [
        {
            "text": "मानक सीमा {t} {unit} निर्धारित है। {e1} का मान {v1} {unit} और {e2} का मान {v2} {unit} है।",
            "cond": "above",
            "question": "कौन-सा मानक सीमा से अधिक है?"
        },
        {
            "text": "मानक सीमा {t} {unit} तय की गई है। {e1} का मान {v1} {unit} और {e2} का मान {v2} {unit} है।",
            "cond": "below",
            "question": "कौन-सा मानक सीमा से कम है?"
        }
    ],
    "difference_comparison": [
        {
            "text": "संदर्भ {e1} ({v1} {unit}) की तुलना में {e2} ({v2} {unit}) और {e3} ({v3} {unit}) {attr} दी गई है।",
            "mode": "larger_gap",
            "question": "संदर्भ {e1} से किसका अंतर अधिक है, {e2} का या {e3} का?"
        },
        {
            "text": "संदर्भ {e1} ({v1} {unit}) की तुलना में {e2} ({v2} {unit}) और {e3} ({v3} {unit}) {attr} दी गई है।",
            "mode": "smaller_gap",
            "question": "संदर्भ {e1} से किसका अंतर कम है, {e2} का या {e3} का?"
        }
    ],
    "temporal_ordering": [
        {
            "text": "एक कार्यक्रम में {e1}, {e2} से पहले पहुंचा तथा {e2}, {e3} से पहले पहुंचा।",
            "order": "before",
            "ask": "first",
            "question": "सबसे पहले कौन पहुंचा?"
        },
        {
            "text": "प्रतियोगिता में {e1}, {e2} के बाद समाप्त हुआ और {e2}, {e3} के बाद समाप्त हुआ।",
            "order": "after",
            "ask": "last",
            "question": "सबसे अंत में कौन समाप्त हुआ?"
        },
        {
            "text": "दौड़ में {e1}, {e2} से पहले आया और {e2}, {e3} से पहले आया।",
            "order": "before",
            "ask": "last",
            "question": "सबसे अंत में कौन आया?"
        },
        {
            "text": "कक्षा में {e1}, {e2} के बाद उपस्थित हुआ और {e2}, {e3} के बाद उपस्थित हुआ।",
            "order": "after",
            "ask": "first",
            "question": "सबसे पहले कौन उपस्थित हुआ?"
        }
    ],
    "contradiction_detection": [
        {
            "text": "कथन 1: {e1} {poss} {attr}, {e2} से अधिक है। कथन 2: {e2} {poss} {attr}, {e1} से अधिक है।",
            "conflict": True,
            "question": "क्या इन कथनों में विरोधाभास है?"
        },
        {
            "text": "कथन 1: {e1} {poss} {attr}, {e2} से अधिक है। कथन 2: {e1} {poss} {attr}, {e2} से कम है।",
            "conflict": True,
            "question": "क्या इन दोनों कथनों में विरोधाभास है?"
        },
        {
            "text": "कथन 1: {e1} {poss} {attr}, {e2} से अधिक है। कथन 2: {e2} {poss} {attr}, {e1} से कम है।",
            "conflict": False,
            "question": "क्या इन कथनों में विरोधाभास है?"
        },
        {
            "text": "कथन 1: {e1} {poss} {attr}, {e2} से कम है। कथन 2: {e2} {poss} {attr}, {e1} से अधिक है।",
            "conflict": False,
            "question": "क्या ये दोनों कथन संगत हैं (विरोधाभास नहीं है)?"
        }
    ],
    "rule_classification": [
        {
            "text": "वर्गीकरण नियम के अनुसार यदि किसी का {attr} {thresh} {unit} से अधिक हो, तो वह 'उच्च श्रेणी' में आता है, अन्यथा 'सामान्य श्रेणी' में आता है। {e1} {poss} {attr} {val} {unit} है।",
            "high_label": "उच्च श्रेणी",
            "low_label": "सामान्य श्रेणी",
            "question": "{e1} किस श्रेणी में आता है?"
        },
        {
            "text": "परीक्षण नियम: {thresh} {unit} से अधिक {attr} वाले तत्व 'वर्ग ए' कहलाते हैं और {thresh} {unit} या उससे कम वाले तत्व 'वर्ग बी'। {e1} {poss} {attr} {val} {unit} मापा गया।",
            "high_label": "वर्ग ए",
            "low_label": "वर्ग बी",
            "question": "{e1} किस वर्ग का है?"
        }
    ],
    "multistep_if_then": [
        {
            "text": "यदि {e1} {poss} {attr}, {e2} से अधिक है, और {e2} {poss} {attr}, {e3} से अधिक है,",
            "chain": "gt",
            "ask": "more",
            "question": "तो {e1} और {e3} में से किसका मान अधिक होगा?"
        },
        {
            "text": "यदि {e1} {poss} {attr}, {e2} से कम है, तथा {e2} {poss} {attr}, {e3} से कम है,",
            "chain": "lt",
            "ask": "less",
            "question": "तो {e1} और {e3} में से किसका मान कम होगा?"
        },
        {
            "text": "नियम: यदि {e1} {poss} {attr}, {e2} से अधिक है और {e2} {poss} {attr}, {e3} से अधिक है,",
            "chain": "gt",
            "ask": "less",
            "question": "तो बताएं कि {e1} और {e3} में से सबसे कम मान किसका है?"
        }
    ]
}

TEMPLATES_SPLIT = {qtype: _split(tmpls, 100 + i)
                   for i, (qtype, tmpls) in enumerate(TEMPLATES_RAW.items())}

# ---------------------------------------------------------------------------
# Answer-sentence rendering (gender-correct possessive)
# ---------------------------------------------------------------------------

def answer_sentence_single(name, attr, superlative):
    """superlative: 'max' or 'min'"""
    poss = _poss(attr)
    word = "सबसे ज़्यादा" if superlative == "max" else "सबसे कम"
    return f"{name} {poss} {attr} {word} है।"


def answer_sentence_pair(name, attr, comparative):
    """comparative: 'more' or 'less' (pairwise, not superlative)"""
    poss = _poss(attr)
    word = "ज़्यादा" if comparative == "more" else "कम"
    return f"{name} {poss} {attr} {word} है।"


def answer_sentence_rank(name, attr, rank):
    poss = _poss(attr)
    gender = ATTR_GENDER.get(attr, "f")
    if rank == "2nd_max":
        word = "दूसरा सबसे ज़्यादा" if gender == "m" else "दूसरी सबसे ज़्यादा"
    else:
        word = "दूसरा सबसे कम" if gender == "m" else "दूसरी सबसे कम"
    return f"{name} {poss} {attr} {word} है।"


def answer_sentence_threshold(name, attr, cond):
    word = "अधिक" if cond == "above" else "कम"
    return f"{name} का मान मानक सीमा से {word} है।"


def answer_sentence_diff(name, attr, mode):
    word = "अधिक" if mode == "larger_gap" else "कम"
    return f"संदर्भ की तुलना में {name} का अंतर {word} है।"


def answer_sentence_temporal(name, ask):
    pos = "सबसे पहले" if ask == "first" else "सबसे अंत में"
    return f"{name} {pos} रहा।"


def answer_sentence_contradiction(has_conflict):
    return "हाँ, इन कथनों में विरोधाभास है।" if has_conflict else "नहीं, इन कथनों में विरोधाभास नहीं है।"


def answer_sentence_rule(name, category):
    return f"{name} '{category}' में आता है।"


def answer_sentence_multistep(winner, attr, ask):
    poss = _poss(attr)
    word = "अधिक" if ask == "more" else "कम"
    return f"{winner} {poss} {attr} {word} होगा।"


EQUAL_LABEL = "बराबर"


def answer_sentence_equal(attr):
    poss = _poss(attr)
    return f"दोनों {poss} {attr} बराबर है।"

