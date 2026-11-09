# -*- coding: utf-8 -*-
"""
Vocabulary, entity pools and templates for the Nepali (Model L) synthetic
reasoning dataset. Same 70/15/15 held-out split mechanism as vocab_hi.py.
"""

import random

_SPLIT_SEED = 20260913

# ---------------------------------------------------------------------------
# Entity name pools
# ---------------------------------------------------------------------------

PERSON_NAMES = [
    "राम",
    "सीता",
    "हरि",
    "गीता",
    "विकास",
    "अनिता",
    "श्याम",
    "कमला",
    "बिनोद",
    "सुनिता",
    "रमेश",
    "सरिता",
    "दिपक",
    "सन्तोषी",
    "प्रकाश",
    "पूजा",
    "निर्मल",
    "सुजाता",
    "गोपाल",
    "मीना",
    "कृष्ण",
    "लक्ष्मी",
    "मोहन",
    "रीता",
    "सुरेश",
    "बिमला",
    "अर्जुन",
    "सपना",
    "दिलिप",
    "उमा",
    "राजु",
    "कान्छी",
    "यम",
    "सरस्वती",
    "विष्णु",
    "पार्वती",
    "नारायण",
    "रुक्मणी",
    "इन्द्र",
    "गंगा",
    "केदार",
    "जानकी",
    "टेक",
    "देवी",
    "भीम",
    "सुमन",
    "बद्री",
    "कमला देवी",
    "चन्द्र",
    "प्रतिमा",
    "रामबहादुर",
    "श्यामबहादुर",
    "तिलक",
    "माधव",
    "दिनेश",
    "गणेश",
    "सन्तोष",
    "राजन",
    "नविन",
    "सुनील",
    "कमल",
    "राजेश",
    "निता",
    "कविता",
    "सविता",
    "दुर्गा",
    "ललिता",
    "अन्जु",
    "मन्जु",
    "सन्जु",
    "रश्मि",
    "सृजना",
    "प्रमिला",
    "अस्मिता",
    "बिपना"
]

ITEM_NAMES = [
    "स्याउ",
    "केरा",
    "आँप",
    "अनार",
    "सुन्तला",
    "अंगुर",
    "मेवा",
    "तरबुजा",
    "किताब",
    "कलम",
    "झोला",
    "घडी",
    "कुर्सी",
    "टेबल",
    "साइकल",
    "छाता",
    "जुत्ता",
    "टोपी",
    "कप",
    "थैलो",
    "बाकस",
    "बत्ती",
    "सिरानी",
    "कम्बल",
    "बोतल",
    "थाल",
    "चक्कु",
    "चम्चा",
    "दराज",
    "ऐना",
    "कापी",
    "चप्पल",
    "चस्मा",
    "रूमाल",
    "कचौरा",
    "गिलास",
    "डस्ना",
    "कुच्चो",
    "डोरी",
    "ताल्चा",
    "साँचो",
    "ढोका",
    "झ्याल",
    "टेबुल"
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
    "काठमाडौँ",
    "पोखरा",
    "ललितपुर",
    "विराटनगर",
    "भरतपुर",
    "वीरगन्ज",
    "धरान",
    "बुटवल",
    "हेटौँडा",
    "जनकपुर",
    "नेपालगन्ज",
    "धनगढी",
    "भक्तपुर",
    "इटहरी",
    "कास्की",
    "झापा",
    "बागमती",
    "गण्डकी",
    "लुम्बिनी",
    "कोशी",
    "कर्णाली",
    "सुदूरपश्चिम",
    "मधेश",
    "चितवन",
    "रुपन्देही",
    "कैलाली",
    "मोरङ",
    "सुर्खेत",
    "पाल्पा",
    "इलाम",
    "तनहुँ",
    "गोर्खा",
    "काठमाडौं",
    "हेटौंडा",
    "तुलसीपुर",
    "घोराही",
    "सिद्धार्थनगर",
    "टिकापुर",
    "ब्यास",
    "दमक",
    "गुलरिया"
]
GEOGRAPHY_NAMES_SPLIT = _split(GEOGRAPHY_NAMES, 3)

# Science pool: metals, liquids, vehicles, appliances (no plants — separate pool)
SCIENCE_NAMES = [
    "फलाम",
    "तामा",
    "सुन",
    "चाँदी",
    "एल्युमिनियम",
    "पित्तल",
    "जस्ता",
    "काँस",
    "सिसा",
    "पारो",
    "टिन",
    "निकल",
    "पानी",
    "दूध",
    "तेल",
    "पेट्रोल",
    "डिजेल",
    "सिरका",
    "मह",
    "अल्कोहल",
    "गाडी",
    "बस",
    "रेल",
    "ट्रक",
    "मोटरसाइकल",
    "बत्ती",
    "हिटर",
    "पोटासियम",
    "सोडियम",
    "क्याल्सियम",
    "म्याग्नेसियम",
    "आल्मुनियम",
    "जिंक",
    "प्लेटिनम",
    "टाइटेनियम",
    "निकेल",
    "कोबाल्ट",
    "मंगानिज",
    "लिथियम",
    "सिलिकन"
]
SCIENCE_NAMES_SPLIT = _split(SCIENCE_NAMES, 4)

# Plant pool: used for tree / lifespan domain attributes
PLANT_NAMES = [
    "निम",
    "बर",
    "पीपल",
    "सिसौ",
    "गुलाफ",
    "आँप",
    "साल",
    "चन्दन",
    "तुलसी",
    "अशोक",
    "अर्जुन",
    "जामुन",
    "चमेली",
    "कमल",
    "सयपत्री",
    "सूर्यमुखी",
    "लालीगुराँस",
    "मखमली",
    "गोदावरी",
    "सुनगाभा",
    "गुराँस",
    "पिपल",
    "वर",
    "खयर",
    "अम्बा",
    "कटहर"
]
PLANT_NAMES_SPLIT = _split(PLANT_NAMES, 5)

# ---------------------------------------------------------------------------
# Attribute configs
# ---------------------------------------------------------------------------

ATTRIBUTES = [
    {"name": "उचाइ", "unit": "सेन्टिमिटर", "domain": "person", "lo": 140, "hi": 200},
    {"name": "उमेर", "unit": "वर्ष", "domain": "person", "lo": 5, "hi": 85},
    {"name": "मूल्य", "unit": "रुपैयाँ", "domain": "item", "lo": 10, "hi": 4500},
    {"name": "परिमाण", "unit": "किलोग्राम", "domain": "item", "lo": 1, "hi": 120},
]

# ---------------------------------------------------------------------------
# Templates (Nepali uses "को" universally — no gender inflection needed)
# ---------------------------------------------------------------------------

TEMPLATES_RAW = {
    "direct_value_more": [
        {
            "text": "{e1} को {attr} {v1} {unit} छ र {e2} को {attr} {v2} {unit} छ।",
            "question": "कसको {attr} बढी छ?"
        },
        {
            "text": "{e1} को {attr} {v1} {unit} छ, जबकि {e2} को {attr} {v2} {unit} छ।",
            "question": "{attr} मा को अगाडि छ?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}।",
            "question": "यीमध्ये कसको {attr} बढी छ?"
        },
        {
            "text": "{e1} र {e2} को {attr} क्रमशः {v1} {unit} र {v2} {unit} छ।",
            "question": "कसको {attr} बढी छ, भन्नुहोस्।"
        },
        {
            "text": "मानौं {e1} को {attr} {v1} {unit} छ र {e2} को {attr} {v2} {unit} छ।",
            "question": "बढी {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr} {v1} {unit} र {e2} को {attr} {v2} {unit} रेकर्ड गरिएको छ।",
            "question": "कसको {attr} बढी छ?"
        }
    ],
    "direct_value_less": [
        {
            "text": "यदि {e1} को {attr} {v1} {unit} छ र {e2} को {attr} {v2} {unit} छ भने,",
            "question": "कसको {attr} कम छ?"
        },
        {
            "text": "{e1} को {attr} {v1} {unit} छ र {e2} को {attr} {v2} {unit} छ।",
            "question": "कसको {attr} कम छ?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}।",
            "question": "कम {attr} कसको छ?"
        },
        {
            "text": "{e1} र {e2} को {attr} क्रमशः {v1} {unit} र {v2} {unit} छ।",
            "question": "कसको {attr} कम छ, भन्नुहोस्।"
        }
    ],
    "transitive_relation": [
        {
            "text": "{e1} को {attr}, {e2} भन्दा बढी छ। {e2} को {attr}, {e3} भन्दा बढी छ।",
            "chain": "gt",
            "ask": "max",
            "question": "तीनैमध्ये सबैभन्दा बढी {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा कम छ। {e2} को {attr}, {e3} भन्दा कम छ।",
            "chain": "lt",
            "ask": "max",
            "question": "सबैभन्दा बढी {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा बढी छ र {e2} को {attr}, {e3} भन्दा बढी छ।",
            "chain": "gt",
            "ask": "cross",
            "question": "{e1} र {e3} मध्ये कसको {attr} बढी छ?"
        },
        {
            "text": "यदि {e1}, {attr} मा {e2} भन्दा अगाडि छ र {e2}, {attr} मा {e3} भन्दा अगाडि छ भने,",
            "chain": "gt",
            "ask": "cross",
            "question": "{e1} र {e3} मध्ये को अगाडि छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा कम छ र {e2} को {attr}, {e3} भन्दा कम छ।",
            "chain": "lt",
            "ask": "min",
            "question": "तीनैमध्ये सबैभन्दा कम {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा बढी छ। {e2} को {attr}, {e3} भन्दा बढी छ।",
            "chain": "gt",
            "ask": "min",
            "question": "सबैभन्दा कम {attr} कसको छ?"
        }
    ],
    "order_3way_max": [
        {
            "text": "{e1} को {attr} {v1} {unit}, {e2} को {attr} {v2} {unit}, र {e3} को {attr} {v3} {unit} छ।",
            "question": "सबैभन्दा बढी {attr} कसको छ?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}, {e3}: {v3} {unit}।",
            "question": "तीनैमध्ये सबैभन्दा बढी {attr} कसको छ?"
        },
        {
            "text": "{e1}, {e2} र {e3} को {attr} क्रमशः {v1} {unit}, {v2} {unit} र {v3} {unit} छ।",
            "question": "सबैभन्दा बढी {attr} कसको छ?"
        }
    ],
    "order_3way_min": [
        {
            "text": "{e1} को {attr} {v1} {unit}, {e2} को {attr} {v2} {unit}, र {e3} को {attr} {v3} {unit} छ।",
            "question": "सबैभन्दा कम {attr} कसको छ?"
        },
        {
            "text": "{e1}: {v1} {unit}, {e2}: {v2} {unit}, {e3}: {v3} {unit}।",
            "question": "तीनैमध्ये सबैभन्दा कम {attr} कसको छ?"
        },
        {
            "text": "{e1}, {e2} र {e3} को {attr} क्रमशः {v1} {unit}, {v2} {unit} र {v3} {unit} छ।",
            "question": "सबैभन्दा कम {attr} कसको छ?"
        }
    ],
    "transitive_chain_4way": [
        {
            "text": "{e1} को {attr}, {e2} भन्दा बढी, {e2} को {attr}, {e3} भन्दा बढी, र {e3} को {attr}, {e4} भन्दा बढी छ।",
            "chain": "gt",
            "ask": "min",
            "question": "चारैमध्ये सबैभन्दा कम {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा बढी, {e2} को {attr}, {e3} भन्दा बढी, र {e3} को {attr}, {e4} भन्दा बढी छ।",
            "chain": "gt",
            "ask": "max",
            "question": "सबैभन्दा बढी {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा कम, {e2} को {attr}, {e3} भन्दा कम, र {e3} को {attr}, {e4} भन्दा कम छ।",
            "chain": "lt",
            "ask": "max",
            "question": "सबैभन्दा बढी {attr} कसको छ?"
        },
        {
            "text": "{e1} को {attr}, {e2} भन्दा बढी, {e2} को {attr}, {e3} भन्दा बढी, र {e3} को {attr}, {e4} भन्दा बढी छ।",
            "chain": "gt",
            "ask": "cross_extreme",
            "question": "{e1} र {e4} मध्ये कसको {attr} बढी छ?"
        }
    ],
    "order_4way_rank": [
        {
            "text": "दिइएको तथ्याङ्कमा {e1} ({v1} {unit}), {e2} ({v2} {unit}), {e3} ({v3} {unit}) र {e4} ({v4} {unit}) {attr} छ।",
            "rank": "2nd_max",
            "question": "दोस्रो सबैभन्दा बढी मान कसको छ?"
        },
        {
            "text": "{e1} ({v1} {unit}), {e2} ({v2} {unit}), {e3} ({v3} {unit}), र {e4} ({v4} {unit}) {attr} मध्ये दोस्रो सबैभन्दा कम मान कसको छ?",
            "rank": "2nd_min",
            "question": ""
        }
    ],
    "threshold_comparison": [
        {
            "text": "मानक सीमा {t} {unit} तोकिएको छ। {e1} को मान {v1} {unit} र {e2} को मान {v2} {unit} छ।",
            "cond": "above",
            "question": "कुन चाहिँ मानक सीमा भन्दा बढी छ?"
        },
        {
            "text": "मानक सीमा {t} {unit} तोकिएको छ। {e1} को मान {v1} {unit} र {e2} को मान {v2} {unit} छ।",
            "cond": "below",
            "question": "कुन चाहिँ मानक सीमा भन्दा कम छ?"
        }
    ],
    "difference_comparison": [
        {
            "text": "सन्दर्भ {e1} ({v1} {unit}) को तुलनामा {e2} ({v2} {unit}) र {e3} ({v3} {unit}) {attr} दिइएको छ।",
            "mode": "larger_gap",
            "question": "सन्दर्भ {e1} बाट कसको अन्तर बढी छ, {e2} को वा {e3} को?"
        },
        {
            "text": "सन्दर्भ {e1} ({v1} {unit}) को तुलनामा {e2} ({v2} {unit}) र {e3} ({v3} {unit}) {attr} दिइएको छ।",
            "mode": "smaller_gap",
            "question": "सन्दर्भ {e1} बाट कसको अन्तर कम छ, {e2} को वा {e3} को?"
        }
    ],
    "temporal_ordering": [
        {
            "text": "एउटा कार्यक्रममा {e1}, {e2} भन्दा पहिले पुग्यो र {e2}, {e3} भन्दा पहिले पुग्यो।",
            "order": "before",
            "ask": "first",
            "question": "सबैभन्दा पहिले को पुग्यो?"
        },
        {
            "text": "प्रतियोगितामा {e1}, {e2} पछि सकियो र {e2}, {e3} पछि सकियो।",
            "order": "after",
            "ask": "last",
            "question": "सबैभन्दा पछि को सकियो?"
        },
        {
            "text": "दौडमा {e1}, {e2} भन्दा पहिले आयो र {e2}, {e3} भन्दा पहिले आयो।",
            "order": "before",
            "ask": "last",
            "question": "सबैभन्दा पछि को आयो?"
        },
        {
            "text": "कक्षामा {e1}, {e2} पछि उपस्थित भयो र {e2}, {e3} पछि उपस्थित भयो।",
            "order": "after",
            "ask": "first",
            "question": "सबैभन्दा पहिले को उपस्थित भयो?"
        }
    ],
    "contradiction_detection": [
        {
            "text": "भनाइ १: {e1} को {attr}, {e2} भन्दा बढी छ। भनाइ २: {e2} को {attr}, {e1} भन्दा बढी छ।",
            "conflict": True,
            "question": "के यी भनाइहरूमा विरोधाभास छ?"
        },
        {
            "text": "भनाइ १: {e1} को {attr}, {e2} भन्दा बढी छ। भनाइ २: {e1} को {attr}, {e2} भन्दा कम छ।",
            "conflict": True,
            "question": "के यी दुई भनाइहरू बाझिएका छन्?"
        },
        {
            "text": "भनाइ १: {e1} को {attr}, {e2} भन्दा बढी छ। भनाइ २: {e2} को {attr}, {e1} भन्दा कम छ।",
            "conflict": False,
            "question": "के यी भनाइहरूमा विरोधाभास छ?"
        },
        {
            "text": "भनाइ १: {e1} को {attr}, {e2} भन्दा कम छ। भनाइ २: {e2} को {attr}, {e1} भन्दा बढी छ।",
            "conflict": False,
            "question": "के यी दुई भनाइहरू सङ्गत छन् (विरोधाभास छैन)?"
        }
    ],
    "rule_classification": [
        {
            "text": "वर्गीकरण नियम अनुसार यदि कुनै वस्तुको {attr} {thresh} {unit} भन्दा बढी भएमा 'उच्च वर्ग' मा पर्छ, अन्यथा 'सामान्य वर्ग' मा पर्छ। {e1} को {attr} {val} {unit} छ।",
            "high_label": "उच्च वर्ग",
            "low_label": "सामान्य वर्ग",
            "question": "{e1} कुन वर्गमा पर्छ?"
        },
        {
            "text": "परीक्षण नियम: {thresh} {unit} भन्दा बढी {attr} भएको वस्तुलाई 'वर्ग क' र {thresh} {unit} वा कमलाई 'वर्ग ख' भनिन्छ। {e1} को {attr} {val} {unit} भेटियो।",
            "high_label": "वर्ग क",
            "low_label": "वर्ग ख",
            "question": "{e1} कुन वर्गमा पर्छ?"
        }
    ],
    "multistep_if_then": [
        {
            "text": "यदि {e1} को {attr}, {e2} भन्दा बढी छ, र {e2} को {attr}, {e3} भन्दा बढी छ भने,",
            "chain": "gt",
            "ask": "more",
            "question": "{e1} र {e3} मध्ये कसको मान बढी हुन्छ?"
        },
        {
            "text": "यदि {e1} को {attr}, {e2} भन्दा कम छ, र {e2} को {attr}, {e3} भन्दा कम छ भने,",
            "chain": "lt",
            "ask": "less",
            "question": "{e1} र {e3} मध्ये कसको मान कम हुन्छ?"
        },
        {
            "text": "नियम: यदि {e1} को {attr}, {e2} भन्दा बढी छ र {e2} को {attr}, {e3} भन्दा बढी छ भने,",
            "chain": "gt",
            "ask": "less",
            "question": "{e1} र {e3} मध्ये सबैभन्दा कम मान कसको हो?"
        }
    ]
}

TEMPLATES_SPLIT = {qtype: _split(tmpls, 100 + i)
                   for i, (qtype, tmpls) in enumerate(TEMPLATES_RAW.items())}

# ---------------------------------------------------------------------------
# Answer-sentence rendering (Nepali "को" is gender-neutral)
# ---------------------------------------------------------------------------

def answer_sentence_single(name, attr, superlative):
    """Function/Class for answer_sentence_single."""
    word = "सबैभन्दा बढी" if superlative == "max" else "सबैभन्दा कम"
    return f"{name} को {attr} {word} छ।"


def answer_sentence_pair(name, attr, comparative):
    """comparative: 'more' or 'less' (pairwise, not superlative)"""
    word = "बढी" if comparative == "more" else "कम"
    return f"{name} को {attr} {word} छ।"


def answer_sentence_rank(name, attr, rank):
    """Function/Class for answer_sentence_rank."""
    word = "दोस्रो सबैभन्दा बढी" if rank == "2nd_max" else "दोस्रो सबैभन्दा कम"
    return f"{name} को {attr} {word} छ।"


def answer_sentence_threshold(name, attr, cond):
    """Function/Class for answer_sentence_threshold."""
    word = "बढी" if cond == "above" else "कम"
    return f"{name} को मान मानक सीमा भन्दा {word} छ।"


def answer_sentence_diff(name, attr, mode):
    """Function/Class for answer_sentence_diff."""
    word = "बढी" if mode == "larger_gap" else "कम"
    return f"सन्दर्भको तुलनामा {name} को अन्तर {word} छ।"


def answer_sentence_temporal(name, ask):
    """Function/Class for answer_sentence_temporal."""
    pos = "सबैभन्दा पहिले" if ask == "first" else "सबैभन्दा पछि"
    return f"{name} {pos} भयो।"


def answer_sentence_contradiction(has_conflict):
    """Function/Class for answer_sentence_contradiction."""
    return "हो, यी भनाइहरूमा विरोधाभास छ।" if has_conflict else "होइन, यी भनाइहरूमा विरोधाभास छैन।"


def answer_sentence_rule(name, category):
    """Function/Class for answer_sentence_rule."""
    return f"{name} '{category}' मा पर्छ।"


def answer_sentence_multistep(winner, attr, ask):
    """Function/Class for answer_sentence_multistep."""
    word = "बढी" if ask == "more" else "कम"
    return f"{winner} को {attr} {word} हुन्छ।"


EQUAL_LABEL = "बराबर"


def answer_sentence_equal(attr):
    """Function/Class for answer_sentence_equal."""
    return f"दुवैको {attr} बराबर छ।"

