# -*- coding: utf-8 -*-
"""
generate_reasoning_dataset.py

Reproducibly generates the synthetic comparative/transitive reasoning
finetuning corpus for Model H (Hindi) and Model L (Nepali).

Usage:
    python generate_reasoning_dataset.py --lang hi --out_dir outputs/hi \
        --n_train 3500 --n_val 500 --n_test 800 --seed 42
    python generate_reasoning_dataset.py --lang ne --out_dir outputs/ne \
        --n_train 3500 --n_val 500 --n_test 800 --seed 42

Design summary (full writeup in README.md):
  - Every fact is generated from sampled integers, so the ground-truth
    label is known by construction (never scraped or model-labelled).
  - Four question families, five question-type ids:
      direct_value_more / direct_value_less  (2 entities, numeric values given)
      transitive_relation                    (3 entities, pure relation chain,
                                               no numbers - the "A > B, B > C" case)
      order_3way_max / order_3way_min        (3 entities, numeric values given,
                                               find the superlative)
  - Leakage control: entity names, item nouns, and template phrasings are
    each independently split 70/15/15 into train/val/test pools (see
    vocab_hi.py / vocab_ne.py). Train/val/test each draw entities and
    templates ONLY from their own pool, so test measures generalization
    to both unseen entities and unseen surface phrasing - not just new
    value combinations of familiar names/wordings.
"""

import argparse
import json
import os
import random

import vocab_hi
import vocab_ne

LANGS = {"hi": vocab_hi, "ne": vocab_ne}

QTYPE_FAMILY_WEIGHTS = {
    "direct_value_more": 0.30,
    "direct_value_less": 0.20,
    "transitive_relation": 0.25,
    "order_3way_max": 0.125,
    "order_3way_min": 0.125,
}
FAMILIES = list(QTYPE_FAMILY_WEIGHTS.keys())
FAMILY_W = list(QTYPE_FAMILY_WEIGHTS.values())

P_EQUAL = 0.08  # prob. a direct_value_* example uses equal values (tests the "equal" label)


def pick_entities(vocab, domain, split, k, rng):
    pool = (vocab.PERSON_NAMES_SPLIT if domain == "person"
            else vocab.ITEM_NAMES_SPLIT)[split]
    return rng.sample(pool, k)


def sample_distinct_values(attr, k, rng):
    lo, hi = attr["lo"], attr["hi"]
    vals, tries = [], 0
    while len(vals) < k:
        v = rng.randint(lo, hi)
        if v not in vals:
            vals.append(v)
        tries += 1
        if tries > 300:
            vals.append(rng.randint(lo, hi))  # fallback if range too small
    return vals


def gen_direct_value(vocab, split, rng, direction):
    """direction: 'more' or 'less' -> which comparative the question asks for"""
    qtype = f"direct_value_{direction}"
    attr = rng.choice(vocab.ATTRIBUTES)
    e1, e2 = pick_entities(vocab, attr["domain"], split, 2, rng)
    if rng.random() < P_EQUAL:
        v1 = rng.randint(attr["lo"], attr["hi"])
        v2 = v1
    else:
        v1, v2 = sample_distinct_values(attr, 2, rng)
    template = rng.choice(vocab.TEMPLATES_SPLIT[qtype][split])
    text = template.format(e1=e1, e2=e2, v1=v1, v2=v2,
                            attr=attr["name"], unit=attr["unit"])
    if v1 == v2:
        label = vocab.EQUAL_LABEL
        answer = vocab.answer_sentence_equal(attr["name"])
    else:
        if direction == "more":
            winner = e1 if v1 > v2 else e2
        else:
            winner = e1 if v1 < v2 else e2
        label = winner
        answer = vocab.answer_sentence_pair(winner, attr["name"], direction)
    meta = {"attribute": attr["name"], "values": {e1: v1, e2: v2}}
    return qtype, text, answer, label, meta


def gen_transitive_relation(vocab, split, rng):
    qtype = "transitive_relation"
    attr = rng.choice(vocab.ATTRIBUTES)
    e1, e2, e3 = pick_entities(vocab, attr["domain"], split, 3, rng)
    tmpl = rng.choice(vocab.TEMPLATES_SPLIT[qtype][split])
    text = tmpl["text"].format(e1=e1, e2=e2, e3=e3, attr=attr["name"])

    # chain "gt": e1 > e2 > e3 (so e1 largest, e3 smallest)
    # chain "lt": e1 < e2 < e3 (so e3 largest, e1 smallest)
    largest, smallest = (e1, e3) if tmpl["chain"] == "gt" else (e3, e1)

    if tmpl["ask"] == "max":
        label, answer = largest, vocab.answer_sentence_single(largest, attr["name"], "max")
    elif tmpl["ask"] == "min":
        label, answer = smallest, vocab.answer_sentence_single(smallest, attr["name"], "min")
    else:  # "cross": e1 vs e3, "who is more" -> answer is whichever is larger
        label, answer = largest, vocab.answer_sentence_single(largest, attr["name"], "max")

    meta = {"attribute": attr["name"], "chain": tmpl["chain"],
            "order_high_to_low": [largest, e2, smallest]}
    return qtype, text, answer, label, meta


def gen_order_3way(vocab, split, rng, superlative):
    """superlative: 'max' or 'min'"""
    qtype = f"order_3way_{superlative}"
    attr = rng.choice(vocab.ATTRIBUTES)
    e1, e2, e3 = pick_entities(vocab, attr["domain"], split, 3, rng)
    v1, v2, v3 = sample_distinct_values(attr, 3, rng)
    template = rng.choice(vocab.TEMPLATES_SPLIT[qtype][split])
    text = template.format(e1=e1, e2=e2, e3=e3, v1=v1, v2=v2, v3=v3,
                            attr=attr["name"], unit=attr["unit"])
    values = {e1: v1, e2: v2, e3: v3}
    winner = max(values, key=values.get) if superlative == "max" else min(values, key=values.get)
    label = winner
    answer = vocab.answer_sentence_single(winner, attr["name"], superlative)
    meta = {"attribute": attr["name"], "values": values}
    return qtype, text, answer, label, meta


def gen_one(vocab, split, rng):
    family = rng.choices(FAMILIES, weights=FAMILY_W, k=1)[0]
    if family in ("direct_value_more", "direct_value_less"):
        direction = family.split("_")[-1]
        return gen_direct_value(vocab, split, rng, direction)
    elif family == "transitive_relation":
        return gen_transitive_relation(vocab, split, rng)
    else:
        superlative = family.split("_")[-1]
        return gen_order_3way(vocab, split, rng, superlative)


def generate_split(vocab, split, n, rng, lang, start_idx=1):
    records = []
    seen_texts = set()
    idx = start_idx
    attempts = 0
    while len(records) < n and attempts < n * 20:
        attempts += 1
        qtype, text, answer, label, meta = gen_one(vocab, split, rng)
        if text in seen_texts:
            continue  # avoid exact-duplicate prompts within a split
        seen_texts.add(text)
        records.append({
            "id": f"{lang}_{split}_{idx:06d}",
            "split": split,
            "type": qtype,
            "source": "template",
            "prompt": text,
            "answer": answer,
            "label": label,
            "meta": meta,
        })
        idx += 1
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["hi", "ne"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=3500)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vocab = LANGS[args.lang]
    os.makedirs(args.out_dir, exist_ok=True)

    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    stats = {}
    for split, n in sizes.items():
        rng = random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[split])
        records = generate_split(vocab, split, n, rng, args.lang)
        out_path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        type_counts = {}
        for r in records:
            type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
        stats[split] = {"n": len(records), "type_counts": type_counts}
        print(f"[{args.lang}] wrote {len(records)} records -> {out_path}")

    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[{args.lang}] stats -> {stats_path}")


if __name__ == "__main__":
    main()
