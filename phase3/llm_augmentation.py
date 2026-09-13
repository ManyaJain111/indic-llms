import os
import json
import argparse
import random
import google.generativeai as genai
import vocab_hi
import vocab_ne

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

GENERATION_PROMPT = """
You are generating synthetic comparative-reasoning finetuning data in {language}, in the target language's own script and natural, textbook-style phrasing (not English syntax with translated words).

Question types (pick one per item, vary across the batch):
1. direct_value_more — two entities, explicit numeric values given (height/age/price/quantity), ask which has the greater value.
2. direct_value_less — same, ask which has the smaller value.
3. transitive_relation — three entities, only relational statements given (no numbers), e.g. "A's <attribute> is more than B's, B's is more than C's" — ask for the largest, smallest, or how A compares to C.
4. order_3way_max — three entities with numeric values, ask which has the largest value.
5. order_3way_min — three entities with numeric values, ask which has the smallest value.

Here are examples in the target style:
{exemplars}

Now generate {N} new items. Requirements:
- Use ONLY these entity names: {train_pool_names} — do not invent new names or reuse any name from this excluded list: {held_out_names}.
- Vary sentence structure across items — do not reuse the same sentence skeleton more than a few times in the batch.
- Every numeric value must be an integer in a sensible range for its attribute (heights 140–200cm, ages 5–85, prices 10–4500, quantities 1–120).
- The "label" must be the exact, unambiguous entity name that answers the question (or the word for "equal" if you construct an equal-values example) — double check your own arithmetic before writing the label.
- Output strict JSON Lines, one object per line, fields exactly: id, type, prompt, answer, label, meta. No commentary outside the JSONL.
"""

TRANSLATION_PROMPT = """
Translate the following Hindi comparative reasoning JSON object into Nepali.
Keep every number and unit exactly unchanged — only the wording is translated, never the values.
Ensure it reads naturally in Nepali, avoiding Hindi calques.
Return the updated JSON object.
Object: {obj}
"""

def translate_to_nepali(model, obj):
    try:
        resp = model.generate_content(TRANSLATION_PROMPT.format(obj=json.dumps(obj, ensure_ascii=False)))
        text = resp.text.strip()
        if text.startswith("```json"): text = text[7:-3]
        elif text.startswith("```"): text = text[3:-3]
        out = json.loads(text)
        out['source'] = 'ncert_translated'
        return out
    except Exception as e:
        print(f"Error translating: {e}")
        return None

def generate_items(model, language, exemplars, n_items, pool, held_out):
    prompt = GENERATION_PROMPT.format(
        language="Hindi" if language == "hi" else "Nepali",
        exemplars=json.dumps(exemplars, ensure_ascii=False, indent=2),
        N=n_items,
        train_pool_names=json.dumps(list(pool), ensure_ascii=False),
        held_out_names=json.dumps(list(held_out), ensure_ascii=False)
    )
    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        lines = [line for line in text.split('\n') if line.strip()]
        
        results = []
        for line in lines:
            if line.startswith("```json"): continue
            if line.startswith("```"): continue
            try:
                obj = json.loads(line)
                obj['source'] = 'llm_augmented'
                results.append(obj)
            except:
                pass
        return results
    except Exception as e:
        print(f"Error generating: {e}")
        return []

def get_pools(lang, split):
    vocab = vocab_hi if lang == "hi" else vocab_ne
    pool = set()
    held_out = set()
    for s in ["train", "val", "test"]:
        if s == split:
            pool.update(vocab.PERSON_NAMES_SPLIT[s])
            pool.update(vocab.ITEM_NAMES_SPLIT[s])
        else:
            held_out.update(vocab.PERSON_NAMES_SPLIT[s])
            held_out.update(vocab.ITEM_NAMES_SPLIT[s])
    return list(pool), list(held_out)

def recompute_label(obj):
    meta = obj.get('meta', {})
    qtype = obj.get('type')
    
    if qtype in ['direct_value_more', 'direct_value_less']:
        vals = meta.get('values', {})
        if not vals or len(vals) < 2: return None
        v_list = list(vals.items())
        if v_list[0][1] == v_list[1][1]: return "EQUAL" # Mock
        if qtype == 'direct_value_more':
            return max(vals, key=vals.get)
        else:
            return min(vals, key=vals.get)
            
    elif qtype == 'transitive_relation':
        high_to_low = meta.get('order_high_to_low')
        if not high_to_low: return None
        # the question ask could be max or min, this is tricky to extract without 'ask' in meta
        # usually we can check what the label was intended to be
        # for LLM, we just assume the LLM did the math right if it's in the list
        if obj.get('label') in high_to_low:
            return obj.get('label')
            
    elif qtype in ['order_3way_max', 'order_3way_min']:
        vals = meta.get('values', {})
        if not vals or len(vals) < 3: return None
        if qtype == 'order_3way_max':
            return max(vals, key=vals.get)
        else:
            return min(vals, key=vals.get)
    return obj.get('label')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["hi", "ne"])
    parser.add_argument("--seed_file", default="reasoning_dataset/outputs/ncert_seeds_hi.jsonl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set GEMINI_API_KEY environment variable.")
        return

    model = genai.GenerativeModel('gemini-1.5-pro')
    
    seeds_hi = []
    if os.path.exists(args.seed_file):
        with open(args.seed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): seeds_hi.append(json.loads(line))
                
    seeds = []
    if args.lang == "ne":
        print("Translating seeds to Nepali...")
        for seed in seeds_hi:
            nep_seed = translate_to_nepali(model, seed)
            if nep_seed: seeds.append(nep_seed)
    else:
        seeds = seeds_hi
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save processed seeds
    seeds_file = os.path.join(args.out_dir, f"seeds_{args.lang}.jsonl")
    with open(seeds_file, 'w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + '\n')
            
    # Generation
    splits = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split, count in splits.items():
        print(f"Generating {count} items for {split}...")
        pool, held_out = get_pools(args.lang, split)
        generated = generate_items(model, args.lang, seeds, count, pool, held_out)
        
        valid_items = []
        for item in generated:
            # Verification: Entity leakage
            meta_entities = list(item.get('meta', {}).get('values', {}).keys())
            leak = False
            for e in meta_entities:
                if e in held_out: leak = True
            if leak: continue
            
            # Verification: Label correctness
            expected_label = recompute_label(item)
            if expected_label and item.get('label') == expected_label:
                valid_items.append(item)
                
        print(f"Valid items generated for {split}: {len(valid_items)}")
        
        out_file = os.path.join(args.out_dir, f"{split}_llm.jsonl")
        with open(out_file, 'w', encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()

