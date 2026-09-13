import os
import json
import argparse
from datasets import load_dataset
import pandas as pd
import google.generativeai as genai

# Setup Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

EXTRACTION_PROMPT = """
You are given a passage from a math, science, or economics textbook.
Extract any comparative quantities found in the text and frame them into a comparative-reasoning question.
Requirements:
1. Identify 2 or 3 named entities with numeric values and units (e.g. two vehicles' speeds, two solutions' concentrations, populations of two countries).
2. Or identify a relational chain between three quantities (A > B, B > C).
3. If no such quantities exist, return an empty JSON list [].
4. Otherwise, return a JSON list of objects matching this schema:
{
  "type": "direct_value_more" | "direct_value_less" | "transitive_relation" | "order_3way_max" | "order_3way_min",
  "prompt": "<The question in natural Hindi>",
  "answer": "<The full sentence answer in Hindi>",
  "label": "<The exact entity name that answers the question>",
  "meta": {"attribute": "<quantity type>", "values": {"Entity1": <num>, "Entity2": <num>}}
}
Only extract if you find clear comparable quantities. Do not invent numbers. Output ONLY valid JSON list.

Passage:
{passage}
"""

def extract_from_passage(passage, model):
    try:
        response = model.generate_content(EXTRACTION_PROMPT.format(passage=passage))
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        if text.startswith("```"):
            text = text[3:-3]
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"Error extracting: {e}")
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reasoning_dataset/outputs/ncert_seeds_hi.jsonl")
    parser.add_argument("--max_seeds", type=int, default=15)
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set GEMINI_API_KEY environment variable.")
        return

    model = genai.GenerativeModel('gemini-1.5-pro')
    seeds = []
    
    print("Loading KadamParth/Ncert_dataset...")
    ds_kadam = load_dataset('KadamParth/Ncert_dataset', split='train', streaming=True)
    
    for i, row in enumerate(ds_kadam):
        if len(seeds) >= args.max_seeds // 2:
            break
        text = f"Topic: {row.get('Topic')}\nQuestion: {row.get('Question')}\nExplanation: {row.get('Explanation')}"
        extracted = extract_from_passage(text, model)
        if extracted:
            print(f"Extracted {len(extracted)} items from KadamParth")
            for item in extracted:
                item['source'] = 'ncert_derived'
                seeds.append(item)

    print("Loading goenkalokesh/NCERT_MCQs...")
    try:
        df_goenka = pd.read_csv('NCERT_MCQs.csv', encoding='latin1')
        for i, row in df_goenka.iterrows():
            if len(seeds) >= args.max_seeds:
                break
            passage = str(row['passage'])
            extracted = extract_from_passage(passage, model)
            if extracted:
                print(f"Extracted {len(extracted)} items from goenkalokesh")
                for item in extracted:
                    item['source'] = 'ncert_derived'
                    seeds.append(item)
    except Exception as e:
        print(f"Could not load NCERT_MCQs.csv: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(seeds)} seeds to {args.out}")

if __name__ == "__main__":
    main()

