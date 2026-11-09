import json
import random
import os

names_hi = ["राम", "श्याम", "सीता", "गीता", "राहुल", "अमित", "सुनील", "पूजा", "नेहा", "विकास", "रोहन", "प्रिया"]
names_ne = ["राम", "सीता", "हरि", "गीता", "विकास", "अनिता", "श्याम", "कमला", "बिनोद", "सुनिता", "दिनेश", "संगीता"]

def generate_easy(lang, names):
    data = []
    for i in range(60):
        a, b = random.sample(names, 2)
        num_a, num_b = random.sample(range(10, 100), 2)
        
        if random.choice([True, False]):
            if lang == "hi":
                prompt = f"{a} के पास {num_a} रुपये हैं और {b} के पास {num_b} रुपये हैं। किसके पास ज़्यादा रुपये हैं?"
            else:
                prompt = f"{a} सँग {num_a} रुपैयाँ छ र {b} सँग {num_b} रुपैयाँ छ। को सँग धेरै रुपैयाँ छ?"
            ans = a if num_a > num_b else b
        else:
            if lang == "hi":
                prompt = f"{a} के पास {num_a} रुपये हैं और {b} के पास {num_b} रुपये हैं। किसके पास कम रुपये हैं?"
            else:
                prompt = f"{a} सँग {num_a} रुपैयाँ छ र {b} सँग {num_b} रुपैयाँ छ। को सँग थोरै रुपैयाँ छ?"
            ans = a if num_a < num_b else b
            
        data.append({"id": f"{lang}_easy_{i:03d}", "prompt": prompt, "answer": ans, "label": ans, "tier": 1})

    os.makedirs(f"phase3/data/{lang}/simple", exist_ok=True)
    with open(f"phase3/data/{lang}/simple/test_easy.jsonl", "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Created 60 ultra-simple prompts in phase3/data/{lang}/simple/test_easy.jsonl")

generate_easy("hi", names_hi)
generate_easy("ne", names_ne)
