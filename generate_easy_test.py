import json
import random
import os

names = ["राम", "श्याम", "सीता", "गीता", "राहुल", "अमित", "सुनील", "पूजा", "नेहा", "विकास", "रोहन", "प्रिया"]

data = []
for _ in range(60):
    a, b = random.sample(names, 2)
    num_a, num_b = random.sample(range(10, 100), 2)
    
    if random.choice([True, False]):
        prompt = f"{a} के पास {num_a} रुपये हैं और {b} के पास {num_b} रुपये हैं। किसके पास ज़्यादा रुपये हैं?"
        ans = a if num_a > num_b else b
    else:
        prompt = f"{a} के पास {num_a} रुपये हैं और {b} के पास {num_b} रुपये हैं। किसके पास कम रुपये हैं?"
        ans = a if num_a < num_b else b
        
    data.append({"prompt": prompt, "answer": ans, "label": ans, "tier": 1})

os.makedirs("phase3/data/simple", exist_ok=True)
with open("phase3/data/simple/test_easy.jsonl", "w", encoding="utf-8") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("Created 60 ultra-simple prompts in phase3/data/simple/test_easy.jsonl")
