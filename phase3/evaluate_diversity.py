import json
import argparse
from collections import Counter

def get_trigrams(text):
    words = text.split()
    return [" ".join(words[i:i+3]) for i in range(len(words)-2)]

def analyze_file(filepath):
    total_trigrams = 0
    unique_trigrams = set()
    prompts = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                prompt = data['prompt']
                prompts.append(prompt)
                
                trigrams = get_trigrams(prompt)
                total_trigrams += len(trigrams)
                unique_trigrams.update(trigrams)
            except:
                pass
                
    ratio = len(unique_trigrams) / total_trigrams if total_trigrams > 0 else 0
    return {
        "count": len(prompts),
        "total_trigrams": total_trigrams,
        "unique_trigrams": len(unique_trigrams),
        "distinct_trigram_ratio": ratio
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_file", required=True)
    parser.add_argument("--augmented_file", required=True)
    args = parser.parse_args()

    print("Analyzing template-only data...")
    t_stats = analyze_file(args.template_file)
    print(f"Total prompts: {t_stats['count']}")
    print(f"Distinct trigram ratio: {t_stats['distinct_trigram_ratio']:.4f}")

    print("\nAnalyzing LLM-augmented data...")
    a_stats = analyze_file(args.augmented_file)
    print(f"Total prompts: {a_stats['count']}")
    print(f"Distinct trigram ratio: {a_stats['distinct_trigram_ratio']:.4f}")
    
    improvement = a_stats['distinct_trigram_ratio'] - t_stats['distinct_trigram_ratio']
    print(f"\nDiversity Improvement: +{improvement:.4f} absolute")

if __name__ == "__main__":
    main()

