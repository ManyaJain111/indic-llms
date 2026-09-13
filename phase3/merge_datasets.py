import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", required=True)
    parser.add_argument("--llm_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {}

    for split in ["train", "val", "test"]:
        records = []
        
        t_file = os.path.join(args.template_dir, f"{split}.jsonl")
        if os.path.exists(t_file):
            with open(t_file, 'r', encoding='utf-8') as f:
                records.extend([json.loads(line) for line in f if line.strip()])
                
        l_file = os.path.join(args.llm_dir, f"{split}_llm.jsonl")
        if os.path.exists(l_file):
            with open(l_file, 'r', encoding='utf-8') as f:
                records.extend([json.loads(line) for line in f if line.strip()])

        out_file = os.path.join(args.out_dir, f"{split}.jsonl")
        type_counts = {}
        source_counts = {}
        
        with open(out_file, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                type_counts[r['type']] = type_counts.get(r['type'], 0) + 1
                source = r.get('source', 'template')
                source_counts[source] = source_counts.get(source, 0) + 1
                
        stats[split] = {
            "n": len(records),
            "type_counts": type_counts,
            "source_counts": source_counts
        }
        print(f"Merged {split}: {len(records)} records")

    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats written to {stats_path}")

if __name__ == "__main__":
    main()

