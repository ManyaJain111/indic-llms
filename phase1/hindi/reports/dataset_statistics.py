#!/usr/bin/env python3
"""Generate reproducible statistics for the Hindi train/validation/test splits."""

from pathlib import Path


def file_stats(path):
    lines = words = characters = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            lines += 1
            words += len(line.split())
            characters += len(line)
    return lines, words, characters


def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "datasets"
    rows = []
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append((split, *file_stats(path)))

    total = tuple(sum(row[index] for row in rows) for index in range(1, 4))
    report = [
        "# Hindi Dataset Statistics Report",
        "",
        "Generated from the current split files. Word counts are whitespace-delimited and are not tokenizer-token counts.",
        "",
        "| Split | Lines | Words | Characters |",
        "|---|---:|---:|---:|",
    ]
    report.extend(f"| {name}.txt | {lines:,} | {words:,} | {chars:,} |" for name, lines, words, chars in rows)
    report.append(f"| **Total** | **{total[0]:,}** | **{total[1]:,}** | **{total[2]:,}** |")
    output = Path(__file__).resolve().parent / "hindi_dataset_statistics.md"
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
