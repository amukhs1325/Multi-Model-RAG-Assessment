#!/usr/bin/env python3
"""Download SQuAD v1.1 and write train.csv / validation.csv in the format the pipeline expects.

processing.py indexes the `context` column of both files, and the two RAG
scripts score answers against the `answers` column of validation.csv. Answers
are stored as JSON ({"text": [...], "answer_start": [...]}), which the scripts'
answer parsers read directly.

Usage:
    python download_squad.py                       # full dataset (~88k train / ~10.5k validation rows)
    python download_squad.py --train-limit 5000    # smaller index for a quick local run
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

COLUMNS = ["id", "title", "context", "question", "answers"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(Path(__file__).parent), help="Where to write the CSVs (default: this folder)")
    p.add_argument("--train-limit", type=int, default=0, help="Keep only the first N training rows (0 = all)")
    p.add_argument("--validation-limit", type=int, default=0, help="Keep only the first N validation rows (0 = all)")
    return p.parse_args()


def answers_to_json(ans):
    return json.dumps({
        "text": [str(t) for t in ans["text"]],
        "answer_start": [int(s) for s in ans["answer_start"]],
    }, ensure_ascii=False)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    squad = load_dataset("rajpurkar/squad")
    for split, limit in [("train", args.train_limit), ("validation", args.validation_limit)]:
        df = squad[split].to_pandas()
        if limit:
            df = df.head(limit)
        df["answers"] = df["answers"].apply(answers_to_json)
        path = out_dir / f"{split}.csv"
        df[COLUMNS].to_csv(path, index=False)
        print(f"Wrote {len(df):,} rows ({df['context'].nunique():,} unique contexts) to {path}")


if __name__ == "__main__":
    main()
