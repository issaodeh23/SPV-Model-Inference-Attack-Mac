"""
Convert the JSONL produced by 01_collect.py into a HuggingFace Dataset on disk
that SPV-MIA's data/prepare.py can load.

SPV-MIA calls `datasets.load_dataset(name, config)` and expects a "train" split
with a "text" column. We save_to_disk() in the format `load_from_disk()`
consumes, then plug in via a patch in 03_run.sh.

Output layout:
    data/hf_dataset/
        dataset_info.json
        train/
            ...
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw/post_cutoff.jsonl")
    p.add_argument("--out", default="data/hf_dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-articles", type=int, default=0,
                   help="Truncate to N articles after shuffling (0 = keep all).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)

    rows: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"loaded {len(rows)} articles from {in_path}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_articles and len(rows) > args.max_articles:
        rows = rows[: args.max_articles]
        print(f"truncated to {len(rows)} articles")

    # SPV-MIA's data/prepare.py auto-detects 'text' / 'document' / 'content' as
    # the text column. We name ours 'text' to match wikitext-103's schema.
    payload = {
        "text": [r["text"] for r in rows],
        "title": [r["title"] for r in rows],
        "url": [r["url"] for r in rows],
        "created": [r["created"] for r in rows],
        "id": [r["id"] for r in rows],
    }

    ds = Dataset.from_dict(payload)
    dd = DatasetDict({"train": ds})

    out_path.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(out_path))

    info = {
        "n_articles": len(rows),
        "earliest_created": min(r["created"] for r in rows) if rows else None,
        "latest_created": max(r["created"] for r in rows) if rows else None,
        "avg_chars": sum(len(r["text"]) for r in rows) / max(1, len(rows)),
    }
    (out_path / "POST_CUTOFF_STATS.json").write_text(json.dumps(info, indent=2))

    print(f"saved HF dataset to {out_path}")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
