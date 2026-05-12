"""
Build a JSONL of Wikipedia articles created strictly after CUTOFF_DATE by
walking MediaWiki's `logevents` API (letype=create) in chronological order
and fetching plain-text extracts for each created page.

Why this approach (vs streaming an HF dump):
- The HF wikimedia/wikipedia dump streams in page-id order, so the first
  millions of rows are 2001-2010 articles. Scanning enough rows to reach
  post-2020 content takes hours.
- The MediaWiki logevents endpoint gives us page creations directly, sorted
  by time. Every result is guaranteed post-cutoff — no filtering wasted on
  ancient pages.

Resumable: writes one JSONL line per kept article, plus a checkpoint file
recording the API continuation token. Re-running picks up where the previous
run stopped.

For GPT-2 (released Feb 2019, WebText scraped through Dec 2017), cutoff
2020-01-01 leaves a safe margin past any plausible retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "spv-mia-post-cutoff-collector/1.0 (issanodeh@gmail.com)",
}

DEFAULT_CUTOFF = "2020-01-01T00:00:00Z"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/post_cutoff.jsonl")
    p.add_argument("--checkpoint", default="data/raw/checkpoint.json")
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                   help="ISO-8601 cutoff. Articles created strictly after this are kept.")
    p.add_argument("--target", type=int, default=5000,
                   help="Stop after this many kept articles (0 = unlimited).")
    p.add_argument("--list-limit", type=int, default=500,
                   help="logevents page size (max 500 anon).")
    p.add_argument("--extract-batch", type=int, default=20,
                   help="extracts query page size (max 20 with explaintext).")
    p.add_argument("--min-chars", type=int, default=1500,
                   help="Skip articles whose extract is shorter than this.")
    p.add_argument("--sleep", type=float, default=0.2,
                   help="Seconds to sleep between API calls.")
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"kept": 0, "lecontinue": None, "last_timestamp": None}


def save_checkpoint(path: Path, ckpt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt))
    tmp.replace(path)


def api_get(params: dict) -> dict:
    """GET with retry/backoff. Raises on persistent failure."""
    for attempt in range(5):
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {e}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"API failed after retries: {params}")


def fetch_create_events(cutoff: str, lecontinue: str | None, limit: int) -> dict:
    """List page-creation events in main namespace, oldest-first, from cutoff."""
    params = {
        "action": "query",
        "list": "logevents",
        "letype": "create",
        "lestart": cutoff,
        "ledir": "newer",
        "lenamespace": 0,           # main namespace only — no Talk:, User:, etc.
        "lelimit": limit,
        "format": "json",
        "formatversion": 2,
    }
    if lecontinue:
        params["lecontinue"] = lecontinue
    return api_get(params)


def fetch_extracts(pageids: list[int]) -> dict[int, dict]:
    """Plain-text article extracts + canonical URLs for given pageids."""
    if not pageids:
        return {}
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "pageids": "|".join(str(p) for p in pageids),
        "format": "json",
        "formatversion": 2,
    }
    data = api_get(params)
    out: dict[int, dict] = {}
    for page in data.get("query", {}).get("pages", []):
        if page.get("missing") or page.get("invalid"):
            continue
        out[int(page["pageid"])] = {
            "title": page.get("title", ""),
            "url": page.get("fullurl") or page.get("canonicalurl") or "",
            "extract": page.get("extract", ""),
        }
    return out


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)

    ckpt = load_checkpoint(ckpt_path)
    kept = ckpt["kept"]
    lecontinue = ckpt["lecontinue"]
    print(f"Resuming: kept={kept}, lecontinue={lecontinue}")

    out_f = out_path.open("a", encoding="utf-8")
    target = args.target or float("inf")

    started = time.time()
    scanned = 0
    try:
        while kept < target:
            data = fetch_create_events(args.cutoff, lecontinue, args.list_limit)
            events = data.get("query", {}).get("logevents", [])
            if not events:
                print("no more create events from Wikipedia API")
                break

            # Map pageid -> create timestamp (events are already post-cutoff,
            # sorted oldest-first because ledir=newer).
            event_ts: dict[int, str] = {}
            for ev in events:
                pid = ev.get("pageid")
                if pid:
                    event_ts[int(pid)] = ev["timestamp"]
            scanned += len(events)

            # Fetch extracts in batches of `extract_batch`.
            pids = list(event_ts.keys())
            for i in range(0, len(pids), args.extract_batch):
                batch = pids[i:i + args.extract_batch]
                infos = fetch_extracts(batch)
                for pid, info in infos.items():
                    extract = info["extract"]
                    if len(extract) < args.min_chars:
                        continue
                    record = {
                        "id": str(pid),
                        "title": info["title"],
                        "url": info["url"],
                        "created": event_ts.get(pid),
                        "text": extract,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept >= target:
                        break
                time.sleep(args.sleep)
                if kept >= target:
                    break

            out_f.flush()

            cont = data.get("continue", {})
            lecontinue = cont.get("lecontinue")
            last_ts = events[-1]["timestamp"] if events else None
            save_checkpoint(ckpt_path, {
                "kept": kept,
                "lecontinue": lecontinue,
                "last_timestamp": last_ts,
            })

            rate = kept / max(1.0, time.time() - started + 0.001)
            print(f"  scanned={scanned} kept={kept} last={last_ts} ({rate:.1f} kept/s)")

            if lecontinue is None:
                print("API returned no continuation — exhausted log")
                break
    finally:
        out_f.close()
        save_checkpoint(ckpt_path, {
            "kept": kept,
            "lecontinue": lecontinue,
            "last_timestamp": ckpt.get("last_timestamp"),
        })

    print(f"done. scanned={scanned}, kept={kept}, output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
