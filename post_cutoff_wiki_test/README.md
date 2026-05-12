# Post-Cutoff Wikipedia Test for SPV-MIA

A self-contained validation harness for the SPV-MIA repo. It builds a corpus
of Wikipedia articles created **strictly after GPT-2's training window**,
runs the full SPV-MIA pipeline on it, and packages the results into a report
suitable for sharing with the research advisor.

## Why this experiment

SPV-MIA claims to detect whether a sample was in a model's **fine-tuning**
data. A skeptical advisor could reasonably ask: *how do we know the AUC on
wikitext-103 isn't inflated by GPT-2's pretraining having already seen that
text?*

This test answers that. Every article in our corpus was created **after
2020-01-01** — well past GPT-2's release (Feb 2019) and its WebText scrape
(Dec 2017). The base model has zero prior on this content. Any membership
signal therefore has to come from the fine-tuning we induce here, which is
exactly what an MIA should be measuring.

## Files

| File | Role |
|---|---|
| `01_collect.py` | Streams `wikimedia/wikipedia` (config `20231101.en`, public ungated dump), queries Wikipedia for each article's creation date, keeps those after 2020-01-01. Resumable. |
| `02_format.py` | Converts the kept JSONL into a HuggingFace Dataset on disk that SPV-MIA can load. |
| `03_run.sh` | Runs the SPV-MIA pipeline (target FT → self-prompt refs → reference FT → attack) on our dataset. |
| `04_report.py` | Reads `roc_stat.npz` from the attack output, computes AUC + TPR@1%FPR, writes Markdown + HTML report + ROC plot. |
| `data/raw/` | Filtered JSONL + checkpoint. |
| `data/hf_dataset/` | HF Dataset directory consumed by SPV-MIA. |
| `results/` | Final REPORT.md / REPORT.html / roc.png. |

## How it integrates with SPV-MIA

A single small patch in `../ANeurIPS2024_SPV-MIA/data/prepare.py`: when the
env var `SPV_MIA_LOCAL_DATASET_PATH` is set, SPV-MIA loads the dataset from
that directory via `datasets.load_from_disk` instead of from the HF Hub.
Everything else — model code, attack code, hyperparameters — is unchanged.

## Running on ALFA-Mac

```bash
ssh alfa-mac
cd "$HOME/Desktop/UROP_spring 2026/post_cutoff_wiki_test"

# First time: install extras into the SPV-MIA venv.
source ../ANeurIPS2024_SPV-MIA/.venv/bin/activate
pip install -r requirements.txt

# Step 1 — build the corpus (~10-30 min depending on network).
#   Streams the HF dataset, queries Wikipedia for creation dates,
#   writes data/raw/post_cutoff.jsonl. Resumable: re-run if interrupted.
python3 01_collect.py --target 5000

# Step 2 — format as HF Dataset.
python3 02_format.py

# Step 3 — run the SPV-MIA pipeline (use tmux; this takes hours on MPS).
tmux new -s spvmia-postcut
bash 03_run.sh
# (Detach with Ctrl-b d; reattach with `tmux a -t spvmia-postcut`.)

# 04_report.py runs automatically at the end of 03_run.sh.
# Read results/REPORT.md or open results/REPORT.html in a browser.
```

## Cutoff choice (and why GPT-2)

- **Target model:** `gpt2` (124M base). Matches the paper's headline-result
  setup and our existing wikitext-103 baseline.
- **Cutoff:** **2020-01-01** UTC. GPT-2 release: Feb 14 2019; training data
  (WebText) scraped through Dec 2017. Anything later than 2020 is provably
  unseen by GPT-2. To swap models (e.g. test a 2023-cutoff model), pass
  `--cutoff 2024-01-01T00:00:00Z` to `01_collect.py`.

## Output for the advisor

`results/REPORT.md` (and `REPORT.html`) contains:
- Headline metrics (AUC, TPR@1%FPR, TPR@0.1%FPR)
- Side-by-side comparison vs. paper's Wikitext-103 numbers
- Dataset stats (article count, date range, avg length)
- A 25-article sample (titles + URLs) so the advisor can spot-check that the
  filtering really did pick post-cutoff content
- Methodology summary, plus reproduction commands

## Known caveats

- Wikipedia's "creation date" comes from the first non-deleted revision. A
  small number of pages have had their early history suppressed; those rows
  silently drop out of our corpus (no entry returned by the API).
- We strip Markdown to plain text, but article structure (sections, lists)
  is preserved as text. This is closer to Wikitext-103's distribution than
  the raw HF dataset's Markdown, but not identical.
- The HF dataset filters articles to ≥2 sections / ≥50 tokens. Very stubby
  articles (the most common type of brand-new page) are already excluded
  upstream, which is actually what we want for fine-tuning.
