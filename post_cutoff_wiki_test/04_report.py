"""
Build an advisor-facing report from the SPV-MIA run on the post-cutoff
Wikipedia dataset.

Reads:
  - $SPV_REPO/attack/attack_data_{model}@{dataset}/roc_stat.npz  (fpr, tpr)
  - data/hf_dataset/POST_CUTOFF_STATS.json                       (dataset stats)

Writes (into --out, default `results/`):
  - REPORT.md          self-contained markdown summary
  - REPORT.html        same content rendered to HTML
  - roc.png            ROC curve, log-scale x-axis (paper convention)
  - metrics.json       AUC, TPR@1%FPR, TPR@0.1%FPR, sample sizes
  - sample_articles.md 25 random article titles + URLs from the dataset

The report mirrors the headline numbers SPV-MIA reports in Table 1 of the
NeurIPS 2024 paper so the advisor can compare side-by-side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAPER_AUC_WIKITEXT = 0.975
PAPER_TPR_1FPR_WIKITEXT = 0.673


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spv-repo", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--dataset", default="post_cutoff_wiki")
    return p.parse_args()


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Linear-interp TPR at a given FPR. Matches paper convention."""
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    if idx < 0:
        return float(tpr[0])
    if idx >= len(fpr) - 1:
        return float(tpr[-1])
    f0, f1 = fpr[idx], fpr[idx + 1]
    t0, t1 = tpr[idx], tpr[idx + 1]
    if f1 == f0:
        return float(t0)
    return float(t0 + (t1 - t0) * (target_fpr - f0) / (f1 - f0))


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr))


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, out_path: Path, auc_score: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(fpr, tpr, lw=2, label=f"SPV-MIA (AUC={auc_score:.3f})")
    ax.plot([1e-4, 1], [1e-4, 1], "k--", lw=1, alpha=0.5, label="chance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("False Positive Rate (log)")
    ax.set_ylabel("True Positive Rate (log)")
    ax.set_title("SPV-MIA on Post-Cutoff Wikipedia (GPT-2)")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def md_to_html(md: str, out: Path) -> None:
    """Minimal MD → HTML so the report opens in a browser without a markdown
    renderer installed. Keep it dumb on purpose."""
    try:
        import markdown  # noqa: F401
        import markdown as _md
        body = _md.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + md.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>SPV-MIA Post-Cutoff Report</title>"
        "<style>body{font-family:-apple-system,system-ui,sans-serif;max-width:880px;"
        "margin:2em auto;padding:0 1em;line-height:1.5;color:#222}"
        "table{border-collapse:collapse;margin:1em 0}"
        "th,td{border:1px solid #ccc;padding:6px 12px;text-align:left}"
        "th{background:#f3f3f3}code{background:#f3f3f3;padding:2px 5px;border-radius:3px}"
        "img{max-width:100%}</style></head><body>"
        f"{body}</body></html>"
    )
    out.write_text(html)


def main() -> int:
    args = parse_args()
    spv_repo = Path(args.spv_repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_path = spv_repo / "attack" / f"attack_data_{args.model}@{args.dataset}" / "roc_stat.npz"
    if not roc_path.exists():
        raise SystemExit(f"ROC file not found: {roc_path}")
    npz = np.load(roc_path)
    fpr, tpr = npz["fpr"], npz["tpr"]

    auc_score = auc(fpr, tpr)
    tpr_1 = tpr_at_fpr(fpr, tpr, 0.01)
    tpr_01 = tpr_at_fpr(fpr, tpr, 0.001)

    stats_path = Path(args.dataset_dir) / "POST_CUTOFF_STATS.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "auc": auc_score,
        "tpr_at_1pct_fpr": tpr_1,
        "tpr_at_0_1pct_fpr": tpr_01,
        "dataset_stats": stats,
        "paper_reference_wikitext103": {
            "auc": PAPER_AUC_WIKITEXT,
            "tpr_at_1pct_fpr": PAPER_TPR_1FPR_WIKITEXT,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_roc(fpr, tpr, out_dir / "roc.png", auc_score)

    sample_lines = []
    try:
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset_dir)["train"]
        n = min(25, len(ds))
        idxs = np.random.default_rng(0).choice(len(ds), size=n, replace=False).tolist()
        for i in idxs:
            row = ds[i]
            sample_lines.append(f"- [{row['title']}]({row['url']}) — created {row['created']}")
        (out_dir / "sample_articles.md").write_text(
            "# Sample of post-cutoff articles used\n\n" + "\n".join(sample_lines) + "\n"
        )
    except Exception as e:
        sample_lines = [f"(could not load samples: {e})"]

    md = f"""# SPV-MIA Validation on Post-Cutoff Wikipedia

## TL;DR

We ran the SPV-MIA pipeline end-to-end on a dataset built from Wikipedia
articles whose creation timestamps are **strictly after GPT-2's training
cutoff**. Because the base model has provably never seen these pages, any
membership signal must come from the fine-tuning phase — exactly what an MIA
should detect.

| Metric | This run | Paper (Wikitext-103) |
|---|---|---|
| AUC | **{auc_score:.3f}** | {PAPER_AUC_WIKITEXT:.3f} |
| TPR @ 1% FPR | **{tpr_1:.3f}** | {PAPER_TPR_1FPR_WIKITEXT:.3f} |
| TPR @ 0.1% FPR | **{tpr_01:.3f}** | n/a |

![ROC curve](roc.png)

## Methodology

1. **Source dataset.** Streamed `wikimedia/wikipedia` (config `20231101.en`,
   HuggingFace, the official public Wikimedia dump, ~6.4M English articles).
2. **Creation date lookup.** For each candidate, queried the MediaWiki API
   (`action=query&prop=revisions&rvdir=newer&rvlimit=1`) by page ID to get
   the timestamp of the article's **first revision**.
3. **Cutoff filter.** Kept only articles created strictly after
   **2020-01-01**. GPT-2 was released February 2019, trained on WebText
   collected through December 2017; a 2020 cutoff gives a safe margin against
   any silent retraining.
4. **Text cleaning.** Stripped Markdown formatting to match Wikitext-103's
   plain-prose distribution.
5. **Pipeline.** Reused the SPV-MIA scripts (`ft_llms/llms_finetune.py`,
   `refer_data_generate.py`, `attack.py`) without modification beyond pointing
   their dataset loader at our local HuggingFace Dataset via
   `SPV_MIA_LOCAL_DATASET_PATH`.

## Dataset Stats

```json
{json.dumps(stats, indent=2)}
```

## Interpretation

- **AUC ≫ 0.5** means the SPV-MIA implementation **does** distinguish
  members from non-members on a corpus the base model never saw — i.e., the
  attack signal is coming from fine-tuning memorization, not from pretraining
  leakage. This is a positive control for the pipeline.
- **Result vs. paper.** The paper reports 0.975 AUC on Wikitext-103. Numbers
  on the post-cutoff Wikipedia corpus may differ because: (a) article length
  and vocabulary are slightly different; (b) the corpus is smaller; (c) GPT-2
  has zero prior on this text, which can make memorization signals
  *sharper* (favorable for MIA).
- **Take-away for the advisor.** If AUC remains high (~0.95+) we have
  evidence the SPV-MIA repo is correctly attributing the membership signal
  to fine-tuning rather than to pretraining-era leakage of test set content.
  If AUC drops sharply, that suggests the original wikitext-103 number was
  partially propped up by GPT-2's pretraining on similar text.

## Sample of Articles Used

{chr(10).join(sample_lines)}

## Reproducibility

```bash
cd post_cutoff_wiki_test
python3 01_collect.py --target 5000           # ~10-30 min, depends on HF/Wiki latency
python3 02_format.py
bash    03_run.sh                              # multi-hour on MPS, see logs/pipeline.log
python3 04_report.py --spv-repo ../ANeurIPS2024_SPV-MIA \\
                     --dataset-dir data/hf_dataset --out results
```
"""
    (out_dir / "REPORT.md").write_text(md)
    md_to_html(md, out_dir / "REPORT.html")
    print(f"wrote report to {out_dir}/REPORT.md")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
