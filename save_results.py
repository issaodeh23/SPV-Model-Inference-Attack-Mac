"""
Run after attack.py to save a timestamped results summary to results/.
Usage: python3 save_results.py
"""
import json
import os
import pathlib
import datetime
import numpy as np
import yaml
from sklearn.metrics import auc, roc_auc_score

REPO = pathlib.Path(__file__).parent

with open(REPO / "configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

model   = cfg["model_name"]
dataset = cfg["dataset_name"]
samples = cfg["maximum_samples"]

data_dir = REPO / cfg["attack_data_path"] / f"attack_data_{model}@{dataset}" / "target"
roc_path = REPO / cfg["attack_data_path"] / f"attack_data_{model}@{dataset}" / "roc_stat.npz"

if not roc_path.exists():
    print(f"No results found at {roc_path}. Run attack.py first.")
    exit(1)

roc = np.load(roc_path)
fpr, tpr = roc["fpr"], roc["tpr"]

auc_score   = auc(fpr, tpr)
asr         = float(tpr[np.argmin(np.abs(tpr - (1 - fpr)))])
tpr_at_1fpr = float(tpr[np.argmin(np.abs(fpr - 0.01))])
tpr_at_5fpr = float(tpr[np.argmin(np.abs(fpr - 0.05))])

results = {
    "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
    "model":            model,
    "dataset":          dataset,
    "maximum_samples":  samples,
    "perturbation_number": cfg["perturbation_number"],
    "sample_number":    cfg["sample_number"],
    "calibration":      cfg["calibration"],
    "target_model":     cfg["target_model"],
    "reference_model":  cfg["reference_model"],
    "metrics": {
        "AUC":          round(auc_score, 4),
        "ASR":          round(asr, 4),
        "TPR@1%FPR":    round(tpr_at_1fpr, 4),
        "TPR@5%FPR":    round(tpr_at_5fpr, 4),
    },
    "paper_reference": {
        "model":        "gpt2 (124M)",
        "dataset":      "wikitext-103-raw-v1 → AUC=0.975 / TPR@1%FPR=0.673",
        "AUC_wikitext": 0.975,
        "TPR1_wikitext": 0.673,
        "AUC_agnews":   0.949,
        "TPR1_agnews":  0.429,
    }
}

out_dir = REPO / "results"
out_dir.mkdir(exist_ok=True)
tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"results_{model}_{dataset}_{tag}.json"
out_path.write_text(json.dumps(results, indent=2))

print(f"\n{'='*50}")
print(f"  Model:        {model}")
print(f"  Dataset:      {dataset}")
print(f"  Samples:      {samples} members + {samples} non-members")
print(f"{'='*50}")
paper_auc = 0.975 if dataset in ("wikitext", "wikitext-103-raw-v1") else 0.949
paper_tpr = 0.673 if dataset in ("wikitext", "wikitext-103-raw-v1") else 0.429
print(f"  AUC:          {auc_score:.4f}  (paper: {paper_auc})")
print(f"  ASR:          {asr:.4f}")
print(f"  TPR@1%FPR:    {tpr_at_1fpr:.4f}  (paper: {paper_tpr})")
print(f"  TPR@5%FPR:    {tpr_at_5fpr:.4f}")
print(f"{'='*50}")
print(f"  Saved to: {out_path}")
