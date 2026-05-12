#!/usr/bin/env bash
# 03_run.sh — run SPV-MIA end-to-end on the post-cutoff Wikipedia dataset.
#
# Mirrors ANeurIPS2024_SPV-MIA/_run_pipeline.sh but points the dataset env-var
# at our locally-built HF dataset (data/hf_dataset/). The target model is GPT-2
# base (124M); see ../REPRODUCTION_NOTES for hyperparameter rationale.
#
# Run on ALFA-Mac under tmux:
#     cd "$HOME/Desktop/UROP_spring 2026/post_cutoff_wiki_test"
#     tmux new -s spvmia-postcut
#     bash 03_run.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SPV-MIA repo is named differently on local vs ALFA-Mac. Honor an explicit
# $SPV_REPO override first; otherwise probe the known names.
if [ -n "${SPV_REPO:-}" ] && [ -d "$SPV_REPO" ]; then
    SPV_REPO="$(cd "$SPV_REPO" && pwd)"
elif [ -d "$HERE/../SPV-MIA" ]; then
    SPV_REPO="$(cd "$HERE/../SPV-MIA" && pwd)"
elif [ -d "$HERE/../ANeurIPS2024_SPV-MIA" ]; then
    SPV_REPO="$(cd "$HERE/../ANeurIPS2024_SPV-MIA" && pwd)"
else
    echo "ERROR: cannot find SPV-MIA repo. Set \$SPV_REPO or place it at ../SPV-MIA"
    exit 1
fi
echo "  using SPV-MIA repo at: $SPV_REPO"

# ── Sanity checks ─────────────────────────────────────────
if [ ! -d "$HERE/data/hf_dataset" ]; then
    echo "ERROR: $HERE/data/hf_dataset is missing."
    echo "       Run 01_collect.py + 02_format.py first."
    exit 1
fi
if [ ! -f "$SPV_REPO/_run_pipeline.sh" ]; then
    echo "ERROR: SPV-MIA repo not found at $SPV_REPO"
    exit 1
fi

# Exported so SPV-MIA's patched data/prepare.py loads our dataset.
export SPV_MIA_LOCAL_DATASET_PATH="$HERE/data/hf_dataset"

# A clean label that becomes the cache directory path.
MODEL="gpt2"
DATASET="post_cutoff_wiki"
DATASET_CONFIG="v1"

VENV="$SPV_REPO/.venv"
LOG="$HERE/logs/pipeline.log"
TARGET_OUT="$SPV_REPO/ft_llms/$MODEL/$DATASET_CONFIG/target"
REFER_OUT="$SPV_REPO/ft_llms/$MODEL/$DATASET_CONFIG/refer"
CACHE="$SPV_REPO/cache"
ACCEL_CFG="$SPV_REPO/accelerate_config_mac.yaml"

mkdir -p "$HERE/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo ""; echo "$(date '+%H:%M:%S') ══ $1 ══"; echo ""; }

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: SPV-MIA venv not found at $VENV — run _run_pipeline.sh once to set it up."
    exit 1
fi
source "$VENV/bin/activate"

DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available(): print('cuda')
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): print('mps')
else: print('cpu')
" 2>/dev/null)

if [ "$DEVICE" = "cuda" ]; then
    BATCH=16
    GRAD_ACCUM=1
    ACCEL_LAUNCH="python3 -m accelerate.commands.launch"
else
    BATCH=16
    GRAD_ACCUM=1
    ACCEL_LAUNCH="python3 -m accelerate.commands.launch --config_file $ACCEL_CFG"
    mkdir -p ~/.cache/huggingface/accelerate
    cp "$ACCEL_CFG" ~/.cache/huggingface/accelerate/default_config.yaml
fi

TARGET_LR="1e-4"
REFER_LR="5e-5"
TARGET_EPOCHS=10
REFER_EPOCHS=4

cd "$SPV_REPO"

log "STEP 1 · Fine-tuning TARGET ($MODEL on post-cutoff Wikipedia, $TARGET_EPOCHS epochs)"
mkdir -p "$TARGET_OUT"
$ACCEL_LAUNCH ft_llms/llms_finetune.py \
    --disable_peft \
    -m "$MODEL" \
    -d "$DATASET" \
    -dc "$DATASET_CONFIG" \
    --output_dir "$TARGET_OUT" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --eval_steps 200 \
    --save_epochs 1000 \
    --log_steps 50 \
    -e "$TARGET_EPOCHS" \
    -b "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    -lr "$TARGET_LR" \
    --disable_flash_attention \
    --use_dataset_cache \
    --packing \
    --train_sta_idx 0 --train_end_idx 50000 \
    --eval_sta_idx 0  --eval_end_idx 5000 \
    -s 2

TARGET_CKPT=$(ls -td "$TARGET_OUT"/checkpoint-* 2>/dev/null | head -1)
[ -z "$TARGET_CKPT" ] && { echo "ERROR: no target checkpoint"; exit 1; }
echo "  target ckpt: $TARGET_CKPT"

log "STEP 2 · Self-prompt reference data generation"
$ACCEL_LAUNCH ft_llms/refer_data_generate.py \
    -tm "$TARGET_CKPT" \
    -m "$MODEL" \
    -d "$DATASET" \
    -dc "$DATASET_CONFIG" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --use_dataset_cache \
    --packing

log "STEP 3 · Fine-tuning REFERENCE ($REFER_EPOCHS epochs)"
mkdir -p "$REFER_OUT"
$ACCEL_LAUNCH ft_llms/llms_finetune.py \
    --refer \
    -m "$MODEL" \
    -d "$DATASET" \
    -dc "$DATASET_CONFIG" \
    --output_dir "$REFER_OUT" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --eval_steps 100 \
    --save_epochs 500 \
    --log_steps 50 \
    -e "$REFER_EPOCHS" \
    -b "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    -lr "$REFER_LR" \
    --disable_flash_attention \
    --use_dataset_cache \
    --packing \
    --train_sta_idx 0 --train_end_idx 50000 \
    --eval_sta_idx 0  --eval_end_idx 5000 \
    -s 2

REFER_CKPT=$(ls -td "$REFER_OUT"/checkpoint-* 2>/dev/null | head -1)
[ -z "$REFER_CKPT" ] && { echo "ERROR: no reference checkpoint"; exit 1; }
echo "  refer ckpt: $REFER_CKPT"

log "STEP 4 · Updating configs/config.yaml"
python3 - "$TARGET_CKPT" "$REFER_CKPT" "$MODEL" "$DATASET" "$DATASET_CONFIG" << 'PYEOF'
import sys, yaml, pathlib
target, refer = sys.argv[1], sys.argv[2]
p = pathlib.Path("configs/config.yaml")
cfg = yaml.safe_load(p.read_text())
cfg["model_name"]            = sys.argv[3]
cfg["target_model"]          = target
cfg["reference_model"]       = refer
cfg["dataset_name"]          = sys.argv[4]
cfg["dataset_config_name"]   = sys.argv[5]
cfg["load_attack_data"]      = False
p.write_text(yaml.dump(cfg, default_flow_style=False))
print(f"  target_model:    {target}")
print(f"  reference_model: {refer}")
PYEOF

log "STEP 5 · Running SPV-MIA attack"
python3 attack.py

log "STEP 6 · Building report"
python3 "$HERE/04_report.py" \
    --spv-repo "$SPV_REPO" \
    --dataset-dir "$HERE/data/hf_dataset" \
    --out "$HERE/results"

log "DONE — see $HERE/results/REPORT.md"
