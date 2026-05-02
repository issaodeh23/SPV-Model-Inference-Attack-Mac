#!/usr/bin/env bash
# ============================================================
#  run_poc.sh — SPV-MIA paper replication run (Mac/MPS)
#  Model: GPT-2-XL (1.5B)  |  Dataset: AG News
#  Reproduces Table 1 cell — paper expected:
#    AUC = 0.949   |   TPR@1%FPR = 42.9%   |   TPR@0.1%FPR = 25.3%
#
#  Usage (from repo root):
#    bash run_poc.sh
#
#  WARNING: Mac/MPS replication of the paper is SLOW.
#  Estimated wall-clock on Apple Silicon: 24-48+ hours.
#  Memory: gpt2-xl + AdamW + activations needs ~20-24 GB unified RAM.
#  If you have <32 GB, this will OOM during target fine-tuning.
#
#  To revert to the smaller smoke-test settings, run:
#    cp run_poc.sh.bak run_poc.sh
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO/logs"
VENV="$REPO/.venv"
SESSION="spv-poc"

mkdir -p "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         SPV-MIA  ·  Overnight POC Runner         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Prevent Mac from sleeping ──────────────────────────
echo "[1/6] Keeping Mac awake with caffeinate..."
caffeinate -i -w $$ &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null; echo 'caffeinate stopped'" EXIT

# ── 2. Ensure tmux is installed ───────────────────────────
echo "[2/6] Checking tmux..."
if ! command -v tmux &>/dev/null; then
    echo "  tmux not found — installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew not found. Install it first: https://brew.sh"
        exit 1
    fi
    brew install tmux
fi
echo "  tmux: OK"

# ── 3. Kill any existing session with same name ───────────
tmux kill-session -t "$SESSION" 2>/dev/null && echo "  Killed old '$SESSION' session" || true

# ── 4. Python venv ────────────────────────────────────────
echo "[3/6] Setting up Python virtual environment..."
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi

# ── 5. Write the actual pipeline script ───────────────────
echo "[4/6] Writing pipeline script..."
cat > "$REPO/_run_pipeline.sh" << 'PIPELINE'
#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
LOG="$REPO/logs/pipeline.log"
MODEL="gpt2-xl"
DATASET="ag_news"
TARGET_OUT="$REPO/ft_llms/gpt2-xl/ag_news/target"
REFER_OUT="$REPO/ft_llms/gpt2-xl/ag_news/refer"
CACHE="$REPO/cache"
ACCEL_CFG="$REPO/accelerate_config_mac.yaml"

# Redirect all output to log AND terminal
exec > >(tee -a "$LOG") 2>&1

log() { echo ""; echo "$(date '+%H:%M:%S') ══ $1 ══"; echo ""; }

# ── Activate venv ─────────────────────────────────────────
source "$VENV/bin/activate"

log "STEP 0 · Installing dependencies"
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio --quiet
pip install -r "$REPO/requirements_mac.txt" --quiet
# NLTK data needed by nlpaug
python3 -c "
import nltk
for pkg in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','omw-1.4']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass
"
echo "  Dependencies: OK"

# ── Verify MPS ────────────────────────────────────────────
python3 -c "
import torch
mps = torch.backends.mps.is_available()
print(f'  PyTorch: {torch.__version__}')
print(f'  MPS (Apple GPU): {\"✅ YES\" if mps else \"⚠️  NO — will use CPU\"}')
ram = __import__('subprocess').check_output(['sysctl','-n','hw.memsize']).decode().strip()
print(f'  RAM: {int(ram)//1024//1024//1024} GB')
"

# ── Write accelerate config ────────────────────────────────
mkdir -p ~/.cache/huggingface/accelerate
cp "$ACCEL_CFG" ~/.cache/huggingface/accelerate/default_config.yaml

# ── STEP 1 · Fine-tune target model ───────────────────────
log "STEP 1 · Fine-tuning TARGET model (gpt2-xl on ag_news, 10 epochs, FULL FT, eff_batch=16)"
mkdir -p "$TARGET_OUT"
cd "$REPO"

# Paper recipe: gpt2-xl, full fine-tuning (no LoRA), 10 epochs, lr=1e-4, eff_batch=16.
# On Mac MPS we use per-device batch=1 + grad_accum=16 to keep memory minimal.
accelerate launch --config_file "$ACCEL_CFG" ft_llms/llms_finetune.py \
    -m "$MODEL" \
    -d "$DATASET" \
    --output_dir "$TARGET_OUT" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --eval_steps 200 \
    --save_epochs 1000 \
    --log_steps 50 \
    -e 10 \
    -b 1 \
    --gradient_accumulation_steps 16 \
    -lr 1e-4 \
    --disable_peft \
    --disable_flash_attention \
    --use_dataset_cache \
    --packing \
    --train_sta_idx 0 --train_end_idx 10000 \
    --eval_sta_idx 0  --eval_end_idx 1000 \
    -s 2

echo "  Target model training complete."

# ── Find latest target checkpoint ─────────────────────────
TARGET_CKPT=$(ls -td "$TARGET_OUT"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$TARGET_CKPT" ]; then
    echo "ERROR: No checkpoint found in $TARGET_OUT"
    exit 1
fi
echo "  Latest checkpoint: $TARGET_CKPT"

# ── STEP 2 · Generate self-prompt reference data ──────────
log "STEP 2 · Generating self-prompt reference dataset"
accelerate launch --config_file "$ACCEL_CFG" ft_llms/refer_data_generate.py \
    -tm "$TARGET_CKPT" \
    -m "$MODEL" \
    -d "$DATASET" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --use_dataset_cache \
    --packing

echo "  Reference data generation complete."

# ── STEP 3 · Fine-tune reference model ────────────────────
log "STEP 3 · Fine-tuning REFERENCE model (gpt2-xl, 4 epochs, FULL FT, eff_batch=16, lr=5e-5)"
mkdir -p "$REFER_OUT"
# Paper recipe: same architecture & full FT as target, but 4 epochs and lr=5e-5.
accelerate launch --config_file "$ACCEL_CFG" ft_llms/llms_finetune.py --refer \
    -m "$MODEL" \
    -d "$DATASET" \
    --output_dir "$REFER_OUT" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --eval_steps 100 \
    --save_epochs 500 \
    --log_steps 50 \
    -e 4 \
    -b 1 \
    --gradient_accumulation_steps 16 \
    -lr 5e-5 \
    --disable_peft \
    --disable_flash_attention \
    --use_dataset_cache \
    --packing \
    --train_sta_idx 0 --train_end_idx 10000 \
    --eval_sta_idx 0  --eval_end_idx 1000 \
    -s 2

echo "  Reference model training complete."

# ── Find latest reference checkpoint ──────────────────────
REFER_CKPT=$(ls -td "$REFER_OUT"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$REFER_CKPT" ]; then
    echo "ERROR: No checkpoint found in $REFER_OUT"
    exit 1
fi
echo "  Latest ref checkpoint: $REFER_CKPT"

# ── STEP 4 · Update config.yaml with real paths ───────────
log "STEP 4 · Updating config.yaml with checkpoint paths"
python3 - "$TARGET_CKPT" "$REFER_CKPT" << 'PYEOF'
import sys, yaml, pathlib
target = sys.argv[1]
refer  = sys.argv[2]
cfg_path = pathlib.Path("configs/config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["model_name"]      = "gpt2-xl"
cfg["target_model"]    = target
cfg["reference_model"] = refer
cfg["dataset_name"]    = "ag_news"
cfg["load_attack_data"] = False
cfg["calibration"]     = True       # full SPV-MIA (PDC on, paper Table 1)
cfg["attack_kind"]     = "stat"     # the statistical attack reported in Table 1
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"  config.yaml updated:")
print(f"    model_name:      gpt2-xl")
print(f"    target_model:    {target}")
print(f"    reference_model: {refer}")
print(f"    calibration:     True (PDC on)")
print(f"    attack_kind:     stat (paper Table 1)")
PYEOF

# ── STEP 5 · Run the attack ────────────────────────────────
log "STEP 5 · Running SPV-MIA attack"
cd "$REPO"
python3 attack.py

log "DONE · Check logs/pipeline.log for full output"
echo ""
echo "  ✅ Pipeline complete! AUC result is printed above."
echo "  Results saved to: $REPO/attack/"
echo ""
PIPELINE

chmod +x "$REPO/_run_pipeline.sh"
echo "  Pipeline script written."

# ── 6. Launch tmux session ────────────────────────────────
echo "[5/6] Launching tmux session '$SESSION'..."
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Nice status bar
tmux set-option -t "$SESSION" status on
tmux set-option -t "$SESSION" status-right "#[fg=green]SPV-MIA POC#[default] | %H:%M"
tmux set-option -t "$SESSION" status-interval 10

# Send the pipeline command
tmux send-keys -t "$SESSION" "cd '$REPO' && bash _run_pipeline.sh" Enter

echo "[6/6] Done! Session '$SESSION' is running."
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  You can now close this terminal and go to bed.  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Watch progress:   tmux attach -t spv-poc        ║"
echo "║  Detach (no kill): Ctrl+B  then  D               ║"
echo "║  Check logs:       tail -f logs/pipeline.log     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Attach so they can see it starting before they leave
tmux attach -t "$SESSION"
