#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
LOG="$REPO/logs/pipeline.log"
# Paper Section 5.1: gpt2 (124M) on wikitext-103 with LoRA → AUC=0.975, TPR@1%FPR=0.673
MODEL="gpt2"
DATASET="wikitext"
DATASET_CONFIG="wikitext-103-raw-v1"
TARGET_OUT="$REPO/ft_llms/$MODEL/$DATASET_CONFIG/target"
REFER_OUT="$REPO/ft_llms/$MODEL/$DATASET_CONFIG/refer"
CACHE="$REPO/cache"
ACCEL_CFG="$REPO/accelerate_config_mac.yaml"

# Create logs dir before redirecting output
mkdir -p "$REPO/logs"
exec > >(tee -a "$LOG") 2>&1

log() { echo ""; echo "$(date '+%H:%M:%S') ══ $1 ══"; echo ""; }

# ── Detect hardware before venv exists ────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    DEVICE="cuda"
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_TAG="cu$(echo "$CUDA_VER" | tr -d '.' | cut -c1-3)"
    TORCH_INSTALL="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_TAG} --quiet"
    REQUIREMENTS="$REPO/requirements_gpu.txt"
    echo "  Detected CUDA $CUDA_VER"
else
    DEVICE="mps"
    # Plain pip install gives the MPS-enabled build on macOS; the /whl/cpu index is CPU-only.
    TORCH_INSTALL="pip install torch torchvision torchaudio --quiet"
    REQUIREMENTS="$REPO/requirements_mac.txt"
    echo "  Detected macOS — will use MPS (Apple GPU)"
fi

# ── Create venv if missing ─────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

log "STEP 0 · Installing dependencies"
python3 -m pip install --upgrade pip --quiet
TORCH_INSTALL="${TORCH_INSTALL/pip /python3 -m pip }"
eval "$TORCH_INSTALL"
python3 -m pip install -r "$REQUIREMENTS" --quiet
python3 -c "
import nltk
for pkg in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','omw-1.4']:
    try: nltk.download(pkg, quiet=True)
    except: pass
"

# ── Confirm device and set training hyperparameters ───────
DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available(): print('cuda')
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): print('mps')
else: print('cpu')
" 2>/dev/null)

python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  PyTorch {torch.__version__}')
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory//1024**3} GB)')
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
    import subprocess
    ram = int(subprocess.check_output(['sysctl','-n','hw.memsize']).decode().strip())
    print(f'  PyTorch {torch.__version__} | MPS (Apple GPU) | RAM: {ram//1024**3} GB')
else:
    print(f'  PyTorch {torch.__version__} | CPU only (no GPU found)')
"

# Paper uses LoRA (default PEFT) with batch=16. GPT-2 (124M) fits easily in
# MPS/CUDA memory with LoRA, and SPV-MIA's self-prompt calibration works with it.
if [ "$DEVICE" = "cuda" ]; then
    BATCH=16
    GRAD_ACCUM=1
    ACCEL_LAUNCH="python3 -m accelerate.commands.launch"
    echo "  Mode: CUDA LoRA fine-tuning"
else
    BATCH=16
    GRAD_ACCUM=1
    ACCEL_LAUNCH="python3 -m accelerate.commands.launch --config_file $ACCEL_CFG"
    mkdir -p ~/.cache/huggingface/accelerate
    cp "$ACCEL_CFG" ~/.cache/huggingface/accelerate/default_config.yaml
    echo "  Mode: MPS LoRA fine-tuning"
fi

# Paper Section 5.1 hyperparameters (Table 1/2 results)
TARGET_LR="1e-4"
REFER_LR="5e-5"
TARGET_EPOCHS=10
REFER_EPOCHS=4

# ── STEP 1 · Fine-tune target model ───────────────────────
log "STEP 1 · Fine-tuning TARGET model ($MODEL on $DATASET_CONFIG, $TARGET_EPOCHS epochs, full fine-tune)"
mkdir -p "$TARGET_OUT"
cd "$REPO"

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

echo "  Target fine-tuning complete."

# ── Find latest target checkpoint ─────────────────────────
TARGET_CKPT=$(ls -td "$TARGET_OUT"/checkpoint-* 2>/dev/null | head -1)
[ -z "$TARGET_CKPT" ] && { echo "ERROR: no checkpoint in $TARGET_OUT"; exit 1; }
echo "  Checkpoint: $TARGET_CKPT"

# ── STEP 2 · Generate self-prompt reference data ──────────
log "STEP 2 · Generating self-prompt reference dataset"
$ACCEL_LAUNCH ft_llms/refer_data_generate.py \
    -tm "$TARGET_CKPT" \
    -m "$MODEL" \
    -d "$DATASET" \
    -dc "$DATASET_CONFIG" \
    --cache_path "$CACHE" \
    --block_size 128 \
    --use_dataset_cache \
    --packing

echo "  Reference data generation complete."

# ── STEP 3 · Fine-tune reference model ────────────────────
log "STEP 3 · Fine-tuning REFERENCE model ($REFER_EPOCHS epochs)"
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

echo "  Reference fine-tuning complete."

# ── Find latest reference checkpoint ──────────────────────
REFER_CKPT=$(ls -td "$REFER_OUT"/checkpoint-* 2>/dev/null | head -1)
[ -z "$REFER_CKPT" ] && { echo "ERROR: no checkpoint in $REFER_OUT"; exit 1; }
echo "  Checkpoint: $REFER_CKPT"

# ── STEP 4 · Write checkpoint paths into config ───────────
log "STEP 4 · Updating config.yaml"
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

# ── STEP 5 · Run the attack ────────────────────────────────
log "STEP 5 · Running SPV-MIA attack"
cd "$REPO"
python3 attack.py

log "DONE"
echo ""
echo "  Pipeline complete. AUC result is above."
echo "  Results: $REPO/attack/"
echo ""
