#!/usr/bin/env bash
# ============================================================
# setup_mac.sh — one-shot environment setup for SPV-MIA on Mac
# Apple Silicon (M1/M2/M3) with MPS GPU acceleration
# ============================================================
set -e

echo "==> Checking Python version..."
python3 --version

echo ""
echo "==> Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "==> Installing PyTorch with MPS support..."
# PyTorch >= 2.0 ships with MPS support built in.
pip install --upgrade pip
pip install torch torchvision torchaudio

echo ""
echo "==> Installing project dependencies (Mac-compatible)..."
pip install -r requirements_mac.txt

echo ""
echo "==> Downloading NLTK data needed by nlpaug..."
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

echo ""
echo "==> Verifying MPS availability..."
python3 - <<'EOF'
import torch
if torch.backends.mps.is_available():
    print("✅  MPS (Apple GPU) is available — models will run on Metal.")
else:
    print("⚠️  MPS not available — falling back to CPU. "
          "Make sure you are on macOS 12.3+ with an Apple Silicon chip.")
EOF

echo ""
echo "==> Setting up Accelerate for MPS (single-device)..."
mkdir -p ~/.cache/huggingface/accelerate
cp accelerate_config_mac.yaml ~/.cache/huggingface/accelerate/default_config.yaml
echo "    Accelerate config written to ~/.cache/huggingface/accelerate/default_config.yaml"

echo ""
echo "============================================================"
echo "  Setup complete!  Activate your environment with:"
echo "    source .venv/bin/activate"
echo ""
echo "  Quick-start fine-tuning (GPT-2 on ag_news, ~10 min on M1):"
echo "    accelerate launch ft_llms/llms_finetune.py \\"
echo "      -m gpt2 -d ag_news \\"
echo "      --output_dir ./ft_llms/gpt2/ag_news/target/ \\"
echo "      --block_size 128 --eval_steps 100 --save_epochs 100 \\"
echo "      -e 5 -b 2 -lr 1e-4 --disable_peft \\"
echo "      --train_sta_idx 0 --train_end_idx 5000 \\"
echo "      --eval_sta_idx 0  --eval_end_idx 500"
echo ""
echo "  Then run the attack:"
echo "    python attack.py"
echo "============================================================"
