import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import random

from attack.attack_model import AttackModel
from data.prepare import dataset_prepare
from attack.utils import Dict, get_device, get_torch_dtype, is_quantization_supported

import yaml
import datasets
from datasets import Image, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Load config file
with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)
    cfg = Dict(cfg)

# Add Logger
accelerator = Accelerator()
logger = get_logger(__name__, "INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))

# Fix the random seed (device-agnostic)
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _load_model(base_name, ckpt_path, base_kwargs, device):
    """Load a fine-tuned model. Auto-detects LoRA adapter vs full fine-tuned checkpoint."""
    if os.path.isfile(os.path.join(ckpt_path, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(base_name, **base_kwargs)
        return PeftModel.from_pretrained(base, ckpt_path, is_trainable=False).to(device)
    return AutoModelForCausalLM.from_pretrained(ckpt_path, **base_kwargs).to(device)


## Load generation models.
if not cfg["load_attack_data"]:
    _device = get_device()
    torch_dtype = get_torch_dtype(_device)
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if is_quantization_supported() else None

    _base_kwargs = dict(torch_dtype=torch_dtype, cache_dir=cfg["cache_path"])
    if quant_config is not None:
        _base_kwargs["quantization_config"] = quant_config

    target_model = _load_model(cfg["model_name"], cfg["target_model"], _base_kwargs, accelerator.device)
    reference_model = _load_model(cfg["model_name"], cfg["reference_model"], _base_kwargs, accelerator.device)


    logger.info("Successfully load models")
    config = AutoConfig.from_pretrained(cfg.model_name)
    # Load tokenizer.
    model_type = config.to_dict()["model_type"]
    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                                  add_bos_token=cfg["add_bos_token"], use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], add_eos_token=cfg["add_eos_token"],
                                                  add_bos_token=cfg["add_bos_token"], use_fast=True)

    if cfg["model_name"] == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        cfg["model_name"] = "decapoda-research/llama-7b-hf"

    if cfg["pad_token_id"] is not None:
        logger.info("Using pad token id %d", cfg["pad_token_id"])
        tokenizer.pad_token_id = cfg["pad_token_id"]

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    train_dataset, valid_dataset = dataset_prepare(cfg, tokenizer=tokenizer)
    train_dataset = Dataset.from_dict(train_dataset[cfg.train_sta_idx:cfg.train_end_idx])
    valid_dataset = Dataset.from_dict(valid_dataset[cfg.eval_sta_idx:cfg.eval_end_idx])
    train_dataset = Dataset.from_dict(train_dataset[random.sample(range(len(train_dataset["text"])), cfg["maximum_samples"])])
    valid_dataset = Dataset.from_dict(valid_dataset[random.sample(range(len(valid_dataset["text"])), cfg["maximum_samples"])])
    logger.info("Successfully load datasets!")

    # Prepare dataloade
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["eval_batch_size"])
    eval_dataloader = DataLoader(valid_dataset, batch_size=cfg["eval_batch_size"])

    # Load Mask-filling model (T5-base by default).
    # On MPS/CPU, 8-bit quantization is unavailable; fall back to float16.
    shadow_model = None
    int8_kwargs = {}
    half_kwargs = {}
    if cfg["int8"] and is_quantization_supported():
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch_dtype)
    elif cfg["half"] or (not is_quantization_supported()):
        half_kwargs = dict(torch_dtype=torch_dtype)
    # T5 mask-filling runs on CPU — MPS generation produces empty fills, breaking the perturbation signal.
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(cfg["mask_filling_model_name"], **int8_kwargs, **half_kwargs).to("cpu")
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = AutoTokenizer.from_pretrained(cfg["mask_filling_model_name"], model_max_length=n_positions)

    # Prepare everything with accelerator
    train_dataloader, eval_dataloader = (
        accelerator.prepare(
            train_dataloader,
            eval_dataloader,
    ))
else:
    target_model = None
    reference_model = None
    shadow_model = None
    mask_model = None
    train_dataloader = None
    eval_dataloader = None
    tokenizer = None
    mask_tokenizer = None


datasets = {
    "target": {
        "train": train_dataloader,
        "valid": eval_dataloader
    }
}


attack_model = AttackModel(target_model, tokenizer, datasets, reference_model, shadow_model, cfg, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
attack_model.conduct_attack(cfg=cfg)
