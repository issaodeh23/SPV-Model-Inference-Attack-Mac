import argparse

import datasets
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, AutoConfig, Trainer, DataCollatorForLanguageModeling)
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import torch
import logging
import os
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig, IA3Config
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from attack.utils import get_device as _get_device, get_torch_dtype as _get_torch_dtype, is_quantization_supported as _is_quant_supported
import pandas as pd
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data.prepare import dataset_prepare
from attack.utils import create_folder
from transformers import LlamaTokenizer, get_scheduler
import os
# Proxy settings removed (were hardcoded to original author's machine).
# Set HTTP_PROXY / HTTPS_PROXY in your shell environment if you need a proxy.

from utils import get_logger, constantlengthdatasetiter, print_trainable_parameters
# trl.trainer.ConstantLengthDataset.__dict__["__iter__"] = constantlengthdatasetiter
# setattr(trl.trainer.ConstantLengthDataset, "__iter__", constantlengthdatasetiter)

logger = get_logger("finetune", "info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-d", "--dataset_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("-dc", "--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--cache_path", type=str, default="./cache")
    parser.add_argument("--use_dataset_cache", action="store_true", default=False)
    parser.add_argument("--refer", action="store_true", default=False)
    parser.add_argument("--refer_data_source", type=str, default=None)
    parser.add_argument("--packing", action="store_true", default=False)
    parser.add_argument("-t", "--token", type=str, default=None)
    parser.add_argument("--split_model", action="store_true", default=False)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--peft", type=str, default="lora")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--p_tokens", type=int, help="The number of virtual tokens for prefix-tuning or p-tuning", default=20)
    parser.add_argument("--p_hidden", type=int, help="The hidden size of the prompt encoder", default=128)

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--output_dir", type=str, default="./ft_llms/checkpoints")
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)

    parser.add_argument("--train_sta_idx", type=int, default=0)
    parser.add_argument("--train_end_idx", type=int, default=6000)
    parser.add_argument("--eval_sta_idx", type=int, default=0)
    parser.add_argument("--eval_end_idx", type=int, default=600)

    parser.add_argument("-s", "--save_limit", type=int, default=None)

    parser.add_argument("--use_int4", action="store_true", default=False)
    parser.add_argument("--use_int8", action="store_true", default=False)
    parser.add_argument("--disable_peft", action="store_true", default=False)
    parser.add_argument("--disable_flash_attention", action="store_true", help="Disable flash attention", default=False)

    parser.add_argument("--pad_token_id", default=None, type=int, help="The end of sequence token.")
    parser.add_argument("--add_eos_token", action="store_true", help="Add EOS token to tokenizer", default=False)
    parser.add_argument("--add_bos_token", action="store_true", help="Add BOS token to tokenizer", default=False)
    parser.add_argument("--validation_split_percentage", default=0.1, help="The percentage of the train set used as validation set in case there's no validation split")
    args = parser.parse_args()

    accelerator = Accelerator()

    if args.token is None:
        access_token = os.getenv("HF_TOKEN") or None  # None = no auth header (needed for public models)
    else:
        access_token = args.token

    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_path)

    config.use_cache = False
    config_dict = config.to_dict()
    model_type = config_dict["model_type"]

    use_flash_attention = False

    # Flash attention requires CUDA with compute capability >= 8 (Ampere+).
    # It is not available on MPS (Apple Silicon) or CPU.
    if not args.disable_flash_attention and model_type != "llama":
        logger.info("Model is not llama, disabling flash attention...")
    elif args.disable_flash_attention and model_type == "llama":
        logger.info("Model is llama, could be using flash attention...")
    elif (not args.disable_flash_attention
          and model_type == "llama"
          and torch.cuda.is_available()
          and torch.cuda.get_device_capability()[0] >= 8):
        from ft_llms.llama_patch import replace_attn_with_flash_attn
        logger.info("Using flash attention for llama...")
        replace_attn_with_flash_attn()
        use_flash_attention = True
    else:
        logger.info("Flash attention not available on this device (MPS/CPU or CUDA < 8.0), skipping.")


    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "GPT_finetuning"

    if args.split_model:
        logger.info("Splitting the model across all available devices...")
        kwargs = {"device_map": "auto"}
    else:
        kwargs = {"device_map": None}

    if model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, token=access_token,
                                                  trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                  add_eos_token=args.add_eos_token, add_bos_token=args.add_bos_token,
                                                  use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token,
                                                  trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                  add_eos_token=args.add_eos_token, add_bos_token=args.add_bos_token,
                                                  use_fast=True)
    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    # good one for LLama is 18610
    if args.pad_token_id is not None:
        logger.info("Using pad token id %d", args.pad_token_id)
        tokenizer.pad_token_id = args.pad_token_id

    if tokenizer.pad_token_id is None:
        logger.info("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    block_size = args.block_size
    logger.info("Using a block size of %d", block_size)

    _device = _get_device()
    torch_dtype = _get_torch_dtype(_device)
    _quant_ok = _is_quant_supported()

    if args.use_int4 and _quant_ok:
        logger.info("Using int4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        optimizer = "adamw_bnb_8bit"
    elif args.use_int8 and _quant_ok:
        logger.info("Using int8 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        optimizer = "adamw_bnb_8bit"
    else:
        if (args.use_int4 or args.use_int8) and not _quant_ok:
            logger.info("Quantization requested but not supported on this device (MPS/CPU). Falling back to no quantization.")
        else:
            logger.info("Using no quantization")
        bnb_config = None
        optimizer = "adamw_torch"

    if args.peft == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    elif args.peft == "prefix-tuing":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=args.p_tokens,
            encoder_hidden_size=args.p_hidden)
    elif args.peft == "p-tuing":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.p_tokens,
            encoder_hidden_size=args.p_hidden)
    elif args.peft == "ia3":
        peft_config = IA3Config(
            peft_type="IA3",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=access_token, quantization_config=bnb_config,
                                                 trust_remote_code=args.trust_remote_code, cache_dir=args.cache_path,
                                                 torch_dtype=torch_dtype, config=config, **kwargs)

    if use_flash_attention:
        from ft_llms.llama_patch import llama_forward_with_flash_attn

        assert model.model.layers[
                   0].self_attn.forward.__doc__ == llama_forward_with_flash_attn.__doc__, "Model is not using flash attention"

    if not args.disable_peft:
        logger.info("Using PEFT...")
        if args.use_int4 or args.use_int8:
            logger.info("Preparing model for kbit training...")
            model = prepare_model_for_kbit_training(model)
            if use_flash_attention:
                from ft_llms.llama_patch import upcast_layer_for_flash_attention
                logger.info("Upcasting flash attention layers...")
                model = upcast_layer_for_flash_attention(model, torch_dtype)
        logger.info("Getting PEFT model...")
        model = get_peft_model(model, peft_config)
    else:
        logger.info("Using Full Finetuning")

    print_trainable_parameters(model)

    if args.refer_data_source is not None:
        args.model_name = args.refer_data_source
    if args.model_name == "/mnt/data0/fuwenjie/MIA-LLMs/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348":
        args.model_name = "decapoda-research/llama-7b-hf"
    with accelerator.main_process_first():
        train_dataset, valid_dataset = dataset_prepare(args, tokenizer=tokenizer)
        if args.refer:
            train_dataset = None
            refer_data_path = f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/refer@{args.model_name}/"
            train_dataset = load_from_disk(refer_data_path)
        train_dataset = Dataset.from_dict(train_dataset[args.train_sta_idx:args.train_end_idx])
        valid_dataset = Dataset.from_dict(valid_dataset[args.eval_sta_idx:args.eval_end_idx])
        # train_dataset = load_from_disk("/mnt/data0/fuwenjie/MIA-LLMs/cache/ag_news/None/refer@gpt2")

    # bf16/fp16 flags: only valid on CUDA. MPS uses its own mixed-precision.
    _use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    _use_fp16 = torch.cuda.is_available() and not _use_bf16
    # On MPS both flags must be False — HuggingFace Trainer handles MPS natively.

    logger.info(f"Training with {Accelerator().num_processes} device(s) | device={_device} | dtype={torch_dtype}")

    # eval_strategy replaces deprecated evaluation_strategy in transformers >= 4.46
    import transformers as _transformers
    _tf_version = tuple(int(x) for x in _transformers.__version__.split(".")[:2])
    _eval_strategy_key = "eval_strategy" if _tf_version >= (4, 46) else "evaluation_strategy"

    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        **{_eval_strategy_key: "steps"},
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_epochs,
        logging_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        optim=optimizer,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        weight_decay=args.weight_decay,
        adam_epsilon=1e-6,
        report_to="none",  # set WANDB_PROJECT env var to re-enable wandb
        load_best_model_at_end=False,
        save_total_limit=args.save_limit,
        bf16=_use_bf16,
        fp16=_use_fp16,
    )

    # Tokenize datasets — replaces SFTTrainer which has breaking API changes across trl versions.
    # Standard HuggingFace Trainer is stable and does the same thing for causal LM fine-tuning.
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=block_size, padding=False)

    logger.info("Tokenizing train dataset...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    logger.info("Tokenizing eval dataset...")
    valid_dataset = valid_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # get trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # train
    trainer.train()