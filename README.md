# SPV-MIA — Mac Port and Post-Cutoff Wikipedia Validation

> 📌 **UROP Spring 2026 — ALFA Group, MIT CSAIL**
>
> - **[Term Progress Report →](./progress_report.md)** — full write-up: BRON context, term readings, cybersecurity LLM benchmark survey, Mac port of SPV-MIA, post-cutoff Wikipedia validation, results.
> - **[Experiment Report →](./post_cutoff_wiki_test/results/REPORT.md)** — standalone report with AUC, ROC plot, sample articles for the post-cutoff Wikipedia run.
>
> **Headline result:** SPV-MIA achieves **AUC = 0.964** on a Wikipedia corpus the base GPT-2 has provably never seen (articles created after 2020-01-01), within 1.1 percentage points of the paper's 0.975 on Wikitext-103. This rules out the concern that the paper's signal comes from pretraining leakage rather than fine-tuning memorization.

---

## What's in this fork

- `_run_pipeline.sh` — end-to-end pipeline (target fine-tune → self-prompt refs → reference fine-tune → attack) that auto-detects MPS / CUDA and uses the right hyperparameters. Matches the paper's Wikitext-103 setup.
- `post_cutoff_wiki_test/` — validation harness that builds a Wikipedia corpus of articles created strictly after GPT-2's training cutoff (2020-01-01) and runs SPV-MIA against it.
- Apple Silicon / MPS fixes throughout: `float32` dtype, no `BitsAndBytesConfig` on non-CUDA, T5 mask-filler pinned to CPU, custom `accelerate_config_mac.yaml`.
- `attack/attack_model.py` — patched to cap T5 mask-fill retries (the upstream implementation has no cap and gets stuck in infinite loops on stubborn texts).

See `REPRODUCTION_NOTES.md` and `EXPERIMENT_REPORT.md` for additional reproduction details from the Wikitext-103 baseline.

---

# Original README (NeurIPS'24)

- [Requirements](#requirements)
- [Target Model Fine-tuning](#target-model-fine-tuning)
- [Self-prompt Reference Model Fine-tuning](#self-prompt-reference-model-fine-tuning)
- [Run SPV-MIA](#run-spv-mia)

This is the official implementation of the paper "Practical Membership Inference Attacks against Fine-tuned 
Large Language Models via Self-prompt Calibration".
The proposed Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA) is implemented as follows.

![The overall architecture of _SPV-MIA_](./Framework.png)

## Requirements

- torch>=1.11.0
- accelerate==0.20.3
- transformers==4.34.0.dev0
- trl==0.7.1
- datasets==2.13.1
- numpy>=1.23.4
- scikit-learn>=1.1.3
- pyyaml>=6.0
- tqdm>=4.64.1

Dependency can be installed with the following command:

```bash
pip install -r requirements.txt
```


## Target Model Fine-tuning
  All large language models (LLMs) are built on the top of [transformers](https://huggingface.co/docs/transformers/index), 
  a go-to library for state-of-the-art transformer models, on which you can fine-tune arbitrary well-known LLMs you want,
  including LLaMA, GPT-series, Falcon, etc.
  We recommend training LLMs with multi-GPU and [accelerate](https://huggingface.co/docs/accelerate/index), 
  a library that enables the same PyTorch code to be run across any distributed configuration:
  ```bash
  accelerate launch ./ft_llms/llms_finetune.py \
  --output_dir ./ft_llms/*pretrained_model_name*/*dataset_name*/target/ \
  --block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
  -d *dataset_name* -m *pretrained_model_name* --packing --use_dataset_cache \
  -e 10 -b 4 -lr 1e-4 --gradient_accumulation_steps 1 \
  --train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000
  ```

Please replace \*pretrained_model_name\* and \*dataset_name\* with the names of pretrained LLM and training dataset, such as `decapoda-research/llama-7b-hf` and `ag_news`.

### Recommended pretrained models
- GPT-2 (1.5B) (https://huggingface.co/gpt2-xl)
- GPT-J (https://huggingface.co/EleutherAI/gpt-j-6b)
- Falcon (https://huggingface.co/tiiuae/falcon-7b)
- LLaMA (https://huggingface.co/decapoda-research/llama-7b-hf) [^1]

[^1]: This third-party repo `decapoda-research/llama-7b-hf` seems to be deleted by unknown reasons, using forked repos [luodian/llama-7b-hf](https://huggingface.co/luodian/llama-7b-hf) 
or [baffo32/decapoda-research-llama-7B-hf](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf) as alternatives.
### Recommended datasets
- Ag News (https://huggingface.co/datasets/ag_news)
- Wikitext-103 (https://huggingface.co/datasets/wikitext) [^2]
- Xsum (https://huggingface.co/datasets/xsum)

[^2]: Please add an additional argument `--dataset_config_name wikitext-2-raw-v1` to specify this dataset.
## Self-prompt Reference Model Fine-tuning
  Before fine-tuning the self-prompt reference model, the reference dataset can be sampled via our proposed 
  self-prompt approach over the fine-tuned LLM. 
  ```bash
  accelerate launch refer_data_generate.py \
  -tm *fine_tuned_model* \
  -m *pretrained_model_name* -d *dataset_name*
  ```
  Replace \*fine_tuned_model\* with the directory of the fine-tuned target model (i.e., the output directory of 
  the [Target Model Fine-tuning](#target-model-fine-tuning) phase). 

 Then fine-tune the self-prompt reference model in the same manner as the target model, but with a smaller training epoch:
```bash
accelerate launch ./ft_llms/llms_finetune.py --refer \
--output_dir ./ft_llms/*pretrained_model_name*/*dataset_name*/refer/ \
--block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
-d *dataset_name* -m *pretrained_model_name* --packing --use_dataset_cache \
-e 2 -b 4 -lr 5e-5 --gradient_accumulation_steps 1 \
--train_sta_idx=0 --train_end_idx=10000 --eval_sta_idx=0 --eval_end_idx=1000
```


## Run SPV-MIA
After accomplishing the preliminary operations, here is the command for deploying SPV-MIA on the target model.
```bash
python attack.py
```
