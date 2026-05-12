import os
import random
import datasets
import trl
from attack.utils import create_folder


def _load_local_or_hub(name, config, split):
    """If env var SPV_MIA_LOCAL_DATASET_PATH is set, load from that directory
    via load_from_disk and slice. Otherwise fall back to datasets.load_dataset
    (HF Hub or builtin script).

    Used by the post-cutoff Wikipedia experiment (see
    `post_cutoff_wiki_test/`). Set SPV_MIA_LOCAL_DATASET_PATH to the directory
    produced by `02_format.py`; `dataset_name` becomes a label used only for
    the cache path.
    """
    local_path = os.environ.get("SPV_MIA_LOCAL_DATASET_PATH", "").strip()
    if local_path:
        loaded = datasets.load_from_disk(local_path)
        if isinstance(loaded, datasets.DatasetDict):
            base = loaded["train"]
        else:
            base = loaded
        # Reproduce the `split="train[:N%]"` semantics used below.
        s = split.replace("train", "").strip("[]")
        if not s:
            return base
        if ":" not in s:
            raise ValueError(f"unsupported slice for local dataset: {split}")
        lo, hi = s.split(":")
        n = len(base)
        def _resolve(tok, default):
            if tok == "":
                return default
            if tok.endswith("%"):
                return int(n * int(tok[:-1]) / 100)
            return int(tok)
        lo = _resolve(lo, 0)
        hi = _resolve(hi, n)
        return base.select(range(lo, hi))
    return datasets.load_dataset(name, config, split=split)

block_size = None
tokenizer_ = None
max_buff_size = None
text_column = None

def packing_texts(examples):
    more_examples = True
    packed_texts = []
    packed_ids = []
    # for key in examples.keys():
    assert list(examples.keys()) == ["text"]
    iterator = iter(examples["text"])
    # for sentence in examples["text"]:
    total_num = 0
    drop_num = 0
    while more_examples:
        buffer, buffer_len = [], 0
        while True:
            if buffer_len >= max_buff_size:
                break
            try:
                buffer.append(next(iterator))
                buffer_len += len(buffer[-1])
            except StopIteration:
                more_examples = False
                break
        tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
        inputs = tokenizer_.batch_decode(tokenized_inputs)
        tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend(tokenized_input)
        for i in range(0, len(all_token_ids), block_size):
            input_ids = all_token_ids[i: i + block_size]
            if len(input_ids) == block_size:
                packed_ids.append(input_ids)
                input_text = tokenizer_.decode(input_ids)
                total_num += 1
                if len(tokenizer_.encode(input_text)) == block_size:
                    packed_texts.append(input_text)
                    drop_num += 1
    # print(f"Total examples: {total_num}, dropped num: {drop_num}, dropped rate: {1 - drop_num/total_num}")
    return {
        "text": packed_texts
    }
def dataset_prepare(args, tokenizer=None, num_of_sequences=1024, chars_per_token=3.6):
    # raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)['train']
    # if "validation" in raw_datasets.keys():
    #     train_dataset = raw_datasets["train"]
    #     valid_dataset = raw_datasets["validation"]
    # else:
    train_dataset = _load_local_or_hub(
        args.dataset_name,
        args.dataset_config_name,
        split=f"train[:{int((1-args.validation_split_percentage)*100)}%]",
    )
    valid_dataset = _load_local_or_hub(
        args.dataset_name,
        args.dataset_config_name,
        split=f"train[{int((1-args.validation_split_percentage)*100)}%:]",
    )

    # train_idxs = set(random.sample(range(len(raw_datasets)), int(len(raw_datasets) * (1 - args.validation_split_percentage))))
    # valid_idxs = set(range(len(raw_datasets))) - train_idxs
    # train_dataset = datasets.Dataset.from_dict(raw_datasets[train_idxs])
    # valid_dataset = datasets.Dataset.from_dict(raw_datasets[valid_idxs])


    global text_column
    column = train_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"

    train_dataset = train_dataset.select_columns(text_column)
    valid_dataset = valid_dataset.select_columns(text_column)
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")

    if args.packing:
        global block_size, tokenizer_, max_buff_size
        block_size = args.block_size
        max_buff_size = block_size * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        create_folder(f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}")
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/train_dataset.arrow",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=args.preprocessing_num_workers,
            cache_file_name=f"{args.cache_path}/{args.dataset_name}/{args.dataset_config_name}/valid_dataset.arrow",
            load_from_cache_file=args.use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
        return train_dataset, valid_dataset