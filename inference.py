import json
import os
from argparse import ArgumentParser
from pathlib import Path
from unittest import TestLoader

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from utils import same_seeds
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator
from utils import *

from time import time
import datetime


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--test_file",
        type=str,
        help="Directory to the dataset.",
        default="./data/public.jsonl",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument("--out_json", type=Path, default="./test.jsonl")

    # data
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--max_answer_len", type=int, default=64)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--strategy", type=str, choices=STRAT)

    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    return args


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator()
    # print(args)
    config = get_config(args)
    print(config)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_dir)
    raw_dataset = load_dataset("json", data_files={"test": args.test_file})
    prep = preprocess_function(tokenizer=tokenizer, args=args, mode="test")
    cols = raw_dataset["test"].column_names
    ids = raw_dataset["test"]["id"]
    dataset = raw_dataset.map(
        prep, batched=True, keep_in_memory=True, num_proc=8, remove_columns=cols
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    test_loader = DataLoader(
        dataset["test"],
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    model, test_loader = accelerator.prepare(model, test_loader)
    start_time = time()
    print(accelerator.device)
    model.eval()
    preds = []
    for step, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=args.max_answer_len,
                **config,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            decoded_preds, _ = postprocess_text(decoded_preds, [])

            preds += decoded_preds
    end_time = time()
    s_time = str(datetime.timedelta(seconds=round(end_time - start_time)))
    print(f"{s_time} &", end="")

    with open(args.out_json, "w") as f:
        for _id, pred in zip(ids, preds):
            print(json.dumps({"title": pred, "id": _id}), file=f)


if __name__ == "__main__":
    args = parse_args()
    main(args)