import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb

from tw_rouge import get_rouge

from utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument("--model_name", type=str, default="google/mt5-small")

    # data
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--max_answer_len", type=int, default=64)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--scheduler", type=str, default="cosine")

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--accu_step", type=int, default=8)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    print(accelerator.device)

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, do_lower_case=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)

    model.resize_token_embeddings(len(tokenizer))

    raw_dataset = load_dataset(
        "json", data_files={"train": os.path.join(args.data_dir, "train.jsonl")}
    )
    cols = raw_dataset["train"].column_names
    train_prep = preprocess_function(tokenizer=tokenizer, args=args)
    tokenized_datasets = raw_dataset.map(
        train_prep, batched=True, keep_in_memory=True, num_proc=8, remove_columns=cols
    )
    if args.validate:
        dataset = tokenized_datasets["train"].train_test_split(0.1, shuffle=False)
    else:
        dataset = tokenized_datasets
    print(dataset)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_loader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    if args.validate:
        valid_loader = DataLoader(
            dataset["test"],
            collate_fn=data_collator,
            batch_size=args.batch_size,
            pin_memory=True,
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.validate:
        model, optimizer, train_loader, valid_loader = accelerator.prepare(
            model, optimizer, train_loader, valid_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    total = len(train_loader) * args.num_epoch
    n_warm = int(0.05 * total)
    n_train = total - n_warm
    scheduler = None  # get_scheduler(
    # args.scheduler, optimizer, num_warmup_steps=n_warm, num_training_steps=n_train
    # )

    if args.wandb:
        wandb.watch(model)

    best_score = 0
    best_loss = np.inf
    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch}:")
        # Training
        train_loss = []
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.accu_step
            train_loss.append(loss.item())
            accelerator.backward(loss)

            if ((step + 1) % args.accu_step == 0) or (step == len(train_loader) - 1):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
        train_loss = np.mean(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        # Evaluation
        if args.validate:
            model.eval()
            all_preds, all_labels, valid_loss = [], [], []
            for step, batch in enumerate(tqdm(valid_loader)):
                with torch.no_grad():
                    outputs = model(**batch)
                    valid_loss.append(outputs.loss.item())
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=args.max_answer_len,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]

                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                    )

                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )
                    labels = accelerator.gather(labels).cpu().numpy()

                    # Replace -100 in the labels as we can't decode them
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    decoded_labels = tokenizer.batch_decode(
                        labels, skip_special_tokens=True
                    )

                    decoded_preds, decoded_labels = postprocess_text(
                        decoded_preds, decoded_labels
                    )
                    all_preds += decoded_preds
                    all_labels += decoded_labels

            rouge_score = get_rouge(all_preds, all_labels)
            print(rouge_score)
            valid_r1 = rouge_score["rouge-1"]["f"]
            valid_r2 = rouge_score["rouge-2"]["f"]
            valid_rL = rouge_score["rouge-l"]["f"]
            mean_score = np.mean([valid_r1, valid_r2, valid_rL])

            valid_loss = np.mean(valid_loss)

            print(f"Valid Loss: {valid_loss:.4f}")
            print(
                f"rouge_r1: {valid_r1:.4f}\nrouge_r2: {valid_r2:.4f}\nrouge_rL: {valid_rL:.4f}"
            )
            if args.wandb:
                wandb.log(
                    {
                        "Train Loss": train_loss,
                        "Validation Loss": valid_loss,
                        "rouge r1": valid_r1,
                        "rouge r2": valid_r2,
                        "rouge rL": valid_rL,
                    }
                )
            if mean_score > best_score:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.ckpt_dir, save_function=accelerator.save
                )
                tokenizer.save_pretrained(args.ckpt_dir)
        else:
            if train_loss < best_loss:
                best_loss = train_loss
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.ckpt_dir, save_function=accelerator.save
                )
                tokenizer.save_pretrained(args.ckpt_dir)


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    if args.wandb:
        wandb.login()
        wandb.init(project="News Summarization")
        wandb.config.update(args)
    main(args)