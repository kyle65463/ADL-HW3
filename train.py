import argparse
import math
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AdamW

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--max_source_length",
        type=str,
        default=1024,
    )
    parser.add_argument(
        "--max_target_length",
        type=str,
        default=128,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=2,
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output",
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="data/train.jsonl",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def preprocess_function(examples):
        inputs = examples['maintext']
        targets = examples['title']
        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_source_length, 
            padding="max_length", 
            truncation=True
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=args.max_target_length, 
                padding="max_length", 
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # train_data = read_jsonl('data/train.jsonl')
    accelerator = Accelerator(fp16=True)
    raw_datasets = load_dataset("json", data_files={"train": args.train_file})
    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_datasets["train"]
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for epoch in range(args.num_epochs):
        print(f"epoch {epoch + 1}:")
        train_loss = []
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            train_loss.append(loss.detach().float())
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step == len(train_dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()
        train_loss = np.mean(train_loss)
        print(f"train loss: {train_loss:.4f}")

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir, save_function=accelerator.save
    )
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
