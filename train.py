import json
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import argparse

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
    args = parser.parse_args()
    return args


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
raw_datasets = load_dataset("json", data_files={"train": 'data/train.jsonl'})
column_names = raw_datasets["train"].column_names
processed_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)
train_dataset = processed_datasets["train"]
print(train_dataset[1])
print(tokenizer.pad_token_id)
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
)
