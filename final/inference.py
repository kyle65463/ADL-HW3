import json
from argparse import ArgumentParser
from pathlib import Path

from accelerate import Accelerator
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        default="./data/public.jsonl",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="./model/",
    )
    parser.add_argument(
        "--output_file",
        type=Path, 
        default="./test.jsonl"
    )
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    accelerator = Accelerator()
    print(accelerator.device)
    config = {
        "greedy": {"do_sample": False, "num_beams": 1},
        "beam": {"do_sample": False, "num_beams": args.num_beams},
        "top_k": {"top_k": args.top_k},
        "top_p": {"top_p": args.top_p},
        "temparature": {"temperature": args.temperature},
    }
    print(config)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["maintext"],
            max_length=args.max_source_len,
            padding="max_length",
            truncation=True,
        )
        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    raw_datasets = load_dataset("json", data_files={"test": args.input_file})
    column_names = raw_datasets["test"].column_names
    ids = raw_datasets["test"]["id"]
    processed_dataset = raw_datasets.map(
        preprocess_function, batched=True, keep_in_memory=True, num_proc=8, remove_columns=column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    test_loader = DataLoader(
        processed_dataset["test"],
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        return preds

    preds = []
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()
    for _, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **config,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds = postprocess_text(decoded_preds)

            preds += decoded_preds

    with open(args.output_file, "w") as f:
        for id, pred in zip(ids, preds):
            print(json.dumps({"title": pred, "id": id}), file=f)


if __name__ == "__main__":
    main()
