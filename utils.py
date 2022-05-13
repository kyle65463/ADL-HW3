import random

import numpy as np
import torch


STRAT = ["greedy", "beam", "top-k", "top-p", "temperature"]


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_function(tokenizer, args, mode="train"):
    def __implementation__(examples):
        model_inputs = tokenizer(
            examples["maintext"],
            max_length=args.max_context_len,
            padding="max_length",
            truncation=True,
        )
        if mode == "test":
            return model_inputs
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["title"],
                max_length=args.max_answer_len,
                padding="max_length",
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return __implementation__


def postprocess_text(preds, labels):
    preds = [pred.strip() + "\n" for pred in preds]
    labels = [label.strip() + "\n" for label in labels]

    return preds, labels


def get_config(args):
    config = {
        "greedy": {"do_sample": False, "num_beams": 1},
        "beam": {"do_sample": False, "num_beams": args.num_beams},
        "top_k": {"top_k": args.top_k},
        "top_p": {"top_p": args.top_p},
        "temparature": {"temperature": args.temperature},
    }
    if args.strategy in STRAT:
        return config[args.strategy]
    return {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }