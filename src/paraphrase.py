#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
import sys, os
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.inference import load_model_and_tokenizer, build_chat_text, generate_batch
from src.load_data import split_response


def load_data(file_path):

    df = pd.read_csv(file_path)

    # Use regex split on the column
    df = split_response(df, file_path)

    template_implicit = "Given the following Korean segment: '{response}'\n\nFirst, check whether the correct honorific speech style is used in the provided Korean utterance. If correct, output the original Korean segment. If not, adjust the honorific style and output the correct segment. Use the pattern: 'Final: <SEGMENT>'"
    template_explicit = "Given the following Korean segment: '{response}'\n\nFirst, check whether the correct honorific speech style is used in the provided Korean utterance (ignore the prefix). If correct, output the original Korean segment. If not, adjust the honorific style and output the correct segment. Use the pattern: 'Final: <SEGMENT>'"

    df["raw_prompts_paraphrase"] = df.apply(
        lambda row: (
            template_explicit.format(response=row["extracted_response"])
            if row["type"] == "Ex"
            else template_implicit.format(response=row["extracted_response"])
        ),
        axis=1
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="/p/project/westai0073/Models/gemma-2-9b-it")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",   # 👈 allows multiple files
        default=["output/Qwen2.5-7B-Instruct.csv"],
        help="File(s) containing prompts for generation"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    for input_file in args.input_files:
        print(input_file)
        # Load the Data
        df = load_data(input_file)
        prompts = df["raw_prompts_paraphrase"].tolist()

        # Inference Time!
        responses = generate_batch(
            model,
            tokenizer,
            prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )

        df["response"] = responses

        df["response"] = df["response"].str.split("Final: ").str[-1]

        base_dir, base_name = os.path.split(input_file)
        output_file = os.path.join(base_dir, "paraphrase_" + base_name)

        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
