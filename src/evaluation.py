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
from src.inference import load_model_and_tokenizer, build_chat_text, generate_batch, inference_api
from src.load_data import split_response


def load_data(file_path):

    df = pd.read_csv(file_path)

    # Use regex split on the column
    df = split_response(df, file_path)

    template_implicit = "Given the following Korean segment: '{response}'\n\nIdentify the honorific speech style in the provided Korean utterance by ONLY looking at the verb endings. The possible choices are: Casual (해), Polite (해요), Deferential (합니다/하십시오) or Others (If nothing fits). Be very short and give your final answer at the end in the format 'Honorific: <Casual/Polite/Deferential/Others>'"
    template_explicit = "Given the following Korean segment: '{response}'\n\nExtract first the Korean utterance by removing the Korean prefix of 'I was talking to {addresse}, and I said:'. Then, identify the honorific speech style in the provided Korean utterance by ONLY looking at the verb endings. The possible choices are: Casual (해), Polite (해요), Deferential (합니다/하십시오) or Others (If nothing fits). Be very short and give your final answer at the end in the format 'Honorific: <Casual/Polite/Deferential/Others>'"

    template_implicit = "Given the following Korean sentence: '{response}'\n\nIdentify the honorific speech style in the provided Korean utterance. The possible choices are: Casual (해), Polite (해요), Deferential (합니다/하십시오). Explain briefly and give your final answer in the format 'Honorific: <Casual/Polite/Deferential>."
    template_explicit = "Given the following Korean sentence: '{response}'\n\nIdentify the honorific speech style in the provided Korean utterance (ignore the prefix). The possible choices are: Casual (해), Polite (해요), Deferential (합니다/하십시오). Explain briefly and give your final answer in the format 'Honorific: <Casual/Polite/Deferential>."


    df["raw_prompts_eval"] = df.apply(
        lambda row: (
            template_explicit.format(response=row["extracted_response"], addresse=row["addresse"])
            if row["type"] == "Ex"
            else template_implicit.format(response=row["extracted_response"])
        ),
        axis=1
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="/lustre/project/ki-topml/minbui/projects/models/Llama-3.3-70B-Instruct",
                        help="HuggingFace model name")
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
    parser.add_argument("--top_k", type=int, default=20, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the model and tokenizer
    if args.model_name not in ["GPT_OSS_120B", "Gemma3_27B", "Qwen3_235B"]:
        model, tokenizer = load_model_and_tokenizer(args.model_name)

    for input_file in args.input_files:
        print(input_file)
        # Load the Data
        df = load_data(input_file)
        prompts = df["raw_prompts_eval"].tolist()

        # Inference Time!
        if args.model_name in ["GPT_OSS_120B", "Gemma3_27B", "Qwen3_235B"]:
            responses = inference_api(
                prompts,
                args,
            )
        else:
            responses = generate_batch(
                model,
                tokenizer,
                prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample
            )

        df["eval_response_eval" + "_" + args.model_name.split("/")[-1]] = responses
        df.to_csv(input_file, index=False)


if __name__ == "__main__":
    main()
