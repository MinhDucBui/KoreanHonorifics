#!/usr/bin/env python3
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.load_data import load_data
from googletrans import Translator
import time


def google_api_translation(prompts):
    translations = []
    translator = Translator()
    translations = translator.translate(prompts, src="ko", dest="en")
    for translation in translations:
        translations.append(translation)
        print(translation)
        time.sleep(1)  # avoid hitting rate limits

    return translations

def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output_folder", type=str, default="output/",
                        help="File containing prompts for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the Data
    df = load_data()
    df["raw_prompts"] = (
        df["raw_prompts"]
        .str.split("Translate the following English sentence into Korean: ")
        .str[-1]
        .str.split("\n\nProvide only")
        .str[0]
    )
    prompts = df["raw_prompts"].tolist()

    # Inference
    responses = google_api_translation(prompts)

    df["response"] = responses
    df.to_csv(f"{args.output_folder}/google_translation.csv", index=False)


if __name__ == "__main__":
    main()
