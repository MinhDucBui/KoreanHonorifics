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
    df = load_data(args.model_name, args.mode, args.file_path)
    prompts = df[f"raw_prompts{('_' + args.mode) if args.mode else ''}"].tolist()

    # Inference
    responses = google_api_translation(prompts)

    df[f"response{('_' + args.mode) if args.mode else ''}"] = responses
    if args.file_path != "":
        df.to_csv(args.file_path, index=False)
    else:
        df.to_csv(f"{args.output_folder}/{args.model_name.split('/')[-1]}.csv", index=False)


if __name__ == "__main__":
    main()
