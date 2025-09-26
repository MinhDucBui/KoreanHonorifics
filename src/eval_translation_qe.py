from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.evaluation import split_response
from src.load_data import create_source_sentence


SYSTEM_PROMPT = "You are a helpful assistant."


def build_chat_text(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    return messages


def load_data(file_path):
    df = pd.read_csv(file_path)
    template = (
    "Score the following translation from English to Korean on a continuous scale from 0 to 100. Only evaluate the semantic of the sentence. "
    "A score of 0 means \"no meaning preserved,\" and a score of 100 means \"perfect meaning and grammar.\"\n"
    "Keep yourself short and end your answer with: Score: <final score>\n\n"
    "English source: \"{source_sentence}\"\n"
    "Korean translation: \"{extracted_response}\""
    )

    df = split_response(df, file_path)

    df = create_source_sentence(df)
    df["raw_prompts"] = df.apply(lambda row: template.format(**row), axis=1)
    return df


@torch.inference_mode()
def generate_batch(
    model,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    **gen_kwargs: Any
) -> List[str]:

    client = OpenAI(base_url="https://ki-chat.uni-mainz.de/api/", api_key="sk-c8ee5a5aecf64650b1c484452e8f266a")
    model_name = model.replace("_", " ")

    batch_size = 1
    outputs: List[str] = []
    chat_texts = [[{"role": "user", "content": p}] for p in prompts]
    num_batches = ceil(len(chat_texts) / batch_size)
    for bi in tqdm(range(num_batches)):
        chunk = chat_texts[bi * batch_size: (bi + 1) * batch_size]

        completion = client.chat.completions.create(
            # Required
            model=model_name,

            # Core inputs
            messages=chunk[0],

            # Generation controls
            max_tokens=max_new_tokens,             # Maximum tokens to generate
            temperature=temperature,             # Randomness (0 = deterministic)
            top_p=top_p,                   # Nucleus sampling (alternative to temperature)
            #presence_penalty=0.0,        # Penalize new tokens if they appear already
            #frequency_penalty=0.0,       # Penalize frequent tokens
            seed=None,                   # Random seed (for reproducibility)
        )

        outputs.append(completion.choices[0].message.content)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="Gemma3_27B",
                        help="File containing prompts for generation")
    parser.add_argument("--file_path", type=str, default="output/aya-expanse-32b.csv",
                        help="File containing prompts for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the Data
    df = load_data(args.file_path)
    prompts = df["raw_prompts"].tolist()

    # Inference Time!
    responses = generate_batch(
        args.model_name,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample
    )

    df["response_eval_trans"] = responses
    df.to_csv(args.file_path, index=False)


if __name__ == "__main__":
    main()
