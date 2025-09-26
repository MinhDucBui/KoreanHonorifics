#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from googletrans import Translator
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.load_data import load_data
from src.api_keys import API_KEY

SYSTEM_PROMPT = "You are a helpful assistant."


def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # let HF handle device placement
        torch_dtype=torch.float16,   # use FP16 for efficiency
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_chat_text(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except (ValueError, AttributeError):  # no template available
        return user_prompt




@torch.inference_mode()
def generate_batch_nllb(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    **gen_kwargs: Any
) -> List[str]:

    outputs: List[str] = []
    device = model.device
    chat_texts = prompts
    num_batches = ceil(len(chat_texts) / batch_size)
    for bi in tqdm(range(num_batches)):
        chunk = chat_texts[bi * batch_size: (bi + 1) * batch_size]
        model_inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_token_type_ids=False
        ).to(device)
        gen = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        generated_texts = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        outputs += generated_texts
    return outputs


@torch.inference_mode()
def generate_batch_api(
    model,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    **gen_kwargs: Any
) -> List[str]:

    client = OpenAI(base_url="https://ki-chat.uni-mainz.de/api/", api_key=API_KEY)
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


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 20,
    do_sample: bool = False,
    repetition_penalty: float = 1.05,
    **gen_kwargs: Any
) -> List[str]:

    outputs: List[str] = []
    device = model.device
    chat_texts = [build_chat_text(tokenizer, p) for p in prompts]
    num_batches = ceil(len(chat_texts) / batch_size)
    for bi in tqdm(range(num_batches)):
        chunk = chat_texts[bi * batch_size: (bi + 1) * batch_size]
        model_inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_token_type_ids=False
        ).to(device)
        gen = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        generated_texts = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        outputs += generated_texts
    return outputs

def inference_general_llms(prompts, args):
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Inference Time!
    responses = generate_batch(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample
    )
    return responses


def inference_api(prompts, args):
    responses = generate_batch_api(
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
    return responses

def inference_nllb(prompts, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",          # let HF handle device placement
        torch_dtype=torch.float16,   # use FP16 for efficiency
        trust_remote_code=True)

    responses = generate_batch(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample
    )
    return responses


def inference_google_translation(prompts, args):
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
    parser.add_argument("--mode", type=str, default="",
                        help="File containing prompts for generation")
    parser.add_argument("--file_path", type=str, default="",
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
    df = load_data(args.model_name, args.mode, args.file_path)
    prompts = df[f"raw_prompts{('_' + args.mode) if args.mode else ''}"].tolist()

    # Load the model and tokenizer
    if "nllb" in args.model_name:
        responses = inference_nllb(prompts, args)
    elif "google_translation" in args.model_name:
        responses = inference_google_translation(prompts, args)
    elif args.model_name in ["GPT_OSS_120B", "Gemma3_27B", "Qwen3_235B"]:
        responses = inference_api(prompts, args)
    else:
        responses = inference_general_llms(prompts, args)

    df[f"response{('_' + args.mode) if args.mode else ''}"] = responses
    if args.file_path != "":
        df.to_csv(args.file_path, index=False)
    else:
        df.to_csv(f"{args.output_folder}/{args.model_name.split('/')[-1]}.csv", index=False)


if __name__ == "__main__":
    main()
