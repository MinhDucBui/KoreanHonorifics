<h1 align="center">Do LLMs Respect Korean Honorifics?</h1>
<h3 align="center"><em>Evaluating Speech Level Awareness in English-to-Korean Machine Translation</em></h3>



This repository contains code, data, and evaluation scripts for our paper:

[![Paper Link](https://img.shields.io/badge/Paper-coming_soon-blue)]()



> **Abstract:** *Honorifics encode social hierarchies and relational nuances, making their correct use a culturally sensitive and challenging aspect of translation. In doing so, they reflect and shape how individuals position themselves and others within a social world. In this work, we investigate how different models handle Korean honorific translation, 
both in implicit scenarios, where only the sentence is given, and explicit scenarios. Our findings are as follows: (i) large language models finetuned for translation (MTLMs) consistently prefer polite forms more than their instruction-tuned counterparts in both scenarios, (ii) sequence-to-sequence models produce less polite outputs in implicit contexts but shift toward more polite forms when the addressee is explicitly provided; and (iii) both types of LM-based models tend to become more casual when the addressee is known.
When compared with human preferences, MTLMs diverge more strongly, exhibiting a systematic overuse of polite forms relative to human judgments.*

---

## 📊 Dataset: Korean Honorifics Translation Benchmark

### Overview

The **Korean Honorifics Dataset** consists of 1,500 English sentences paired with addressee context — each annotated by multiple human annotators for the appropriate Korean honorific speech style. The dataset covers both **explicit** (addressee stated in context) and **implicit** (no addressee context) conditions, across 30 real-world social scenarios.

### Honorific Speech Levels

| Speech Level | Korean Style | Example Context |
|---|---|---|
| Casual | 해 | Close friends, younger siblings |
| Polite | 해요 | Strangers, acquaintances |
| Deferential | 합니다 / 하십시오 | Professor, boss, formal settings |

### Examples

| Addressee | Condition | Expected Style |
|---|---|---|
| professor at university | Explicit | Deferential (합니다) |
| close friend | Explicit | Casual (해) |
| stranger on the street | Explicit | Polite (해요) |
| *(no context)* | Implicit | Annotator majority vote |

### Access to Dataset

The dataset is distributed under **CC BY-NC-ND 4.0**.

---

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/MinhDucBui/KoreanHonorifics
cd KoreanHonorifics

# Install dependencies
pip install -r requirements.txt
```

⚠️ **IMPORTANT**

- Insert correct model paths in the scripts before running local models.

---

## 🧪 Experiments

### Experiment 1: Honorific-Aware Translation

**Task:** Translate English sentences into Korean with the correct honorific speech level given addressee context.

**Run inference:**
```bash
python src/inference.py \
    --model_name "/path/to/model" \
    --batch_size 128 \
    --max_new_tokens 128
```


---

### Experiment 2: Back-Translation Evaluation

**Task:** Translate the model's Korean output back into English to verify that the original meaning was preserved.

**Run back-translation:**
```bash
python src/inference.py \
    --model_name "google_translation" \
    --batch_size 128 \
    --mode "backtrans" \
    --file_path "output/<model_name>.csv"
```

---

## 📝 Evaluation: LLM-as-a-Judge

Automatically evaluate whether the translated Korean output uses the correct honorific speech style via LLM-as-a-Judge.

```bash
python src/evaluation.py \
    --input_files output/model1.csv output/model2.csv \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --batch_size 64 \
    --max_new_tokens 8
```

---

## 🔧 Models Evaluated

| Model | Type | Size |
|---|---|---|
| Llama-3.3-70B-Instruct | General LLM | 70B |
| Qwen2.5-72B-Instruct | General LLM | 72B |
| aya-expanse-32b | General LLM | 32B |
| GemmaX2-28-9B-v0.1 | General LLM | 9B |
| gemma-2-9b-it | General LLM | 9B |
| Tower-Plus-72B | Translation LLM | 72B |
| Tower-Plus-9B | Translation LLM | 9B |
| Hunyuan-7B-Instruct | General LLM | 7B |
| Hunyuan-MT-7B | Specialized MT | 7B |
| Seed-X-PPO-7B | Specialized MT | 7B |
| NLLB-200-3.3B | Specialized MT | 3.3B |
| NLLB-200-distilled-600M | Specialized MT | 600M |
| Google Translate | API Baseline | — |

---

## 📄 Citation

If you use this dataset or code, please cite:

```text
tba
```

---

## 📧 Contact

Minh Duc Bui — [minhducbui@uni-mainz.de](mailto:minhducbui@uni-mainz.de)
