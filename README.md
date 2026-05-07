# Sequential vs. Joint Bilingual Language Modeling: A BabyBabelLM Study

**Reihaneh Iranmanesh, Saeedeh Davoudi, Lin Ai, Kohei Kajikawa** · Georgetown University

---

## Overview

This project compares three bilingual training regimes for small language models using the [BabyBabelLM](https://huggingface.co/BabyLM-community) framework. All variants share the GoldFish GPT-2-small architecture (~17.1M parameters) and differ only in starting checkpoint, tokenizer, and whether fine-tuning is performed.

| Variant | Description |
|---------|-------------|
| **A — eng→fas** | `eng-baseline-small` fine-tuned on Persian data with the Persian tokenizer |
| **B — fas→eng** | `fas-baseline-small` fine-tuned on English data with the English tokenizer |
| **C — joint** | `eng-fas-baseline-small` used off the shelf, no fine-tuning |

---

## Results

Evaluated on the BabyLM held-out test sets. BLiMP chance = 50%. PPL is not directly comparable across variants due to different tokenizers.

| Variant | Language | PPL ↓ | BERTScore ↑ | ROUGE-L ↑ | BLiMP ↑ |
|---------|----------|-------|-------------|-----------|---------|
| A: eng→fas | English | 46.55 | 0.748 | 2.19% | 66.2% |
| A: eng→fas | Persian | 99.05 | 0.401 | 8.65% | 73.9% |
| B: fas→eng | English | **20.75** | **0.809** | **8.96%** | 75.8% |
| B: fas→eng | Persian | 11.81 | 0.166 | 0.13% | 61.4% |
| C: joint | English | 22.34 | 0.807 | 8.46% | **83.4%** |
| C: joint | Persian | 54.75 | **0.407** | 7.70% | **77.9%** |

Key findings:
- Full fine-tuning causes near-total **catastrophic forgetting** of the source language in both sequential variants.
- **Direction matters**: Persian→English (B) outperforms English→Persian (A) on the target language.
- **Joint pretraining** (C) is the only variant that retains reasonable performance in both languages simultaneously.

---

## Checkpoints & Results

Trained model checkpoints and full evaluation results are available on Google Drive:

🔗 **[https://drive.google.com/drive/folders/1FjcHqkNleLkyptNIlbOY75sArSxC3XDC?usp=sharing](https://drive.google.com/drive/folders/1FjcHqkNleLkyptNIlbOY75sArSxC3XDC?usp=sharing)**

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `BabyLM_Persian_Training.ipynb` | Variant A — fine-tunes `eng-baseline-small` on Persian data using the Persian tokenizer |
| `BabyLM_English_Training.ipynb` | Variant B — fine-tunes `fas-baseline-small` on English data using the English tokenizer |

Both notebooks are designed to run on **Google Colab** with Drive mounting for checkpoint saving. They share identical hyperparameters: AdamW optimizer, lr=1e-4, linear schedule with 10% warmup, batch size 64, 5 epochs, seed=42.

---

## Setup

```bash
pip install transformers datasets accelerate huggingface-hub tensorboard torch
```

A HuggingFace account is required to download the BabyBabelLM base checkpoints. Run `notebook_login()` inside the notebook or set your `HF_TOKEN` environment variable before running.

---

## Pretrained Baselines (HuggingFace Hub)

All base models are from [`BabyLM-community`](https://huggingface.co/BabyLM-community):

- [`BabyLM-community/eng-baseline-small`](https://huggingface.co/BabyLM-community/eng-baseline-small)
- [`BabyLM-community/fas-baseline-small`](https://huggingface.co/BabyLM-community/fas-baseline-small)
- [`BabyLM-community/eng-fas-baseline-small`](https://huggingface.co/BabyLM-community/eng-fas-baseline-small)

Training data:
- [`BabyLM-community/babylm-fas`](https://huggingface.co/datasets/BabyLM-community/babylm-fas) (Persian)
- [`BabyLM-community/babylm-eng`](https://huggingface.co/datasets/BabyLM-community/babylm-eng) (English)
