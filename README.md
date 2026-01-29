# Basque Text Classification

Cross-lingual transfer learning strategies for text classification in Basque, a low-resource language. This project explores multiple approaches to classify news headlines when labeled training data in the target language is limited or unavailable.

## Project Overview

This project investigates three main tasks for Basque document classification:

| Task | Approach | F1-Score |
|------|----------|----------|
| **Z1** | Direct fine-tuning with BERTeus on Basque data | **74.76** |
| **Z2** | Cross-lingual transfer (3 strategies) | 25.40 - 36.24 |
| **Z3** | Few-shot prompting with Basque GPT-2 | 9.36 |

### State-of-the-Art Comparison

The Z1 result (F1 = 74.76) is competitive with published benchmarks on BasqueGLUE:

| Model | F1-Score |
|-------|----------|
| RoBERTa-large (EusCrawl) | 77.6 ± 0.5 |
| RoBERTa-base (CC100) | 76.2 ± 0.4 |
| **Z1 - BERTeus (this work)** | **74.76** |
| RoBERTa-base (Wikipedia) | 70.0 ± 0.8 |

## Datasets

**BasqueGLUE BHTC** (Basque Headline Topic Classification):
- 12,296 news headlines in Basque
- 12 categories: Ekonomia, Euskal Herria, Euskara, Gizartea, Historia, Ingurumena, Iritzia, Komunikazioa, Kultura, Nazioartea, Politika, Zientzia

**BBC News**:
- English news articles (2004-2005)
- 5 categories: business, entertainment, politics, sport, tech

## Methods

### Z1: Direct Basque Classification

Fine-tune BERTeus (`ixa-ehu/berteus-base-cased`) directly on Basque training data. This serves as the upper bound when native training data is available.

### Z2: Cross-Lingual Transfer Strategies

Three strategies for when Basque training data is unavailable:

| Strategy | Training Data | Model | Test Data | F1 (Map) | F1 (FT) |
|----------|--------------|-------|-----------|----------|---------|
| **S1** | BBC News (EN) | bert-base-uncased | BasqueGLUE → EN | 33.81 | 28.96 |
| **S2** | BBC News → EU | BERTeus | BasqueGLUE (EU) | **36.24** | 32.06 |
| **S3** | BBC News (EN) | mBERT | BasqueGLUE (EU) | 31.61 | 25.40 |

**Label Mapping** (12 → 5 categories):
```
Ekonomia           → Business
Gizartea, Iritzia, 
Politika, Euskal Herria → Politics
Kultura, Euskara, 
Historia, Komunikazioa  → Entertainment
Ingurumena, Zientzia    → Tech
Nazioartea             → Sports
```

### Z3: Few-Shot Prompting

Use Basque GPT-2 (`ClassCat/gpt2-small-basque-v2`) with few-shot prompting. The model predicts labels based on next-token probability, resulting in poor performance (F1 = 9.36).

## Repository Structure

```
basque-text-classification/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── z1_berteus_baseline.ipynb      # Direct Basque classification
│   ├── z2_strategy1_translate_test.ipynb
│   ├── z2_strategy2_translate_train.ipynb
│   ├── z2_strategy3_multilingual.ipynb
│   └── z3_gpt2_prompting.ipynb
├── src/
│   ├── data_utils.py                  # Data loading and preprocessing
│   ├── translation.py                 # Google Translate utilities
│   └── label_mapping.py               # Category mapping functions
└── results/
    └── f1_scores_summary.csv
```

## Models Used

| Model | Description | Parameters |
|-------|-------------|------------|
| `ixa-ehu/berteus-base-cased` | BERT pretrained on Basque corpora | 110M |
| `bert-base-uncased` | English BERT | 110M |
| `bert-base-multilingual-cased` | mBERT (104 languages) | 110M |
| `ClassCat/gpt2-small-basque-v2` | GPT-2 for Basque | 124M |

## Key Findings

1. **Native language models excel**: BERTeus achieves near state-of-the-art performance when Basque training data is available.

2. **Translation quality matters**: Strategy 2 (translate training data to Basque) outperforms Strategy 1 (translate test data to English), suggesting that evaluating on native text is preferable.

3. **mBERT limitations**: Despite including Basque in pretraining, mBERT's cross-lingual transfer without fine-tuning is limited for low-resource languages.

4. **Prompting challenges**: Few-shot prompting with GPT-2 is ineffective for classification tasks, especially with small models not specifically tuned for instruction-following.

5. **Fine-tuning paradox**: Surprisingly, label mapping outperformed fine-tuning in Z2 strategies, likely due to the small size of the fine-tuning dataset and frozen base layers.

## Requirements

```
transformers>=4.30.0
datasets>=2.13.0
torch>=2.0.0
scikit-learn>=1.0.0
googletrans==3.1.0a0
pandas>=1.5.0
sentencepiece>=0.1.99
```

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load fine-tuned BERTeus model
model_name = "ixa-ehu/berteus-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    "./trained_models/berteus_basque", 
    num_labels=12
)

# Classify Basque text
text = "Ekonomia globalak hazkunde egonkorra erakusten du"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()
```

## References

- [BasqueGLUE Benchmark](https://github.com/orai-nlp/BasqueGLUE)
- [BERTeus: BERT for Basque](https://huggingface.co/ixa-ehu/berteus-base-cased)
- [BBC News Dataset](https://huggingface.co/datasets/SetFit/bbc-news)

## Course Information

Developed for **Natural Language Processing** course at the University of the Basque Country (EHU).