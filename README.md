# English–Hindi Machine Translation using Transformer (From Scratch)

This project implements an **English → Hindi Neural Machine Translation system** using a **Transformer architecture built from scratch in PyTorch**.  
It includes **custom BPE tokenization**, **training**, **evaluation using BLEU**, and **inference** in a clean, modular ML project structure.

---

## Features

- Transformer implemented **from scratch**
- Custom **BPE tokenizer** for English & Hindi
- Encoder–Decoder architecture
- Self-attention & Cross-attention
- Positional Encoding
- Training with CrossEntropy Loss
- BLEU score evaluation
- Greedy decoding inference
- Config-driven pipeline (YAML)
- Logging support

---

## Project Structure

```
project-root/
├── .github/
│ └── workflows/
├── artifacts/
│ ├── data_ingestion/
│ │ └── translation/
│ └── tokenization_trainer/
│ └── tokenizer/
│ ├── eng.json
│ └── hin.json
├── config/
│ ├── config.yaml
│ └── params.yaml
├── logs/
├── Research/
│ └── experiments.ipynb
├── src/
│ ├── components/
│ │ ├── attention/
│ │ ├── layers/
│ │ ├── models/
│ │ ├── network/
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ ├── training_pipeline.py
│ │ ├── evaluation_pipeline.py
│ │ └── prediction.py
│ ├── entity/
│ ├── utils/
│ ├── constants/
│ ├── config/
│ └── logger/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```


---

## Installation

### Create Virtual Environment
```bash
pip install transformer_rs
```
or
```bash
pip install git+https://github.com/Ronak-Sah/Transformer_rs.git
```
---

### Model Architecture

- Token Embedding
- Positional Encoding
- Encoder Stack
- Multi-Head Self Attention
- Feed Forward Network
- Decoder Stack
- Masked Self Attention
- Cross Attention
- Linear + Softmax Output