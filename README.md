# SEMDR: A Semantic-Aware Dual Encoder Model for Legal Judgment Prediction with Legal Clue Tracing

This repository contains the official implementation for the paper: **SEMDR: A Semantic-Aware Dual Encoder Model for Legal Judgment Prediction with Legal Clue Tracing**.

## Introduction

Legal Judgment Prediction (LJP) is a critical task in Legal AI, aiming to predict the judgment results (e.g., relevant law articles, charges, and terms of penalty) based on the fact descriptions of cases. Existing methods often struggle with confusing charges and long-tail legal cases due to the lack of fine-grained legal knowledge reasoning.

To address these challenges, we propose **SEMDR** (Semantic-Aware Dual Encoder Model for Legal Judgment Prediction). SEMDR introduces a novel **Legal Clue Tracing Mechanism**, which constructs a legal judgment reasoning knowledge graph and utilizes a Graph Attention Network (GAT) to perform multi-hop reasoning. By aligning case representations with legal label representations in a unified semantic space, SEMDR effectively captures fine-grained legal clues and achieves state-of-the-art performance on benchmark datasets.

## Core Components

- **`semdr_kg_gat.py`**: Implementation of the Legal Knowledge Graph Reasoner using Graph Attention Networks (GAT) for multi-fact reasoning and legal clue tracing.
- **`semdr_tt_ls.py`**: The Dual-Encoder (Two-Tower) backbone architecture for encoding case facts and legal labels.
- **`semdr_kg_gat_v1_sailer_low.py`**: The integrated SEMDR model training and evaluation script.

## Requirements

The code is implemented in Python and PyTorch. Main dependencies include:
- Python 3.8+
- PyTorch 1.10+
- Transformers (Hugging Face)
- scikit-learn
- tqdm

## Usage

### Training the SEMDR Model

To train the SEMDR model on the LJP dataset:

```bash
python3 semdr_kg_gat_v1_sailer_low.py \
    --data-dir ./data \
    --epochs 16 \
    --batch-size 32 \
    --learning-rate 2e-5
```

### Knowledge Graph Reasoning

The Graph Attention Network (GAT) module can be initialized and integrated as follows:

```python
from semdr_kg_gat import LegalKGReasoner

# Initialize the reasoner with 2-hop reasoning and 4 attention heads
reasoner = LegalKGReasoner(dim=256, gat_layers=2, num_heads=4)
```

## Citation

If you find our code or paper useful, please consider citing our work.

*(Citation information will be updated upon publication)*

## License

This project is licensed under the MIT License.
