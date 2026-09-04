# CAIL Dataset Resources

This directory contains the CAIL dataset resources used by the SEMDR experiments.

## Layout

```text
data/
├── cail_small/
│   ├── train_cs.json
│   ├── valid_cs.json
│   ├── test_cs.json
│   └── mappings_cail_small.pkl
└── cail_big/
    ├── train_cs.json
    ├── test_cs.json
    └── mappings_cail_big_fixed.pkl
```

Each `*.json` file is stored as JSON Lines, with one case record per line.

## Dataset Splits

| Dataset | Split | Records |
|---|---:|---:|
| CAIL-small | train | 101,619 |
| CAIL-small | valid | 13,768 |
| CAIL-small | test | 26,749 |
| CAIL-big | train | 1,587,979 |
| CAIL-big | test | 185,120 |

## Record Fields

| Field | CAIL-small | CAIL-big | Description |
|---|---:|---:|---|
| `fact_cut` | Yes | Yes | Tokenized case-fact text with whitespace separators. This is the shared text field used for cross-dataset length statistics. |
| `fact` | Yes | No | Original, non-tokenized case-fact text. |
| `accu` | Yes | Yes | Charge label identifier. |
| `law` | Yes | Yes | Law-article label identifier. |
| `time` | Yes | Yes | Term-of-penalty related label identifier. |
| `term_cate` | Yes | Yes | Term-of-penalty category identifier. |
| `term` | Yes | Yes | Term-of-penalty label identifier. |

The mapping files provide the label-space resources required to decode task identifiers.

## Length Statistics

The repository script `scripts/analyze_cail_lengths.py` streams JSON Lines data and reports both character lengths and BERT input token lengths. The tokenizer used for the reported results is the local `bert-base-chinese` tokenizer. Token counts include `[CLS]` and `[SEP]` and are measured before truncation.

The token-length bins are:

| Bin | Definition |
|---|---|
| `token_0_256` | 0 to 256 tokens, inclusive |
| `token_257_512` | 257 to 512 tokens, inclusive |
| `token_gt_512` | More than 512 tokens |

Run the analysis with:

```bash
python scripts/analyze_cail_lengths.py \
  --base-dir ./data \
  --datasets cail_small cail_big \
  --text-fields fact_cut fact \
  --tokenizer /path/to/bert-base-chinese \
  --output-dir ./results
```

The CAIL-big files do not include `fact`; the script therefore reports `fact_cut` for CAIL-big and both available text fields for CAIL-small.
