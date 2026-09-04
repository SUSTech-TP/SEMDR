#!/usr/bin/env python3
"""Compute text-length statistics for CAIL datasets using a local BERT tokenizer.

The script reads JSON Lines files in a streaming manner and writes one summary row
per dataset, split, and text field. Token counts include BERT special tokens
([CLS] and [SEP]) because they reflect actual model input length before truncation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_BINS = (
    ("token_0_256", 0, 256),
    ("token_257_512", 257, 512),
    ("token_gt_512", 513, None),
)


@dataclass
class HistogramStats:
    """Exact statistics based on a compact value-frequency histogram."""

    count: int = 0
    total: int = 0
    minimum: int | None = None
    maximum: int | None = None
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.histogram[value] += 1

    def quantile(self, q: float) -> float | None:
        if not self.count:
            return None
        rank = max(1, math.ceil(q * self.count))
        cumulative = 0
        for value in sorted(self.histogram):
            cumulative += self.histogram[value]
            if cumulative >= rank:
                return float(value)
        return float(self.maximum)  # Defensive fallback.

    def as_dict(self, prefix: str) -> dict[str, int | float | None]:
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": round(self.total / self.count, 4) if self.count else None,
            f"{prefix}_min": self.minimum,
            f"{prefix}_p50": self.quantile(0.50),
            f"{prefix}_p90": self.quantile(0.90),
            f"{prefix}_p95": self.quantile(0.95),
            f"{prefix}_p99": self.quantile(0.99),
            f"{prefix}_max": self.maximum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute character and BERT-token length statistics for CAIL JSONL datasets."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/cwadmin/Tompanda/LegalDuet/ljp_labels"),
        help="Directory containing cail_small and cail_big.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cail_small", "cail_big"],
        help="Dataset directory names below --base-dir.",
    )
    parser.add_argument(
        "--text-fields",
        nargs="+",
        default=["fact_cut", "fact"],
        help="Text fields to analyze when present. fact_cut is shared by both datasets.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/home/cwadmin/Tompanda/pretrained_models/google_bert_base_chinese",
        help="Local Hugging Face tokenizer path or model identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("statistics"),
        help="Directory for CSV and JSON summary files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of records tokenized per batch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Write progress to stderr every N input lines.",
    )
    return parser.parse_args()


def split_files(dataset_dir: Path) -> Iterable[tuple[str, Path]]:
    for split in ("train", "valid", "test"):
        path = dataset_dir / f"{split}_cs.json"
        if path.is_file():
            yield split, path


def tokenize_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    if not texts:
        return []
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def analyze_file(
    tokenizer: Any,
    dataset: str,
    split: str,
    input_path: Path,
    field_name: str,
    batch_size: int,
    progress_every: int,
) -> dict[str, Any] | None:
    char_raw = HistogramStats()
    char_no_space = HistogramStats()
    token_stats = HistogramStats()
    token_bin_counts = {name: 0 for name, _, _ in TOKEN_BINS}
    missing_field = 0
    invalid_json = 0
    non_string_field = 0
    input_lines = 0
    texts: list[str] = []

    def consume_batch() -> None:
        nonlocal texts
        token_lengths = tokenize_lengths(tokenizer, texts)
        for token_length in token_lengths:
            token_stats.add(token_length)
            for name, lower, upper in TOKEN_BINS:
                if token_length >= lower and (upper is None or token_length <= upper):
                    token_bin_counts[name] += 1
                    break
        texts = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            input_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if field_name not in record:
                missing_field += 1
                continue
            text = record[field_name]
            if not isinstance(text, str):
                non_string_field += 1
                continue

            char_raw.add(len(text))
            char_no_space.add(len(WHITESPACE_RE.sub("", text)))
            texts.append(text)
            if len(texts) >= batch_size:
                consume_batch()
            if progress_every and input_lines % progress_every == 0:
                print(
                    f"[{dataset}/{split}/{field_name}] processed {input_lines:,} lines",
                    file=sys.stderr,
                    flush=True,
                )
    consume_batch()

    if not token_stats.count:
        return None

    result: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "field": field_name,
        "source_file": str(input_path),
        "input_lines": input_lines,
        "missing_field_lines": missing_field,
        "non_string_field_lines": non_string_field,
        "invalid_json_lines": invalid_json,
        **char_raw.as_dict("char_raw"),
        **char_no_space.as_dict("char_no_whitespace"),
        **token_stats.as_dict("bert_input_tokens"),
    }
    for name, _, _ in TOKEN_BINS:
        count = token_bin_counts[name]
        result[f"{name}_count"] = count
        result[f"{name}_pct"] = round(100.0 * count / token_stats.count, 4)
    return result


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cail_text_length_statistics.csv"
    json_path = args.output_dir / "cail_text_length_statistics.json"

    if not results:
        raise RuntimeError("No analyzable text fields were found.")

    fieldnames = list(results[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "settings": {
            "base_dir": str(args.base_dir),
            "datasets": args.datasets,
            "text_fields_requested": args.text_fields,
            "tokenizer": args.tokenizer,
            "token_count_definition": "BERT input_ids length with [CLS] and [SEP], without truncation.",
            "token_bins": {
                "0_256": "0 <= token length <= 256",
                "257_512": "257 <= token length <= 512",
                "gt_512": "token length > 512",
            },
        },
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=Path(args.tokenizer).exists(),
        use_fast=True,
    )
    tokenizer.model_max_length = int(1e12)

    results: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_dir = args.base_dir / dataset
        if not dataset_dir.is_dir():
            print(f"Skipping missing dataset directory: {dataset_dir}", file=sys.stderr)
            continue
        for split, input_path in split_files(dataset_dir):
            for field_name in args.text_fields:
                summary = analyze_file(
                    tokenizer=tokenizer,
                    dataset=dataset,
                    split=split,
                    input_path=input_path,
                    field_name=field_name,
                    batch_size=args.batch_size,
                    progress_every=args.progress_every,
                )
                if summary is not None:
                    results.append(summary)
                else:
                    print(
                        f"Skipping unavailable or empty field: {dataset}/{split}/{field_name}",
                        file=sys.stderr,
                    )

    csv_path, json_path = write_outputs(results, args)
    print(f"Wrote {len(results)} summary rows to {csv_path}")
    print(f"Wrote detailed summary to {json_path}")


if __name__ == "__main__":
    main()
