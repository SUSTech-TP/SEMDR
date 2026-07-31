# -*- coding: utf-8 -*-
"""
semdr_kg_gat_v1_sailer_low.py  ——  SAILER 默认版（v3）
==========================================
本版本在历史修改基础上新增以下功能：
  1. 修复 JSONL 读取导致的 json.decoder.JSONDecodeError: Extra data 错误
  2. 默认使用 SAILER 预训练模型（--encoder-init 默认值为 "sailer"）
  3. 新增 SERVER_SAILER_MODEL_DIR / DEFAULT_SAILER_MODEL 路径变量
  4. 【v3 新增】--warmup-ckpt 参数：支持在训练开始前自动加载
     warmup_gat_transe_low_v3.py 生成的 GAT 预热权重，实现参数预热接入。

推荐运行示例（无预热）：
  python semdr_kg_gat_v1_sailer_low.py \\
      --data-dir /home/cwadmin/Tompanda/LegalDuet/ljp_labels/cail_small/semdr_low \\
      --output-dir ./checkpoints/semdr_kg_v1_no_warmup \\
      --fp16 --do-test --epochs 16 --batch-size 30

推荐运行示例（启用 TransE 预热）：
  # 第一步：运行预热脚本生成 GAT 权重
  python warmup_gat_transe_low_v3.py \\
      --data-dir /home/cwadmin/Tompanda/LegalDuet/ljp_labels/cail_small/semdr_low \\
      --sailer-model-dir /home/cwadmin/Tompanda/LegalDuet/sailer/ \\
      --output-dir ./checkpoints/gat_warmup_low \\
      --max-aggregations 5000 --max-epochs 200 --patience 20 \\
      --batch-size 1200 --lr 0.01 --dropout 0.5 --compare-baseline --record-attn

  # 第二步：将预热权重传入主训练
  python semdr_kg_gat_v1_sailer_low.py \\
      --data-dir /home/cwadmin/Tompanda/LegalDuet/ljp_labels/cail_small/semdr_low \\
      --output-dir ./checkpoints/semdr_kg_v1_with_warmup \\
      --warmup-ckpt ./checkpoints/gat_warmup_low/best_warmup_gat.pth \\
      --fp16 --do-test --epochs 16 --batch-size 30

若需切换回 BERT：
  python semdr_kg_gat_v1_sailer_low.py --encoder-init bert ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ============================================================================
# 1. 路径默认值
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SERVER_PROJECT_ROOT = Path("/home/cwadmin/Tompanda/LegalDuet")
SERVER_FINE_TUNING_DIR = SERVER_PROJECT_ROOT / "Fine-Tuning"
SERVER_DATA_DIR = SERVER_PROJECT_ROOT / "ljp_labels" / "cail_small"
SERVER_BERT_MODEL_DIR  = SERVER_PROJECT_ROOT / "bert"
SERVER_SAILER_MODEL_DIR = SERVER_PROJECT_ROOT / "sailer"
SERVER_OUTPUT_DIR      = SERVER_FINE_TUNING_DIR / "checkpoints" / "semdr_kg_v1"

DEFAULT_DATA_DIR    = os.environ.get("SEMDR_DATA_DIR",    str(SERVER_DATA_DIR))
DEFAULT_BERT_MODEL  = os.environ.get("SEMDR_BERT_MODEL",  str(SERVER_BERT_MODEL_DIR))
DEFAULT_SAILER_MODEL = os.environ.get("SEMDR_SAILER_MODEL", str(SERVER_SAILER_MODEL_DIR))
DEFAULT_OUTPUT_DIR  = os.environ.get("SEMDR_OUTPUT_DIR",  str(SERVER_OUTPUT_DIR))

DEFAULT_TRAIN_FILE = "train_cs.json"
DEFAULT_VALID_FILE = "valid_cs.json"
DEFAULT_TEST_FILE = "test_cs.json"

DEFAULT_MAPPING_PKL = SERVER_PROJECT_ROOT / "ljp_labels" / "cail_small" / "mappings_cail_small.pkl"
DEFAULT_LAW_CSV = "cail2018law2text.csv"
DEFAULT_ACCU_CSV = "cail2018charge2text.csv"
DEFAULT_TERM_CSV = "cail2018term2text.csv"
DEFAULT_LAW_LABEL_FILE = "new_law.txt"
DEFAULT_ACCU_LABEL_FILE = "new_accu.txt"
DEFAULT_TERM_LABEL_FILE = "new_term.txt"

MAPPING_PKL_CANDIDATES = (
    "mappings_cail_small.pkl",
    "mappings_low.pkl",
    "mappings_lowfreq_fixed.pkl",
)
TRAIN_FILE_CANDIDATES = (
    "train_cs.json", "train_cs.jsonl",
    "train_cs_bert_small.json", "train_processed_bert.pkl",
    "train.pkl", "train.json", "train.jsonl",
)
VALID_FILE_CANDIDATES = (
    "valid_cs.json", "valid_cs.jsonl",
    "valid_cs_bert_small.json", "valid_processed_bert.pkl",
    "valid.pkl", "valid.json", "valid.jsonl",
    "val.json", "val.jsonl",
)
TEST_FILE_CANDIDATES = (
    "test_cs.json", "test_cs.jsonl",
    "test_cs_bert_small.json", "test_processed_bert.pkl",
    "test.pkl", "test.json", "test.jsonl",
)
LAW_LABEL_CANDIDATES = ("new_law.txt", "law.txt", "law_label.txt", "law_labels.txt")
ACCU_LABEL_CANDIDATES = ("new_accu.txt", "accu.txt", "accu_label.txt", "accu_labels.txt", "charge.txt")
TERM_LABEL_CANDIDATES = ("new_term.txt", "term.txt", "term_label.txt", "term_labels.txt")

# ============================================================================
# 2. 数据读取与字段兼容（与 content_8 完全一致）
# ============================================================================

FACT_FIELD_CANDIDATES = ("fact", "fact_cut", "facts", "content", "text", "case_fact", "criminal_fact")
LAW_FIELD_CANDIDATES = ("law", "law_label", "article", "article_label", "law_label_lists")
ACCU_FIELD_CANDIDATES = ("accu", "accu_label", "charge", "charge_label", "accu_label_lists")
TERM_FIELD_CANDIDATES = ("term", "time", "penalty", "term_label", "time_label", "imprisonment", "term_lists")

FactInputFormat = Literal["auto", "bert_ids", "text", "pretokenized_words", "pretokenized_wordpieces"]
PoolingType = Literal["cls", "mean"]
SimilarityType = Literal["dot", "cosine"]


@dataclass
class FieldConfig:
    fact_field: Optional[str] = None
    law_field: Optional[str] = None
    accu_field: Optional[str] = None
    term_field: Optional[str] = None


def _first_existing(sample: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in sample:
            return key
    return None


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} 不是 JSON 数组。")
        return data
    # JSONL 格式：每行一个独立 JSON 对象（兼容 json.load 报 Extra data 的情况）
    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON 行：{exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no} 不是 JSON 对象。")
        records.append(obj)
    return records


def _load_pkl(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的 pickle 内容既不是 list 也不是 dict。")
    if "fact_list" not in data:
        raise ValueError(f"{path} 是 dict，但缺少 `fact_list` 键。")
    law_key = _first_existing(data, LAW_FIELD_CANDIDATES)
    accu_key = _first_existing(data, ACCU_FIELD_CANDIDATES)
    term_key = _first_existing(data, TERM_FIELD_CANDIDATES)
    if law_key is None or accu_key is None or term_key is None:
        raise ValueError(f"{path} 缺少 law/accu/term 标签列表键。")
    records: List[Dict[str, Any]] = []
    facts = data["fact_list"]
    for i in range(len(facts)):
        fact = facts[i]
        if hasattr(fact, "tolist"):
            fact = fact.tolist()
        records.append({
            "fact": fact,
            "law": int(data[law_key][i]),
            "accu": int(data[accu_key][i]),
            "term": int(data[term_key][i]),
        })
    return records


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return _load_pkl(path)
    return _load_json_or_jsonl(path)


def _as_single_label(value: Any) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("标签列表为空。")
        return int(value[0])
    return int(value)


class CAILDataset(Dataset):
    def __init__(
            self,
            path: str | Path,
            tokenizer,
            max_length: int = 512,
            fields: Optional[FieldConfig] = None,
            text_is_pretokenized: bool = False,
            add_special_tokens_for_pretokenized: bool = True,
            pad_token_id: Optional[int] = None,
            fact_input_format: FactInputFormat = "auto",
    ) -> None:
        self.path = Path(path)
        self.records = load_records(self.path)
        if not self.records:
            raise ValueError(f"数据文件为空：{self.path}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fields = fields or FieldConfig()
        self.text_is_pretokenized = text_is_pretokenized
        self.add_special_tokens_for_pretokenized = add_special_tokens_for_pretokenized
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id
        self.fact_input_format = fact_input_format

        first = self.records[0]
        self.fact_field = self.fields.fact_field or _first_existing(first, FACT_FIELD_CANDIDATES)
        self.law_field = self.fields.law_field or _first_existing(first, LAW_FIELD_CANDIDATES)
        self.accu_field = self.fields.accu_field or _first_existing(first, ACCU_FIELD_CANDIDATES)
        self.term_field = self.fields.term_field or _first_existing(first, TERM_FIELD_CANDIDATES)

        missing = [
            name for name, value in (
                ("fact", self.fact_field), ("law", self.law_field),
                ("accu", self.accu_field), ("term", self.term_field),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                f"{self.path} 无法识别字段：{missing}。"
                "请通过 --fact-field/--law-field/--accu-field/--term-field 显式指定。"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _encode_ids(self, ids: Sequence[int]) -> Tuple[List[int], List[int]]:
        ids = [int(x) for x in ids[: self.max_length]]
        attn = [1 if token_id != self.pad_token_id else 0 for token_id in ids]
        if len(ids) < self.max_length:
            pad_len = self.max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
        return ids, attn

    def _encode_plain_text(self, text: str) -> Tuple[List[int], List[int]]:
        encoded = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_attention_mask=True,
            add_special_tokens=True,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _encode_pretokenized(self, tokens: List[str]) -> Tuple[List[int], List[int]]:
        encoded = self.tokenizer(
            tokens, truncation=True, padding="max_length",
            max_length=self.max_length, return_attention_mask=True,
            is_split_into_words=True,
            add_special_tokens=self.add_special_tokens_for_pretokenized,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _auto_encode(self, fact_value: Any) -> Tuple[List[int], List[int]]:
        if isinstance(fact_value, (list, tuple)):
            if fact_value and isinstance(fact_value[0], int):
                return self._encode_ids(fact_value)
            return self._encode_pretokenized([str(t) for t in fact_value])
        text = str(fact_value)
        if " " in text.strip() and self.fact_field and self.fact_field.endswith("cut"):
            return self._encode_pretokenized(text.split())
        return self._encode_plain_text(text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        fact_value = record[self.fact_field]
        fmt = self.fact_input_format

        if fmt == "bert_ids":
            input_ids, attention_mask = self._encode_ids(fact_value)
        elif fmt == "text":
            input_ids, attention_mask = self._encode_plain_text(str(fact_value))
        elif fmt == "pretokenized_words":
            tokens = fact_value.split() if isinstance(fact_value, str) else [str(t) for t in fact_value]
            input_ids, attention_mask = self._encode_pretokenized(tokens)
        elif fmt == "pretokenized_wordpieces":
            input_ids, attention_mask = self._encode_ids(fact_value)
        else:
            input_ids, attention_mask = self._auto_encode(fact_value)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "law": torch.tensor(_as_single_label(record[self.law_field]), dtype=torch.long),
            "accu": torch.tensor(_as_single_label(record[self.accu_field]), dtype=torch.long),
            "term": torch.tensor(_as_single_label(record[self.term_field]), dtype=torch.long),
        }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


# ============================================================================
# 3. 标签库（与 content_8 完全一致）
# ============================================================================

class LabelStore:
    def __init__(
            self,
            law_texts: List[str],
            accu_texts: List[str],
            term_texts: List[str],
    ) -> None:
        self.law_texts = law_texts
        self.accu_texts = accu_texts
        self.term_texts = term_texts

    def summary(self) -> Dict[str, int]:
        return {"law": len(self.law_texts), "accu": len(self.accu_texts), "term": len(self.term_texts)}


def _read_label_file(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines


def build_label_store(
        num_law: int, num_accu: int, num_term: int,
        law_label_file: Optional[Path] = None,
        accu_label_file: Optional[Path] = None,
        term_label_file: Optional[Path] = None,
) -> LabelStore:
    law_texts = _read_label_file(law_label_file) or [f"法条{i}" for i in range(num_law)]
    accu_texts = _read_label_file(accu_label_file) or [f"罪名{i}" for i in range(num_accu)]
    term_texts = _read_label_file(term_label_file) or [f"刑期{i}" for i in range(num_term)]
    law_texts = (law_texts + [f"法条{i}" for i in range(len(law_texts), num_law)])[:num_law]
    accu_texts = (accu_texts + [f"罪名{i}" for i in range(len(accu_texts), num_accu)])[:num_accu]
    term_texts = (term_texts + [f"刑期{i}" for i in range(len(term_texts), num_term)])[:num_term]
    return LabelStore(law_texts, accu_texts, term_texts)


def build_label_store_from_pkl_csv(
        mapping_pkl: Optional[Path],
        law_csv: Optional[Path], accu_csv: Optional[Path], term_csv: Optional[Path],
        num_law: int, num_accu: int, num_term: int,
) -> Optional[LabelStore]:
    if mapping_pkl is None or not mapping_pkl.exists():
        return None
    try:
        with mapping_pkl.open("rb") as f:
            mapping = pickle.load(f)
    except Exception:
        return None

    def _from_byid(key: str, n: int, prefix: str) -> List[str]:
        d = mapping.get(key, {})
        return [str(d.get(i, f"{prefix}{i}")) for i in range(n)]

    if "law2def_byid" in mapping and "accu2def_byid" in mapping and "term2def_byid" in mapping:
        return LabelStore(
            _from_byid("law2def_byid", num_law, "法条"),
            _from_byid("accu2def_byid", num_accu, "罪名"),
            _from_byid("term2def_byid", num_term, "刑期"),
        )

    def _from_csv(csv_path: Optional[Path], id2num: Dict[int, int], n: int, prefix: str) -> List[str]:
        if csv_path is None or not csv_path.exists():
            return [f"{prefix}{i}" for i in range(n)]
        rows: Dict[str, str] = {}
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keys = list(row.keys())
                    if len(keys) >= 2:
                        rows[row[keys[0]].strip()] = row[keys[1]].strip()
        except Exception:
            return [f"{prefix}{i}" for i in range(n)]
        num2id = {v: k for k, v in id2num.items()}
        texts = []
        for i in range(n):
            raw_id = num2id.get(i, i)
            text = rows.get(str(raw_id), f"{prefix}{i}")
            texts.append(text)
        return texts

    law2num = mapping.get("law2num", {})
    accu2num = mapping.get("accu2num", {})
    term2num = mapping.get("term2num", {})
    return LabelStore(
        _from_csv(law_csv, law2num, num_law, "法条"),
        _from_csv(accu_csv, accu2num, num_accu, "罪名"),
        _from_csv(term_csv, term2num, num_term, "刑期"),
    )


def validate_label_space(
        data_paths: List[Path],
        fields: FieldConfig,
        label_summary: Dict[str, int],
) -> None:
    max_ids: Dict[str, int] = {"law": -1, "accu": -1, "term": -1}
    field_map = {
        "law": fields.law_field or "law",
        "accu": fields.accu_field or "accu",
        "term": fields.term_field or "term",
    }
    for p in data_paths:
        try:
            records = load_records(p)
        except Exception:
            continue
        for rec in records:
            for task, field in field_map.items():
                if field in rec:
                    try:
                        v = _as_single_label(rec[field])
                        max_ids[task] = max(max_ids[task], v)
                    except Exception:
                        pass
    for task, max_id in max_ids.items():
        if max_id >= label_summary.get(task, 0):
            print(
                f"[WARNING] {task} 最大标签 ID={max_id} >= 类别数={label_summary[task]}。"
                "请检查 --num-{task} 参数或标签文件。"
            )


def infer_num_classes(
        args: argparse.Namespace,
        data_paths: List[Path],
        fields: FieldConfig,
        law_label_path: Optional[Path],
        accu_label_path: Optional[Path],
        term_label_path: Optional[Path],
) -> Dict[str, Optional[int]]:
    result: Dict[str, Optional[int]] = {
        "law": args.num_law if args.num_law > 0 else None,
        "accu": args.num_accu if args.num_accu > 0 else None,
        "term": args.num_term if args.num_term > 0 else None,
    }
    if all(v is not None for v in result.values()):
        return result

    def count_file(p: Optional[Path]) -> Optional[int]:
        if p is None or not p.exists():
            return None
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return len(lines) if lines else None

    from_files = {
        "law": count_file(law_label_path),
        "accu": count_file(accu_label_path),
        "term": count_file(term_label_path),
    }
    for task, val in from_files.items():
        if result[task] is None and val is not None:
            result[task] = val

    if all(v is not None for v in result.values()):
        return result

    max_ids: Dict[str, int] = {"law": -1, "accu": -1, "term": -1}
    field_map = {
        "law": fields.law_field or "law",
        "accu": fields.accu_field or "accu",
        "term": fields.term_field or "term",
    }
    for p in data_paths:
        try:
            records = load_records(p)
        except Exception:
            continue
        for rec in records:
            for task, field in field_map.items():
                if field in rec:
                    try:
                        v = _as_single_label(rec[field])
                        max_ids[task] = max(max_ids[task], v)
                    except Exception:
                        pass
    for task, max_id in max_ids.items():
        if result[task] is None and max_id >= 0:
            result[task] = max_id + 1
    return result


class TokenizedLabelStore:
    def __init__(self, label_store: LabelStore, tokenizer, max_label_length: int = 200) -> None:
        self.law = self._encode(label_store.law_texts, tokenizer, max_label_length)
        self.accu = self._encode(label_store.accu_texts, tokenizer, max_label_length)
        self.term = self._encode(label_store.term_texts, tokenizer, max_label_length)

    def _encode(self, texts: List[str], tokenizer, max_label_length: int) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_label_length,
            return_tensors="pt", add_special_tokens=True,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def to(self, device: torch.device | str) -> "TokenizedLabelStore":
        for pack in (self.law, self.accu, self.term):
            for key, value in list(pack.items()):
                pack[key] = value.to(device)
        return self


# ============================================================================
# 4. 异构 GAT 模块
# ============================================================================

class AttentionWeightRecorder:
    """
    通过 PyTorch forward hook 钩取 GAT 各层的注意力权重 α_ij，
    计算注意力分布的熵（entropy）和稀疏度（sparsity），
    用于量化注意力聚焦程度的变化。
    """

    def __init__(self, kg_reasoner: 'LegalKGReasoner') -> None:
        self.kg_reasoner = kg_reasoner
        self._hooks: List = []
        self._attn_weights: List[torch.Tensor] = []

    def _hook_fn(self, module, input, output):
        if hasattr(module, '_last_attn_weights') and module._last_attn_weights is not None:
            self._attn_weights.append(module._last_attn_weights.detach().cpu())

    def register_hooks(self) -> None:
        for name, module in self.kg_reasoner.named_modules():
            if isinstance(module, HeteroGATLayer):
                h = module.register_forward_hook(self._hook_fn)
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        self._attn_weights.clear()

    def compute_stats(self) -> Dict[str, float]:
        if not self._attn_weights:
            return {"mean_entropy": float("nan"), "mean_sparsity": float("nan"), "max_attn_mean": float("nan")}

        all_stats = []
        for attn in self._attn_weights:
            a = attn.float().flatten()
            a = a.clamp(min=1e-9)
            a = a / a.sum()

            entropy = -(a * torch.log(a + 1e-9)).sum().item()
            a_sorted, _ = torch.sort(a)
            n = len(a_sorted)
            idx = torch.arange(1, n + 1, dtype=torch.float)
            gini = (2 * (idx * a_sorted).sum() / (n * a_sorted.sum()) - (n + 1) / n).item()
            max_attn = a.max().item()

            all_stats.append({"entropy": entropy, "gini": gini, "max_attn": max_attn})

        mean_entropy  = float(np.mean([s["entropy"]  for s in all_stats]))
        mean_sparsity = float(np.mean([s["gini"]     for s in all_stats]))
        max_attn_mean = float(np.mean([s["max_attn"] for s in all_stats]))

        return {
            "mean_entropy":  mean_entropy,
            "mean_sparsity": mean_sparsity,
            "max_attn_mean": max_attn_mean,
        }


# 六种边类型的索引常量
EDGE_F_TO_LA = 0  # 案件 → 法条（被指控的法条是）
EDGE_F_TO_LC = 1  # 案件 → 罪名（被指控的罪名是）
EDGE_F_TO_LI = 2  # 案件 → 刑期（被指控的刑期是）
EDGE_LA_TO_LC = 3  # 法条 → 罪名（基准罪名为）
EDGE_LC_TO_LA = 4  # 罪名 → 法条（基准法条为）
EDGE_LC_TO_LI = 5  # 罪名 → 刑期（基准刑期为）
NUM_EDGE_TYPES = 6


class HeteroGATLayer(nn.Module):
    """异构图注意力层（单层，支持 NUM_EDGE_TYPES 种有向边类型）。

    每种边类型使用独立的线性变换矩阵 W_r，实现关系感知的消息传递。
    注意力机制使用非对称设计：源节点和目标节点各自有独立的注意力向量。
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1,
            residual: bool = True,
            num_edge_types: int = NUM_EDGE_TYPES,
    ) -> None:
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim 必须能被 num_heads 整除"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.residual = residual
        self.num_edge_types = num_edge_types

        # 每种边类型独立的变换矩阵 [num_edge_types, in_dim, out_dim]
        self.W_src = nn.Parameter(torch.empty(num_edge_types, in_dim, out_dim))
        self.W_dst = nn.Parameter(torch.empty(num_edge_types, in_dim, out_dim))

        # 非对称注意力向量 [num_edge_types, num_heads, head_dim * 2]
        self.attn_vec = nn.Parameter(torch.empty(num_edge_types, num_heads, self.head_dim * 2))

        # 残差投影（当 in_dim != out_dim 时使用）
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(out_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_src.view(-1, self.in_dim, self.out_dim).reshape(-1, self.out_dim))
        nn.init.xavier_uniform_(self.W_dst.view(-1, self.in_dim, self.out_dim).reshape(-1, self.out_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(-1, self.head_dim * 2))

    def forward(
            self,
            h: torch.Tensor,  # [V, in_dim] 所有节点的输入表征
            edge_index: torch.Tensor,  # [2, E] 边的源节点和目标节点索引
            edge_type: torch.Tensor,  # [E] 每条边的类型（0~5）
    ) -> torch.Tensor:
        """
        Args:
            h:          [V, in_dim]
            edge_index: [2, E]，edge_index[0] 是源节点，edge_index[1] 是目标节点
            edge_type:  [E]

        Returns:
            h_out: [V, out_dim]
        """
        V = h.size(0)
        E = edge_index.size(1)
        src_idx = edge_index[0]  # [E]
        dst_idx = edge_index[1]  # [E]
        r_type = edge_type  # [E]

        # 获取源节点和目标节点的特征
        h_src = h[src_idx]  # [E, in_dim]
        h_dst = h[dst_idx]  # [E, in_dim]

        # 优化显存：避免创建 [E, in_dim, out_dim] 巨型张量，按关系类型分组计算
        msg_src = torch.empty(E, self.out_dim, device=h.device, dtype=h.dtype)
        msg_dst = torch.empty(E, self.out_dim, device=h.device, dtype=h.dtype)

        for r in range(self.num_edge_types):
            mask = (r_type == r)
            if not mask.any():
                continue
            # [num_edges_r, in_dim] @ [in_dim, out_dim] -> [num_edges_r, out_dim]
            # 显式转换结果 dtype 以匹配 msg_src (h.dtype)
            res_src = h_src[mask] @ self.W_src[r].to(h.dtype)
            res_dst = h_dst[mask] @ self.W_dst[r].to(h.dtype)
            msg_src[mask] = res_src.to(h.dtype)
            msg_dst[mask] = res_dst.to(h.dtype)

        # reshape 为多头：[E, num_heads, head_dim]
        msg_src = msg_src.view(E, self.num_heads, self.head_dim)
        msg_dst = msg_dst.view(E, self.num_heads, self.head_dim)

        # 拼接源和目标的表征用于注意力计算：[E, num_heads, head_dim*2]
        msg_cat = torch.cat([msg_src, msg_dst], dim=-1)

        # 注意力分数：attn_vec[r] 是 [num_heads, head_dim*2]
        attn_r = self.attn_vec[r_type].to(h.dtype)  # [E, num_heads, head_dim*2]
        # 点积：[E, num_heads]
        e = self.leaky_relu((msg_cat * attn_r).sum(dim=-1))  # [E, num_heads]

        # softmax（按目标节点分组）
        # 用 scatter softmax：先 exp，再按 dst_idx 归一化
        e_exp = torch.exp(e - e.max())  # 数值稳定
        # 为每个目标节点累加 exp(e)：[V, num_heads]
        denom = torch.zeros(V, self.num_heads, device=h.device, dtype=e_exp.dtype)
        denom.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.num_heads), e_exp)
        # 避免除零
        alpha = e_exp / (denom[dst_idx] + 1e-9)  # [E, num_heads]
        
        # 保存注意力权重用于分析（形状: [E, num_heads]）
        self._last_attn_weights = alpha.detach()
        
        alpha = self.dropout(alpha)

        # 消息聚合：alpha * msg_src → 按 dst_idx 累加
        # msg_src: [E, num_heads, head_dim]，alpha: [E, num_heads, 1]
        weighted_msg = msg_src * alpha.unsqueeze(-1)  # [E, num_heads, head_dim]
        # 展平多头：[E, out_dim]
        weighted_msg = weighted_msg.view(E, self.out_dim)

        # 按目标节点聚合：[V, out_dim]
        h_out = torch.zeros(V, self.out_dim, device=h.device, dtype=weighted_msg.dtype)
        h_out.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.out_dim), weighted_msg)

        # 残差连接
        if self.residual:
            if self.residual_proj is not None:
                h_res = self.residual_proj(h)
            else:
                h_res = h
            h_out = h_out + h_res

        h_out = self.layer_norm(h_out)
        return h_out


class LegalKGReasoner(nn.Module):
    """法律知识图谱推理器。

    负责：
    1. 构建完整图（训练集案件 + 标签节点）的边索引
    2. 维护案件节点的 epoch 级缓存（no_grad）
    3. 在完整图上运行 K 层异构 GAT，返回增强后的标签表征
    4. 提供 precompute_label_vectors()：在推理前一次性计算并固定标签向量

    图节点布局：
      [0 .. N_case-1]                 → 训练集案件节点（F）
      [N_case .. N_case+N_law-1]      → 法条节点（L_A）
      [N_case+N_law .. N_case+N_law+N_accu-1] → 罪名节点（L_C）
      [N_case+N_law+N_accu .. V-1]    → 刑期节点（L_I）
    """

    def __init__(
            self,
            hidden_dim: int,
            num_law: int,
            num_accu: int,
            num_term: int,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_law = num_law
        self.num_accu = num_accu
        self.num_term = num_term
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList([
            HeteroGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout, residual=True)
            for _ in range(num_layers)
        ])

        # 案件节点缓存（CPU，epoch 级更新，no_grad）
        self._case_cache: Optional[torch.Tensor] = None  # [N_case, hidden_dim]
        self._n_case: int = 0

        # 图的边索引（训练前构建，固定不变）
        self._edge_index: Optional[torch.Tensor] = None  # [2, E]
        self._edge_type: Optional[torch.Tensor] = None  # [E]

        # 推理时使用的预计算标签向量（固定，不参与梯度）
        self._cached_law_repr: Optional[torch.Tensor] = None  # [N_law, hidden_dim]
        self._cached_accu_repr: Optional[torch.Tensor] = None  # [N_accu, hidden_dim]
        self._cached_term_repr: Optional[torch.Tensor] = None  # [N_term, hidden_dim]
        self._use_cached_labels: bool = False

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def build_graph(
            self,
            train_law_labels: List[int],  # 训练集每个案件的法条标签 [N_case]
            train_accu_labels: List[int],  # 训练集每个案件的罪名标签 [N_case]
            train_term_labels: List[int],  # 训练集每个案件的刑期标签 [N_case]
            threshold: float = 0.3,  # L→L 边的条件概率阈值
    ) -> None:
        """根据训练集标签构建完整图的边索引（训练前调用一次，固定不变）。

        节点偏移：
          法条节点 i 的全局 ID = N_case + i
          罪名节点 j 的全局 ID = N_case + N_law + j
          刑期节点 k 的全局 ID = N_case + N_law + N_accu + k
        """
        n_case = len(train_law_labels)
        self._n_case = n_case
        law_off = n_case
        accu_off = n_case + self.num_law
        term_off = n_case + self.num_law + self.num_accu

        src_list: List[int] = []
        dst_list: List[int] = []
        typ_list: List[int] = []

        # 案件 → 标签的三类边（F→L_A, F→L_C, F→L_I）
        for case_idx, (law_id, accu_id, term_id) in enumerate(
                zip(train_law_labels, train_accu_labels, train_term_labels)
        ):
            # F → L_A
            src_list.append(case_idx);
            dst_list.append(law_off + law_id);
            typ_list.append(EDGE_F_TO_LA)
            # F → L_C
            src_list.append(case_idx);
            dst_list.append(accu_off + accu_id);
            typ_list.append(EDGE_F_TO_LC)
            # F → L_I
            src_list.append(case_idx);
            dst_list.append(term_off + term_id);
            typ_list.append(EDGE_F_TO_LI)

        # 标签内部的三类边（L→L，基于法律本体知识）
        # 这里使用共现统计来建立 L_A↔L_C 和 L_C→L_I 的边：
        # 统计训练集中 law_id 和 accu_id 的共现，以及 accu_id 和 term_id 的共现
        from collections import Counter
        la_lc_counter: Counter = Counter()
        lc_la_counter: Counter = Counter()
        lc_li_counter: Counter = Counter()

        for law_id, accu_id, term_id in zip(train_law_labels, train_accu_labels, train_term_labels):
            la_lc_counter[(law_id, accu_id)] += 1
            lc_la_counter[(accu_id, law_id)] += 1
            lc_li_counter[(accu_id, term_id)] += 1

        # 统计每个法条/罪名出现的总次数，用于计算条件概率
        law_count = Counter(train_law_labels)
        accu_count = Counter(train_accu_labels)

        # L_A → L_C（基准罪名为）：P(accu|law) > threshold
        for (law_id, accu_id), cnt in la_lc_counter.items():
            if law_count[law_id] > 0 and cnt / law_count[law_id] >= threshold:
                src_list.append(law_off + law_id)
                dst_list.append(accu_off + accu_id)
                typ_list.append(EDGE_LA_TO_LC)

        # L_C → L_A（基准法条为）：P(law|accu) > threshold
        for (accu_id, law_id), cnt in lc_la_counter.items():
            if accu_count[accu_id] > 0 and cnt / accu_count[accu_id] >= threshold:
                src_list.append(accu_off + accu_id)
                dst_list.append(law_off + law_id)
                typ_list.append(EDGE_LC_TO_LA)

        # L_C → L_I（基准刑期为）：P(term|accu) > threshold
        for (accu_id, term_id), cnt in lc_li_counter.items():
            if accu_count[accu_id] > 0 and cnt / accu_count[accu_id] >= threshold:
                src_list.append(accu_off + accu_id)
                dst_list.append(term_off + term_id)
                typ_list.append(EDGE_LC_TO_LI)

        self._edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        self._edge_type = torch.tensor(typ_list, dtype=torch.long)

        total_nodes = n_case + self.num_law + self.num_accu + self.num_term
        print(
            f"[KG] 图构建完成：节点数={total_nodes}（案件={n_case}, 法条={self.num_law}, "
            f"罪名={self.num_accu}, 刑期={self.num_term}），边数={len(src_list)}"
        )

    # ------------------------------------------------------------------
    # 案件缓存（epoch 级更新）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def refresh_case_cache(
            self,
            fact_tower,
            train_dataset,
            device: torch.device,
            fp16: bool = False,
            batch_size: int = 256,
            num_workers: int = 0,
    ) -> None:
        """用当前 fact_tower 重新编码全部训练案件，更新 CPU 缓存。

        每个 epoch 开始前调用一次。案件表征以 detach 状态存储在 CPU，
        不占用 GPU 显存，不参与反向传播。

        重要：必须使用 shuffle=False 的 DataLoader，确保案件节点的表征
        顺序与 build_graph() 中 train_dataset.records 的顺序严格一致。
        """
        fact_tower.eval()
        # 使用固定顺序（shuffle=False）遍历，保证与图节点 ID 对齐
        ordered_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # ← 必须 False，保证顺序与图节点 ID 一致
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
        )
        all_reprs: List[torch.Tensor] = []
        for batch in tqdm(ordered_loader, desc="[KG] 刷新案件缓存", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast(enabled=fp16 and device.type == "cuda"):
                repr_ = fact_tower(input_ids=input_ids, attention_mask=attention_mask)
            all_reprs.append(repr_.detach().cpu())
        self._case_cache = torch.cat(all_reprs, dim=0)  # [N_case, hidden_dim]，存在 CPU
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # GAT 前向传播（训练时调用）
    # ------------------------------------------------------------------

    def forward(
            self,
            law_repr: torch.Tensor,  # [N_law, hidden_dim]，来自 label_tower，带梯度
            accu_repr: torch.Tensor,  # [N_accu, hidden_dim]
            term_repr: torch.Tensor,  # [N_term, hidden_dim]
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在完整图上运行 GAT，返回增强后的标签表征。

        案件节点使用 CPU 缓存（no_grad），标签节点使用实时编码（带梯度）。
        """
        if self._case_cache is None:
            # 如果在推理阶段或者没有初始化案件缓存，用空的或伪造的补齐（虽然推理时不应该跑到这里，为了鲁棒性）
            case_repr = torch.zeros((self._n_case, self.hidden_dim), device=device, dtype=law_repr.dtype)
        else:
            # 案件节点：从 CPU 缓存搬到 GPU（no_grad）
            case_repr = self._case_cache.to(device)  # [N_case, hidden_dim]，no_grad

        if self._edge_index is None:
            raise RuntimeError("图结构未构建，请先调用 build_graph()。")

        # 拼接：案件 + 法条 + 罪名 + 刑期
        h = torch.cat([case_repr, law_repr, accu_repr, term_repr], dim=0)  # [V, hidden_dim]

        # 边索引搬到 GPU
        edge_index = self._edge_index.to(device)
        edge_type = self._edge_type.to(device)

        # 多层 GAT
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_type)

        # 切片取回标签节点的增强表征
        n_case = self._n_case
        law_off = n_case
        accu_off = n_case + self.num_law
        term_off = n_case + self.num_law + self.num_accu

        enhanced_law = h[law_off: law_off + self.num_law]  # [N_law, hidden_dim]
        enhanced_accu = h[accu_off: accu_off + self.num_accu]  # [N_accu, hidden_dim]
        enhanced_term = h[term_off: term_off + self.num_term]  # [N_term, hidden_dim]

        return enhanced_law, enhanced_accu, enhanced_term

    # ------------------------------------------------------------------
    # 预计算标签向量（推理前调用一次，保存固定向量）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute_label_vectors(
            self,
            label_tower,
            tokenized_labels: TokenizedLabelStore,
            device: torch.device,
            fp16: bool = False,
    ) -> None:
        """用当前 label_tower + GAT 预计算并缓存增强后的标签向量。

        推理时直接使用这些固定向量，不再调用 label_tower 或 GAT。
        在保存 best checkpoint 之前调用此函数，并将结果一并保存。
        """
        label_tower.eval()

        with autocast(enabled=fp16 and device.type == "cuda"):
            law_repr = label_tower(
                input_ids=tokenized_labels.law["input_ids"].to(device),
                attention_mask=tokenized_labels.law["attention_mask"].to(device),
            )
            accu_repr = label_tower(
                input_ids=tokenized_labels.accu["input_ids"].to(device),
                attention_mask=tokenized_labels.accu["attention_mask"].to(device),
            )
            term_repr = label_tower(
                input_ids=tokenized_labels.term["input_ids"].to(device),
                attention_mask=tokenized_labels.term["attention_mask"].to(device),
            )

            # 在完整图上跑 GAT（此时案件缓存已是最新的）
            enhanced_law, enhanced_accu, enhanced_term = self.forward(
                law_repr, accu_repr, term_repr, device
            )

        self._cached_law_repr = enhanced_law.detach().cpu()
        self._cached_accu_repr = enhanced_accu.detach().cpu()
        self._cached_term_repr = enhanced_term.detach().cpu()
        self._use_cached_labels = True

        # 释放显存/内存：预计算完成后，案件缓存不再需要（仅训练时使用）
        self._case_cache = None
        torch.cuda.empty_cache()

        print(
            f"[KG] 标签向量预计算完成：law={self._cached_law_repr.shape}, "
            f"accu={self._cached_accu_repr.shape}, term={self._cached_term_repr.shape}"
        )

    def enable_cached_labels(self) -> None:
        """切换到推理模式：使用预计算的固定标签向量。"""
        if self._cached_law_repr is None:
            raise RuntimeError("标签向量缓存为空，请先调用 precompute_label_vectors()。")
        self._use_cached_labels = True

    def disable_cached_labels(self) -> None:
        """切换回训练模式：使用实时编码的标签向量。"""
        self._use_cached_labels = False

    def get_cached_label_vectors(
            self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回预计算的固定标签向量（推理时使用）。"""
        return (
            self._cached_law_repr.to(device),
            self._cached_accu_repr.to(device),
            self._cached_term_repr.to(device),
        )

    def save_cached_vectors(self, path: Path) -> None:
        """将预计算的标签向量保存到文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "law_repr": self._cached_law_repr,
            "accu_repr": self._cached_accu_repr,
            "term_repr": self._cached_term_repr,
        }, path)
        print(f"[KG] 预计算标签向量已保存 -> {path}")

    def load_cached_vectors(self, path: Path, device: torch.device) -> None:
        """从文件加载预计算的标签向量。"""
        data = torch.load(path, map_location="cpu", weights_only=True)
        self._cached_law_repr = data["law_repr"]
        self._cached_accu_repr = data["accu_repr"]
        self._cached_term_repr = data["term_repr"]
        self._use_cached_labels = True
        print(f"[KG] 预计算标签向量已加载 <- {path}")


# ============================================================================
# 5. BERT 双塔模型（集成 GAT）
# ============================================================================

class BertTower(nn.Module):
    def __init__(
            self,
            bert: nn.Module,
            projection_dim: int = 256,
            pooling: PoolingType = "cls",
            dropout: float = 0.1,
            normalize: bool = False,
    ) -> None:
        super().__init__()
        self.bert = bert
        self.pooling = pooling
        self.normalize = normalize
        hidden_size = int(bert.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.Tanh(),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            return summed / denom
        raise ValueError(f"未知 pooling 类型：{self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        projected = self.projection(pooled)
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)
        return projected


def load_sailer_encoder(base_model_name_or_path: str, sailer_model_dir: str) -> nn.Module:
    """加载 SAILER 模型。
    SAILER 保存的 checkpoint 是包含外壳的模型，但 config.json 里声明了 architectures=["BertForCotMAE"]，
    实际上其主体还是 BertModel。我们用 AutoModel 加载其目录，如果包含 bert 前缀则提取。
    """
    from transformers import BertModel
    # 先初始化一个标准的 BERT 模型（结构一致）
    bert = BertModel.from_pretrained(base_model_name_or_path)

    # 尝试加载 SAILER 的 pytorch_model.bin
    sailer_bin = Path(sailer_model_dir) / "pytorch_model.bin"
    if not sailer_bin.exists():
        raise FileNotFoundError(f"SAILER 模型权重文件未找到：{sailer_bin}")

    state_dict = torch.load(str(sailer_bin), map_location="cpu")

    # SAILER 可能是 BertForCotMAE，权重 key 可能带有 'bert.' 前缀
    bert_state_dict = {}
    bert_keys = set(bert.state_dict().keys())
    for k, v in state_dict.items():
        if k.startswith("bert.") and k[5:] in bert_keys:
            bert_state_dict[k[5:]] = v
        elif k in bert_keys:
            bert_state_dict[k] = v

    if not bert_state_dict:
        raise RuntimeError("无法从 SAILER checkpoint 中解析出 BERT 权重。")

    incompatible = bert.load_state_dict(bert_state_dict, strict=False)
    print(f"[SAILER] 已加载 SAILER 权重。匹配 keys: {len(bert_state_dict)}，缺失: {len(incompatible.missing_keys)}")
    return bert


class SEMDRWithKG(nn.Module):
    """双塔 + 知识图谱 GAT 标签增强模型。

    训练时：
      - fact_tower 编码案件事实（带梯度）
      - label_tower 编码标签文本（带梯度）
      - LegalKGReasoner 在完整图上增强标签表征（带梯度）
      - 点积相似度 + 三任务交叉熵损失

    推理时（use_cached_labels=True）：
      - fact_tower 编码案件事实（带梯度或 no_grad）
      - 直接使用预计算的固定标签向量（不调用 label_tower 和 GAT）
      - 点积相似度
    """

    def __init__(
            self,
            model_name_or_path: str,
            projection_dim: int = 256,
            pooling: PoolingType = "cls",
            dropout: float = 0.1,
            similarity: SimilarityType = "dot",
            temperature: float = 1.0,
            share_encoder: bool = False,
            num_law: int = 103,
            num_accu: int = 119,
            num_term: int = 11,
            gat_layers: int = 2,
            gat_heads: int = 4,
            use_kg: bool = True,
            encoder_init: str = "bert",
            sailer_model_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.similarity = similarity
        self.temperature = temperature
        self.use_kg = use_kg
        normalize = similarity == "cosine"

        if encoder_init == "sailer":
            if not sailer_model_dir:
                raise ValueError("使用 --encoder-init sailer 时必须提供 --sailer-model-dir")
            fact_bert = load_sailer_encoder(model_name_or_path, sailer_model_dir)
            label_bert = fact_bert if share_encoder else load_sailer_encoder(model_name_or_path, sailer_model_dir)
        else:
            fact_bert = AutoModel.from_pretrained(model_name_or_path)
            label_bert = fact_bert if share_encoder else AutoModel.from_pretrained(model_name_or_path)

        self.fact_tower = BertTower(
            fact_bert, projection_dim=projection_dim,
            pooling=pooling, dropout=dropout, normalize=normalize,
        )
        self.label_tower = BertTower(
            label_bert, projection_dim=projection_dim,
            pooling=pooling, dropout=dropout, normalize=normalize,
        )

        if use_kg:
            self.kg_reasoner = LegalKGReasoner(
                hidden_dim=projection_dim,
                num_law=num_law, num_accu=num_accu, num_term=num_term,
                num_layers=gat_layers, num_heads=gat_heads, dropout=dropout,
            )
        else:
            self.kg_reasoner = None

    def encode_fact(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.fact_tower(input_ids=input_ids, attention_mask=attention_mask)

    def encode_labels_raw(self, labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """用 label_tower 编码标签（不经过 GAT）。"""
        return self.label_tower(input_ids=labels["input_ids"], attention_mask=labels["attention_mask"])

    def similarity_logits(self, fact_repr: torch.Tensor, label_repr: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            fact_repr = F.normalize(fact_repr, p=2, dim=-1)
            label_repr = F.normalize(label_repr, p=2, dim=-1)
        logits = fact_repr @ label_repr.t()
        if self.temperature and self.temperature != 1.0:
            logits = logits / self.temperature
        return logits

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            law_labels: Dict[str, torch.Tensor],
            accu_labels: Dict[str, torch.Tensor],
            term_labels: Dict[str, torch.Tensor],
            targets: Optional[Dict[str, torch.Tensor]] = None,
            loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device

        # 编码案件事实
        fact_repr = self.encode_fact(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_kg and self.kg_reasoner is not None:
            if self.kg_reasoner._use_cached_labels:
                # ── 推理模式：使用预计算的固定标签向量 ──────────────────────
                law_repr, accu_repr, term_repr = self.kg_reasoner.get_cached_label_vectors(device)
            else:
                # ── 训练模式：实时编码标签 + GAT 增强 ───────────────────────
                law_repr_raw = self.encode_labels_raw(law_labels)
                accu_repr_raw = self.encode_labels_raw(accu_labels)
                term_repr_raw = self.encode_labels_raw(term_labels)
                law_repr, accu_repr, term_repr = self.kg_reasoner.forward(
                    law_repr_raw, accu_repr_raw, term_repr_raw, device
                )
        else:
            # ── 无 GAT 模式（消融对比）──────────────────────────────────────
            law_repr = self.encode_labels_raw(law_labels)
            accu_repr = self.encode_labels_raw(accu_labels)
            term_repr = self.encode_labels_raw(term_labels)

        law_logits = self.similarity_logits(fact_repr, law_repr)
        accu_logits = self.similarity_logits(fact_repr, accu_repr)
        term_logits = self.similarity_logits(fact_repr, term_repr)

        output: Dict[str, torch.Tensor] = {
            "law_logits": law_logits,
            "accu_logits": accu_logits,
            "term_logits": term_logits,
            "fact_repr": fact_repr,
            "law_repr": law_repr,
            "accu_repr": accu_repr,
            "term_repr": term_repr,
        }

        if targets is not None:
            weights = {"law": 1.0, "accu": 1.0, "term": 1.0}
            if loss_weights is not None:
                weights.update(loss_weights)
            law_loss = F.cross_entropy(law_logits, targets["law"])
            accu_loss = F.cross_entropy(accu_logits, targets["accu"])
            term_loss = F.cross_entropy(term_logits, targets["term"])
            output.update({
                "law_loss": law_loss,
                "accu_loss": accu_loss,
                "term_loss": term_loss,
                "loss": (
                        weights["law"] * law_loss
                        + weights["accu"] * accu_loss
                        + weights["term"] * term_loss
                ),
            })
        return output


def build_optimizer(
        model: SEMDRWithKG,
        encoder_lr: float = 2e-5,
        head_lr: float = 1e-3,
        gat_lr: float = 1e-3,
        weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight")
    encoder_params: List[Tuple[torch.nn.Parameter, float]] = []
    gat_params: List[Tuple[torch.nn.Parameter, float]] = []
    head_params: List[Tuple[torch.nn.Parameter, float]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".bert." in name:
            group = encoder_params
            lr = encoder_lr
        elif "kg_reasoner" in name:
            group = gat_params
            lr = gat_lr
        else:
            group = head_params
            lr = head_lr
        decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        group.append((param, decay))

    optimizer_groups: List[Dict[str, Any]] = []
    for params, lr in ((encoder_params, encoder_lr), (gat_params, gat_lr), (head_params, head_lr)):
        decay_params = [p for p, decay in params if decay > 0]
        nodecay_params = [p for p, decay in params if decay == 0]
        if decay_params:
            optimizer_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
        if nodecay_params:
            optimizer_groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(optimizer_groups)


# ============================================================================
# 6. 评估指标
# ============================================================================

def classification_report_dict(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "macro_precision": float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(metrics.precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(metrics.recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def multitask_metrics(
        targets: Dict[str, List[int]], preds: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    return {task: classification_report_dict(targets[task], preds[task]) for task in ("law", "accu", "term")}


def average_macro_f1(result: Dict[str, Dict[str, float]]) -> float:
    return float(np.mean([result[task]["macro_f1"] for task in ("law", "accu", "term")]))


# ============================================================================
# 7. 预测记录（与 content_8 完全一致）
# ============================================================================

def _calculate_topk_acc(targets: List[int], logits: torch.Tensor, max_k: int = 10) -> Dict[str, float]:
    topk_acc = {}
    _, topk_preds = logits.topk(max_k, dim=-1)
    topk_preds = topk_preds.cpu().numpy()
    for k in range(1, max_k + 1):
        correct = sum(1 for i, target in enumerate(targets) if target in topk_preds[i, :k])
        topk_acc[f"top{k}_acc"] = correct / len(targets) if len(targets) > 0 else 0.0
    return topk_acc


def _get_label_text(task: str, label_id: int, label_store: LabelStore) -> str:
    if task == "law":
        return label_store.law_texts[label_id] if label_id < len(label_store.law_texts) else str(label_id)
    if task == "accu":
        return label_store.accu_texts[label_id] if label_id < len(label_store.accu_texts) else str(label_id)
    if task == "term":
        return label_store.term_texts[label_id] if label_id < len(label_store.term_texts) else str(label_id)
    return str(label_id)


def record_test_predictions(
        output_dir: Path,
        dataset: CAILDataset,
        preds_logits: Dict[str, torch.Tensor],
        targets: Dict[str, List[int]],
        fact_reprs: torch.Tensor,
        label_reprs: Dict[str, torch.Tensor],
        label_store: LabelStore,
) -> None:
    tasks = ["law", "accu", "term"]
    N = len(dataset)
    overall_metrics = {}
    for task in tasks:
        task_logits = preds_logits[task]
        task_targets = targets[task]
        task_preds = torch.argmax(task_logits, dim=-1).cpu().tolist()
        topk = _calculate_topk_acc(task_targets, task_logits, max_k=10)
        overall_metrics[task] = {
            **topk,
            "macro_p": float(metrics.precision_score(task_targets, task_preds, average="macro", zero_division=0)),
            "macro_r": float(metrics.recall_score(task_targets, task_preds, average="macro", zero_division=0)),
            "macro_f1": float(metrics.f1_score(task_targets, task_preds, average="macro", zero_division=0)),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_all_predictions.csv"
    TOP_K = 3  # 输出 Top-K 候选及其 dot_sim
    headers = ["id", "fact"]
    for task in tasks:
        headers.extend([f"{task}_target", f"{task}_pred", f"{task}_target_rank",
                        f"{task}_dot_sim", f"{task}_cosine_sim", f"{task}_l2_distance"])
        for k in range(1, TOP_K + 1):
            headers.extend([f"{task}_top{k}_label", f"{task}_top{k}_dot_sim"])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(N):
            fact_text = str(dataset.records[i].get(dataset.fact_field, ""))
            row = [i, fact_text]
            fact_vec = fact_reprs[i].unsqueeze(0).float()
            # 预计算全部标签的 dot_sim（用于 Top-K 候选分数）
            for task in tasks:
                target_id = targets[task][i]
                target_vec = label_reprs[task][target_id].unsqueeze(0).float()
                cosine_sim = F.cosine_similarity(fact_vec, target_vec).item()
                dot_sim = torch.matmul(fact_vec, target_vec.T).item()
                l2_dist = torch.norm(fact_vec - target_vec, p=2).item()
                task_logits_i = preds_logits[task][i]
                sorted_indices = torch.argsort(task_logits_i, descending=True).cpu().tolist()
                target_rank = sorted_indices.index(target_id) + 1
                pred_id = sorted_indices[0]
                row.extend([_get_label_text(task, target_id, label_store),
                            _get_label_text(task, pred_id, label_store),
                            target_rank, dot_sim, cosine_sim, l2_dist])
                # 输出 Top-K 候选及其 dot_sim
                all_label_reprs = label_reprs[task].float()  # [num_labels, dim]
                all_dot_sims = torch.matmul(fact_vec, all_label_reprs.T).squeeze(0)  # [num_labels]
                for k in range(TOP_K):
                    topk_id = sorted_indices[k]
                    topk_dot = all_dot_sims[topk_id].item()
                    row.extend([_get_label_text(task, topk_id, label_store), f"{topk_dot:.4f}"])
            writer.writerow(row)
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)
    print(f"Test all predictions saved to {csv_path}")


def record_error_predictions(
        output_dir: Path,
        dataset: CAILDataset,
        preds_logits: Dict[str, torch.Tensor],
        targets: Dict[str, List[int]],
        fact_reprs: torch.Tensor,
        label_reprs: Dict[str, torch.Tensor],
        label_store: LabelStore,
) -> None:
    tasks = ["law", "accu", "term"]
    N = len(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_error_predictions.csv"
    headers = ["id", "fact"]
    for task in tasks:
        headers.extend([f"{task}_target", f"{task}_pred", f"{task}_target_rank",
                        f"{task}_dot_sim", f"{task}_cosine_sim", f"{task}_l2_distance"])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(N):
            is_error = False
            row_data = []
            fact_vec = fact_reprs[i].unsqueeze(0).float()
            for task in tasks:
                target_id = targets[task][i]
                target_vec = label_reprs[task][target_id].unsqueeze(0).float()
                cosine_sim = F.cosine_similarity(fact_vec, target_vec).item()
                dot_sim = torch.matmul(fact_vec, target_vec.T).item()
                l2_dist = torch.norm(fact_vec - target_vec, p=2).item()
                task_logits_i = preds_logits[task][i]
                sorted_indices = torch.argsort(task_logits_i, descending=True).cpu().tolist()
                target_rank = sorted_indices.index(target_id) + 1
                pred_id = sorted_indices[0]
                if pred_id != target_id:
                    is_error = True
                row_data.extend([_get_label_text(task, target_id, label_store),
                                 _get_label_text(task, pred_id, label_store),
                                 target_rank, dot_sim, cosine_sim, l2_dist])
            if is_error:
                fact_text = str(dataset.records[i].get(dataset.fact_field, ""))
                writer.writerow([i, fact_text] + row_data)
    print(f"Test error predictions saved to {csv_path}")


def record_training_loss(output_dir: Path, history: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_loss_history.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "valid_macro_f1"])
        for record in history:
            writer.writerow([record.get("epoch", 0), record.get("train_loss", 0.0),
                             record.get("valid", {}).get("loss", 0.0),
                             record.get("valid", {}).get("avg_macro_f1", 0.0)])
    print(f"Training loss history saved to {csv_path}")


# ============================================================================
# 8. 评估函数
# ============================================================================

def run_eval(
        model: SEMDRWithKG,
        loader: DataLoader,
        labels: TokenizedLabelStore,
        device: torch.device,
        loss_weights: Dict[str, float],
        fp16: bool = False,
        return_detailed_preds: bool = False,
) -> Dict[str, Any]:
    """在给定 DataLoader 上运行评估。

    推理时模型使用预计算的固定标签向量（kg_reasoner._use_cached_labels=True），
    不再调用 label_tower 或 GAT，速度极快。
    """
    model.eval()
    preds: Dict[str, List[int]] = {"law": [], "accu": [], "term": []}
    targets: Dict[str, List[int]] = {"law": [], "accu": [], "term": []}
    all_logits: Dict[str, List[torch.Tensor]] = {"law": [], "accu": [], "term": []}
    all_fact_reprs: List[torch.Tensor] = []
    label_reprs_dict: Dict[str, torch.Tensor] = {}
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_targets = {
                "law": batch["law"].to(device),
                "accu": batch["accu"].to(device),
                "term": batch["term"].to(device),
            }
            with autocast(enabled=fp16 and device.type == "cuda"):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    law_labels=labels.law,
                    accu_labels=labels.accu,
                    term_labels=labels.term,
                    targets=batch_targets,
                    loss_weights=loss_weights,
                )
            total_loss += float(output["loss"].detach().cpu())
            steps += 1

            if return_detailed_preds:
                all_fact_reprs.append(output["fact_repr"].cpu())
                if not label_reprs_dict:
                    label_reprs_dict["law"] = output["law_repr"].cpu()
                    label_reprs_dict["accu"] = output["accu_repr"].cpu()
                    label_reprs_dict["term"] = output["term_repr"].cpu()

            for task in ("law", "accu", "term"):
                preds[task].extend(torch.argmax(output[f"{task}_logits"], dim=-1).cpu().tolist())
                targets[task].extend(batch_targets[task].cpu().tolist())
                if return_detailed_preds:
                    all_logits[task].append(output[f"{task}_logits"].cpu())

    metric = multitask_metrics(targets, preds)
    result = {
        "loss": total_loss / max(steps, 1),
        "metrics": metric,
        "avg_macro_f1": average_macro_f1(metric),
    }
    if return_detailed_preds:
        result["detailed_preds"] = {
            "logits": {task: torch.cat(all_logits[task], dim=0) for task in ("law", "accu", "term")},
            "targets": targets,
            "fact_reprs": torch.cat(all_fact_reprs, dim=0),
            "label_reprs": label_reprs_dict,
        }
    return result


# ============================================================================
# 9. Checkpoint 保存
# ============================================================================

def save_checkpoint(
        path: Path,
        model: SEMDRWithKG,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        args: argparse.Namespace,
        label_summary: Dict[str, int],
        metric: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "label_summary": label_summary,
            "metric": metric,
        },
        path,
    )


# ============================================================================
# 10. 参数解析
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEMDR-KG：双塔 + 知识图谱 GAT 标签增强。"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--valid-file", type=str, default=DEFAULT_VALID_FILE)
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--mapping-pkl", type=str, default=str(DEFAULT_MAPPING_PKL))
    parser.add_argument("--law-csv", type=str, default=DEFAULT_LAW_CSV)
    parser.add_argument("--accu-csv", type=str, default=DEFAULT_ACCU_CSV)
    parser.add_argument("--term-csv", type=str, default=DEFAULT_TERM_CSV)
    parser.add_argument("--law-label-file", type=str, default=DEFAULT_LAW_LABEL_FILE)
    parser.add_argument("--accu-label-file", type=str, default=DEFAULT_ACCU_LABEL_FILE)
    parser.add_argument("--term-label-file", type=str, default=DEFAULT_TERM_LABEL_FILE)
    # parser.add_argument("--num-law", type=int, default=12)
    # parser.add_argument("--num-accu", type=int, default=10)
    # parser.add_argument("--num-term", type=int, default=10)
    parser.add_argument("--num-law", type=int, default=103)
    parser.add_argument("--num-accu", type=int, default=119)
    parser.add_argument("--num-term", type=int, default=11)
    parser.add_argument("--fact-field", type=str, default="fact_cut")
    parser.add_argument("--law-field", type=str, default="law")
    parser.add_argument("--accu-field", type=str, default="accu")
    parser.add_argument("--term-field", type=str, default="term")
    parser.add_argument("--text-is-pretokenized", action="store_true")
    parser.add_argument(
        "--fact-input-format",
        type=str,
        choices=["auto", "bert_ids", "text", "pretokenized_words", "pretokenized_wordpieces"],
        default="pretokenized_words",
    )
    parser.add_argument("--add-special-tokens-for-pretokenized",
                        dest="add_special_tokens_for_pretokenized", action="store_true", default=True)
    parser.add_argument("--no-add-special-tokens-for-pretokenized",
                        dest="add_special_tokens_for_pretokenized", action="store_false")
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--max-label-length", type=int, default=200)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--similarity", type=str, choices=["dot", "cosine"], default="dot")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout 概率（低频罪名建议 0.3，默认 0.3）")
    parser.add_argument("--share-encoder", action="store_true")
    # 编码器初始化参数
    parser.add_argument("--encoder-init", type=str, choices=["bert", "sailer"], default="sailer",
                        help="编码器初始化方式：bert=原始AutoModel，sailer=加载SAILER模型（默认）")
    parser.add_argument("--sailer-model-dir", type=str, default=DEFAULT_SAILER_MODEL,
                        help="SAILER 模型所在目录（需包含 pytorch_model.bin），默认与 bert 同级")
    # GAT 相关参数
    parser.add_argument("--record-attn", action="store_true", help="记录 GAT 的注意力分布统计信息")
    parser.add_argument("--gat-layers", type=int, default=1,
                        help="GAT 层数（低频罪名建议 1 层，默认 1）")
    parser.add_argument("--gat-heads", type=int, default=2,
                        help="GAT 注意力头数（低频罪名建议 2 头，默认 2）")
    parser.add_argument("--gat-lr", type=float, default=5e-4,
                        help="GAT 参数的学习率（低频罪名建议 5e-4，默认 5e-4）")
    parser.add_argument("--no-kg", action="store_true", help="禁用 GAT（消融对比）")
    parser.add_argument("--kg-threshold", type=float, default=0.3, help="L→L 边的共现概率阈值（默认 0.3）")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="权重衰减（低频罪名建议 0.05，默认 0.05）")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--law-loss-weight", type=float, default=1.0)
    parser.add_argument("--accu-loss-weight", type=float, default=1.0)
    parser.add_argument("--term-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--do-test", action="store_true")
    parser.add_argument("--skip-training", action="store_true",
                        help="跳过训练循环，直接加载已有 best_model.pth.tar 进行测试（需配合 --do-test 使用）")
    parser.add_argument("--best-ckpt-name", type=str, default="best_model.pth.tar")
    # ── Early Stopping ──
    parser.add_argument(
        "--early-stopping-patience", type=int, default=5,
        help="Early stopping 耐心値：连续 N 个 epoch 验证集 avg_macro_f1 未提升则停止训练。"
             "设为 0 则禁用 early stopping（默认 5）"
    )
    # ── GAT 预热权重加载（v3 新增）──
    parser.add_argument(
        "--warmup-ckpt",
        type=str,
        default=None,
        help="（可选）TransE 预热脚本生成的最佳 GAT checkpoint 路径\n"
             "（即 warmup_gat_transe_low_v3.py 输出的 best_warmup_gat.pth）。\n"
             "若提供此参数，将在模型构建完成后、训练开始前自动加载 kg_reasoner 的权重。\n"
             "不提供则保持随机初始化（无预热）。"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境不可用 CUDA。")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(base_dir: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(maybe_path).expanduser()
    return p if p.is_absolute() else base_dir / p


def find_file(
        base_dir: Path, explicit: Optional[str],
        candidates: Sequence[str], required: bool = True,
) -> Optional[Path]:
    if explicit:
        path = resolve_path(base_dir, explicit)
        if path is not None and path.exists():
            return path
        if required:
            raise FileNotFoundError(f"显式指定的文件不存在：{path}")
        return None
    for name in candidates:
        direct = base_dir / name
        if direct.exists():
            return direct
    for name in candidates:
        matches = sorted(base_dir.rglob(name)) if base_dir.exists() else []
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"在 {base_dir} 下没有找到候选文件：{', '.join(candidates)}。")
    return None


# ============================================================================
# 11. 主函数（训练 + 推理）
# ============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    train_path = find_file(data_dir, args.train_file, TRAIN_FILE_CANDIDATES, required=True)
    valid_path = find_file(data_dir, args.valid_file, VALID_FILE_CANDIDATES, required=True)
    test_path = find_file(data_dir, args.test_file, TEST_FILE_CANDIDATES, required=False)
    law_label_path = find_file(data_dir, args.law_label_file, LAW_LABEL_CANDIDATES, required=False)
    accu_label_path = find_file(data_dir, args.accu_label_file, ACCU_LABEL_CANDIDATES, required=False)
    term_label_path = find_file(data_dir, args.term_label_file, TERM_LABEL_CANDIDATES, required=False)

    def _resolve_aux(name: Optional[str], candidates: Sequence[str]) -> Optional[Path]:
        if not name:
            return None
        p = Path(name).expanduser()
        if p.is_absolute() and p.exists():
            return p
        for base in (data_dir, SCRIPT_DIR):
            cand = base / name
            if cand.exists():
                return cand
        return find_file(data_dir, name, candidates, required=False)

    mapping_pkl_path = _resolve_aux(args.mapping_pkl, MAPPING_PKL_CANDIDATES)
    law_csv_path = _resolve_aux(args.law_csv, (DEFAULT_LAW_CSV,))
    accu_csv_path = _resolve_aux(args.accu_csv, (DEFAULT_ACCU_CSV,))
    term_csv_path = _resolve_aux(args.term_csv, (DEFAULT_TERM_CSV,))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = FieldConfig(
        fact_field=args.fact_field, law_field=args.law_field,
        accu_field=args.accu_field, term_field=args.term_field,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    def _make_dataset(path: Path) -> CAILDataset:
        return CAILDataset(
            path, tokenizer=tokenizer, max_length=args.max_length,
            fields=fields, text_is_pretokenized=args.text_is_pretokenized,
            add_special_tokens_for_pretokenized=args.add_special_tokens_for_pretokenized,
            fact_input_format=args.fact_input_format,
        )

    train_dataset = _make_dataset(train_path)
    valid_dataset = _make_dataset(valid_path)
    test_dataset = _make_dataset(test_path) if test_path is not None and test_path.exists() else None

    data_paths = [p for p in [train_path, valid_path, test_path] if p is not None and p.exists()]
    num_classes = infer_num_classes(args, data_paths, fields, law_label_path, accu_label_path, term_label_path)
    if any(value is None for value in num_classes.values()):
        raise ValueError(f"无法推断类别数：{num_classes}。")
    num_classes = {k: int(v) for k, v in num_classes.items()}

    label_store = build_label_store_from_pkl_csv(
        mapping_pkl=mapping_pkl_path,
        law_csv=law_csv_path, accu_csv=accu_csv_path, term_csv=term_csv_path,
        num_law=num_classes["law"], num_accu=num_classes["accu"], num_term=num_classes["term"],
    )
    if label_store is None:
        label_store = build_label_store(
            num_law=num_classes["law"], num_accu=num_classes["accu"], num_term=num_classes["term"],
            law_label_file=law_label_path, accu_label_file=accu_label_path, term_label_file=term_label_path,
        )
    validate_label_space(data_paths, fields, label_store.summary())

    print(f"Data: train={len(train_dataset)}, valid={len(valid_dataset)}, "
          f"test={len(test_dataset) if test_dataset else 0}")
    print(f"Labels: {label_store.summary()}")
    print(f"Mode: {'纯双塔（无 GAT）' if args.no_kg else f'双塔 + GAT（{args.gat_layers}层 {args.gat_heads}头）'}")

    device = get_device(args.device)
    labels = TokenizedLabelStore(label_store, tokenizer, max_label_length=args.max_label_length).to(device)

    model = SEMDRWithKG(
        model_name_or_path=args.bert_model,
        projection_dim=args.projection_dim,
        pooling=args.pooling,
        dropout=args.dropout,
        similarity=args.similarity,
        temperature=args.temperature,
        share_encoder=args.share_encoder,
        num_law=num_classes["law"],
        num_accu=num_classes["accu"],
        num_term=num_classes["term"],
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        use_kg=not args.no_kg,
        encoder_init=args.encoder_init,
        sailer_model_dir=args.sailer_model_dir,
    ).to(device)

    eval_batch_size = args.eval_batch_size or args.batch_size
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = make_loader(valid_dataset, eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = (
        make_loader(test_dataset, eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if test_dataset is not None else None
    )

    # ── 构建知识图谱（训练前一次性完成）────────────────────────────────────
    if not args.no_kg and model.kg_reasoner is not None:
        print("[KG] 从训练集构建知识图谱...")
        train_law_labels = [_as_single_label(r[train_dataset.law_field]) for r in train_dataset.records]
        train_accu_labels = [_as_single_label(r[train_dataset.accu_field]) for r in train_dataset.records]
        train_term_labels = [_as_single_label(r[train_dataset.term_field]) for r in train_dataset.records]
        model.kg_reasoner.build_graph(
            train_law_labels, train_accu_labels, train_term_labels,
            threshold=args.kg_threshold,  # 使用命令行参数，默认 0.3
        )

    # ── 加载 GAT 预热权重（若提供 --warmup-ckpt）────────────────────────────────────────────────────────────────────────────────
    if args.warmup_ckpt and not args.no_kg and model.kg_reasoner is not None:
        warmup_ckpt_path = Path(args.warmup_ckpt).expanduser().resolve()
        if not warmup_ckpt_path.exists():
            raise FileNotFoundError(
                f"[WarmupCkpt] 指定的预热 checkpoint 不存在：{warmup_ckpt_path}\n"
                f"请先运行 warmup_gat_transe_low_v3.py 生成预热权重。"
            )
        print(f"[WarmupCkpt] 加载 GAT 预热权重: {warmup_ckpt_path}")
        warmup_state = torch.load(warmup_ckpt_path, map_location=device)
        # 预热 checkpoint 中保存的是 kg_reasoner_state_dict
        if "kg_reasoner_state_dict" not in warmup_state:
            raise KeyError(
                f"[WarmupCkpt] checkpoint 中未找到 'kg_reasoner_state_dict' 键。\n"
                f"实际键名: {list(warmup_state.keys())}"
            )
        # strict=False 允许部分匹配（防止版本差异导致加载失败）
        missing, unexpected = model.kg_reasoner.load_state_dict(
            warmup_state["kg_reasoner_state_dict"], strict=False
        )
        warmup_epoch = warmup_state.get("epoch", "N/A")
        warmup_agg   = warmup_state.get("total_aggregations", "N/A")
        warmup_mrr   = warmup_state.get("metrics", {}).get("MRR", "N/A")
        print(f"[WarmupCkpt] 预热权重加载成功！")
        print(f"  预热 epoch={warmup_epoch}, 聚合次数={warmup_agg}, 验证 MRR={warmup_mrr}")
        if missing:
            print(f"  [Warning] 未匹配的层（保持随机初始化）: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [Warning] checkpoint 中多余的键（已忽略）: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    elif args.warmup_ckpt and args.no_kg:
        print("[WarmupCkpt] 警告：提供了 --warmup-ckpt 但开启了 --no-kg，预热权重将被忽略。")
    else:
        print("[WarmupCkpt] 未提供 --warmup-ckpt，GAT 保持随机初始化（无预热）。")

    # ── 训练前基线评估：记录未经微调的原始 BERT 预测表现 ──────────────────
    print("\n[Baseline] 评估未训练 BERT 的预测表现（验证集）...")
    # 训练前 GAT 权重随机初始化，需先预计算一次标签向量供评估使用
    if not args.no_kg and model.kg_reasoner is not None:
        model.kg_reasoner.refresh_case_cache(
            model.fact_tower, train_dataset, device,
            fp16=args.fp16,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
        )
        model.kg_reasoner.precompute_label_vectors(
            model.label_tower, labels, device, fp16=args.fp16
        )
        model.kg_reasoner.enable_cached_labels()
    pretrain_eval = run_eval(
        model, valid_loader, labels, device,
        loss_weights={"law": args.law_loss_weight, "accu": args.accu_loss_weight, "term": args.term_loss_weight},
        fp16=args.fp16,
        return_detailed_preds=True,
    )
    pretrain_detailed = pretrain_eval.pop("detailed_preds", None)
    (output_dir / "pretrain_eval_result.json").write_text(
        json.dumps(pretrain_eval, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Baseline] avg_macro_f1={pretrain_eval['avg_macro_f1']:.4f}")
    for task, res in pretrain_eval["metrics"].items():
        print(f"  {task:4s}: acc={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f}")
    # 保存未训练时 GAT 侧的判决标签向量（law/accu/term repr）和案件侧 fact_tower 向量
    if pretrain_detailed is not None:
        torch.save(
            {
                "law_repr":  pretrain_detailed["label_reprs"].get("law"),
                "accu_repr": pretrain_detailed["label_reprs"].get("accu"),
                "term_repr": pretrain_detailed["label_reprs"].get("term"),
                "epoch": 0,
                "note": "pretrain: label repr from label_tower + GAT (random init)",
            },
            output_dir / "pretrain_label_repr_bank.pt",
        )
        torch.save(
            {
                "fact_repr": pretrain_detailed["fact_reprs"],
                "epoch": 0,
                "note": "pretrain: fact repr from fact_tower (random projection head, no fine-tuning)",
            },
            output_dir / "pretrain_fact_repr_bank.pt",
        )
        print(f"[Baseline] 预训练向量已保存 -> pretrain_label_repr_bank.pt / pretrain_fact_repr_bank.pt")
    if not args.no_kg and model.kg_reasoner is not None:
        model.kg_reasoner.disable_cached_labels()  # 恢复训练模式
    model.train()
    print("[Baseline] 完成。开始正式训练...\n")

    optimizer = build_optimizer(
        model, encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        gat_lr=args.gat_lr, weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    total_training_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    scaler = GradScaler(enabled=args.fp16 and device.type == "cuda")
    loss_weights = {"law": args.law_loss_weight, "accu": args.accu_loss_weight, "term": args.term_loss_weight}

    best_score = -1.0
    history: List[Dict[str, Any]] = []
    global_step = 0
    # 预计算标签向量的保存路径（与 best checkpoint 同目录）
    cached_label_path = output_dir / "best_model_cached_labels.pt"

    # ── Early Stopping 计数器 ──
    no_improve_epochs: int = 0
    early_stop_triggered: bool = False

    # ── 初始化注意力记录器 ──
    attn_recorder: Optional[AttentionWeightRecorder] = None
    if args.record_attn and not args.no_kg and model.kg_reasoner is not None:
        attn_recorder = AttentionWeightRecorder(model.kg_reasoner)
        attn_recorder.register_hooks()
        print("[Attn] 注意力权重记录器已注册，将在验证时收集 GAT 注意力分布")

    # CSV 结果文件初始化
    csv_path = output_dir / "train_metrics.csv"
    csv_header = ["epoch", "train_loss", "valid_loss", "valid_macro_f1"]
    if args.record_attn:
        csv_header += ["attn_entropy", "attn_sparsity", "max_attn"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # ── 训练主循环 ──────────────────────────────────────────────────────────
    if args.skip_training:
        print("[SkipTraining] --skip-training 已指定，跳过训练循环，直接进入测试阶段。")
    for epoch in range(1, args.epochs + 1) if not args.skip_training else []:

        # ── Epoch 开始：刷新案件缓存（no_grad，epoch 级更新）──────────────
        if not args.no_kg and model.kg_reasoner is not None:
            print(f"[KG] Epoch {epoch}：刷新案件节点缓存...")
            model.kg_reasoner.disable_cached_labels()  # 确保训练时不使用固定标签向量
            model.kg_reasoner.refresh_case_cache(
                model.fact_tower, train_dataset, device,
                fp16=args.fp16,
                batch_size=args.batch_size * 2,  # 缓存刷新不需要梯度，batch 可以更大
                num_workers=args.num_workers,
            )
            model.train()  # refresh_case_cache 会调用 eval()，这里恢复训练模式

        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_targets = {
                "law": batch["law"].to(device),
                "accu": batch["accu"].to(device),
                "term": batch["term"].to(device),
            }

            with autocast(enabled=args.fp16 and device.type == "cuda"):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    law_labels=labels.law,
                    accu_labels=labels.accu,
                    term_labels=labels.term,
                    targets=batch_targets,
                    loss_weights=loss_weights,
                )
                loss = output["loss"] / max(args.gradient_accumulation_steps, 1)

            scaler.scale(loss).backward()
            epoch_loss += float(output["loss"].detach().cpu())
            step_count += 1

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix(loss=f"{epoch_loss / step_count:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = epoch_loss / max(step_count, 1)

        # ── Epoch 结束：预计算标签向量，切换到推理模式做 valid 评估 ────────
        attn_stats: Dict[str, float] = {}
        if not args.no_kg and model.kg_reasoner is not None:
            print(f"[KG] Epoch {epoch}：预计算增强标签向量...")
            
            # 注意力记录器：清空上一次的注意力缓存
            if attn_recorder is not None:
                attn_recorder.clear()
                
            # 预计算标签向量（此过程会调用 GAT forward，触发 hook）
            model.kg_reasoner.precompute_label_vectors(
                model.label_tower, labels, device, fp16=args.fp16
            )
            
            # 注意力记录器：计算统计量
            if attn_recorder is not None:
                attn_stats = attn_recorder.compute_stats()
                attn_recorder.clear()
                print(f"  [Attn] Entropy={attn_stats.get('mean_entropy', float('nan')):.4f}, "
                      f"Sparsity={attn_stats.get('mean_sparsity', float('nan')):.4f}, "
                      f"MaxAttn={attn_stats.get('max_attn_mean', float('nan')):.4f}")
                      
            # 切换到推理模式：valid 评估时使用固定标签向量
            model.kg_reasoner.enable_cached_labels()

        valid_result = run_eval(model, valid_loader, labels, device, loss_weights=loss_weights, fp16=args.fp16)
        score = float(valid_result["avg_macro_f1"])

        epoch_record = {
            "epoch": epoch, "train_loss": train_loss,
            "valid": valid_result, "global_step": global_step,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 将注意力统计信息写入 epoch 记录
        if attn_stats:
            epoch_record["attn_entropy"] = attn_stats.get("mean_entropy", float("nan"))
            epoch_record["attn_sparsity"] = attn_stats.get("mean_sparsity", float("nan"))
            epoch_record["max_attn_mean"] = attn_stats.get("max_attn_mean", float("nan"))
        history.append(epoch_record)

        print(
            f"\nEpoch {epoch} | train_loss={train_loss:.4f} | "
            f"valid_loss={valid_result['loss']:.4f} | avg_macro_f1={score:.4f}"
        )
        for task, result in valid_result["metrics"].items():
            print(
                f"  {task:4s}: acc={result['accuracy']:.4f} "
                f"macro_p={result['macro_precision']:.4f} "
                f"macro_r={result['macro_recall']:.4f} "
                f"macro_f1={result['macro_f1']:.4f}"
            )

        (output_dir / "train_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 写入 CSV（简单追加）
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            row = [
                epoch,
                f"{train_loss:.6f}",
                f"{valid_result['loss']:.6f}",
                f"{valid_result['avg_macro_f1']:.6f}"
            ]
            if args.record_attn:
                row += [
                    f"{attn_stats.get('mean_entropy', float('nan')):.6f}",
                    f"{attn_stats.get('mean_sparsity', float('nan')):.6f}",
                    f"{attn_stats.get('max_attn_mean', float('nan')):.6f}",
                ]
            writer.writerow(row)

        # ── 保存最佳 checkpoint + Early Stopping 检查 ────────────────
        if score > best_score:
            best_score = score
            no_improve_epochs = 0  # 有提升，重置计数器
            save_checkpoint(
                output_dir / args.best_ckpt_name,
                model, optimizer, scheduler, epoch,
                args, label_store.summary(), valid_result,
            )
            # 同步保存预计算的标签向量（推理时直接加载，无需重新跑 GAT）
            if not args.no_kg and model.kg_reasoner is not None:
                model.kg_reasoner.save_cached_vectors(cached_label_path)
            print(f"  Saved best checkpoint -> {args.best_ckpt_name} | avg_macro_f1={best_score:.4f}")

            # ── 保存 best epoch 的 GAT 判决标签向量和案件侧 fact_tower 向量 ───────────────
            # 获取当前 epoch 的增强标签向量（kg_reasoner 已在上方 precompute 并 enable_cached_labels）
            if not args.no_kg and model.kg_reasoner is not None:
                _cached_law, _cached_accu, _cached_term = model.kg_reasoner.get_cached_label_vectors(device)
                torch.save(
                    {
                        "law_repr":  _cached_law.cpu(),
                        "accu_repr": _cached_accu.cpu(),
                        "term_repr": _cached_term.cpu(),
                        "epoch": epoch,
                        "note": f"best epoch={epoch}: GAT-enhanced label repr (law/accu/term)",
                    },
                    output_dir / "best_label_repr_bank.pt",
                )
            # 保存当前 epoch 的案件侧 fact_tower 向量（遍历训练集）
            _fact_reprs_list: List[torch.Tensor] = []
            model.eval()
            with torch.no_grad():
                _bank_loader = make_loader(
                    train_dataset, args.eval_batch_size or args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                )
                for _batch in _bank_loader:
                    _ids  = _batch["input_ids"].to(device)
                    _mask = _batch["attention_mask"].to(device)
                    with autocast(enabled=args.fp16 and device.type == "cuda"):
                        _fact_repr = model.encode_fact(input_ids=_ids, attention_mask=_mask)
                    _fact_reprs_list.append(_fact_repr.cpu())
            model.train()
            torch.save(
                {
                    "fact_repr": torch.cat(_fact_reprs_list, dim=0),
                    "epoch": epoch,
                    "note": f"best epoch={epoch}: fact_tower repr for all training samples",
                },
                output_dir / "best_fact_repr_bank.pt",
            )
            print(f"  向量库已更新 -> best_label_repr_bank.pt / best_fact_repr_bank.pt (epoch={epoch})")

        else:
            # 未提升，计数器加一
            no_improve_epochs += 1
            print(f"  [EarlyStopping] 未提升 {no_improve_epochs}/{args.early_stopping_patience} "
                  f"| best={best_score:.4f}")
            if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
                print(f"\n[EarlyStopping] 连续 {args.early_stopping_patience} 个 epoch 无提升，"
                      f"在 epoch {epoch} 提前停止训练。最佳验证 avg_macro_f1={best_score:.4f}")
                early_stop_triggered = True
                break

    if early_stop_triggered:
        print("[EarlyStopping] 训练已提前终止。")

    record_training_loss(output_dir, history)

    # ── 测试集评估 ───────────────────────────────────────────────────────────
    if args.do_test and test_loader is not None:
        best_ckpt = output_dir / args.best_ckpt_name
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Test] 加载最佳 checkpoint: epoch={checkpoint.get('epoch', '?')}")

        # 加载预计算的标签向量（推理时直接使用，不再跑 GAT）
        if not args.no_kg and model.kg_reasoner is not None:
            if cached_label_path.exists():
                model.kg_reasoner.load_cached_vectors(cached_label_path, device)
            else:
                # 如果文件不存在（如直接跑 --do-test），重新计算
                print("[KG] 预计算标签向量文件不存在，重新计算...")
                model.kg_reasoner.refresh_case_cache(
                    model.fact_tower, train_loader, device, fp16=args.fp16
                )
                model.kg_reasoner.precompute_label_vectors(
                    model.label_tower, labels, device, fp16=args.fp16
                )
                model.kg_reasoner.enable_cached_labels()

        test_result = run_eval(
            model, test_loader, labels, device,
            loss_weights=loss_weights, fp16=args.fp16,
            return_detailed_preds=True,
        )
        detailed_preds = test_result.pop("detailed_preds", None)
        (output_dir / "test_result.json").write_text(
            json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Test avg_macro_f1={test_result['avg_macro_f1']:.4f}")

        if detailed_preds is not None:
            record_test_predictions(
                output_dir, test_dataset,
                detailed_preds["logits"], detailed_preds["targets"],
                detailed_preds["fact_reprs"], detailed_preds["label_reprs"],
                label_store,
            )
            record_error_predictions(
                output_dir, test_dataset,
                detailed_preds["logits"], detailed_preds["targets"],
                detailed_preds["fact_reprs"], detailed_preds["label_reprs"],
                label_store,
            )

    print(f"\nTraining done. Best avg_macro_f1={best_score:.4f}. Output dir: {output_dir}")


if __name__ == "__main__":
    main()


# python semdr_kg_gat_v1_sailer_low_原 版本.py     --data-dir /home/cwadmin/Tompanda/LegalDuet/ljp_labels/cail_small     --bert-model /home/cwadmin/Tompanda/LegalDuet/bert     --sailer-model-dir /home/cwadmin/Tompanda/LegalDuet/sailer     --output-dir ./checkpoints/gat_warmup

