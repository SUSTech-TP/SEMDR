#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEMDR BERT Two-Tower Backbone for CAIL-small
================================================

这是一个单文件版训练脚本，适合你把主入口重命名为 `semdr-tt.py` 后直接放到服务器上运行。
它保留了 SEMDR 底座模型的核心形式：事实塔编码案件事实，标签塔编码 law/accu/term 三组候选标签，
再通过相似度矩阵得到三个任务的 logits，并进行三任务联合交叉熵训练。

推荐运行示例：

cd /root/Tompanda/LegalDuet/Fine-Tuning
python semdr-tt.py

当前版本已按你的服务器目录写死默认路径：BERT 位于 `/root/Tompanda/LegalDuet/bert`，
数据位于 `/root/Tompanda/LegalDuet/cail_small_standard_legalduet_ready/cail_small_standard`，
训练、验证、测试文件默认分别为 `train_cs_bert_small.json`、`valid_cs_bert_small.json`、`test_cs_bert_small.json`。
如果需要临时覆盖训练轮数、batch size 或最大 token 数，可以继续使用命令行参数：

python semdr-tt.py --max-length 512 --epochs 20 --batch-size 16 --fact-input-format auto
"""

from __future__ import annotations

import argparse
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
# 1. 位置默认值：已按你的服务器截图写死；仍可用命令行参数覆盖
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# 你的服务器目录结构：
#   主项目目录       /root/Tompanda/LegalDuet
#   当前脚本目录     /root/Tompanda/LegalDuet/Fine-Tuning
#   BERT 预训练目录  /root/Tompanda/LegalDuet/bert
#   CAIL-small 数据  /root/Tompanda/LegalDuet/cail_small_standard_legalduet_ready/cail_small_standard
# 注意：项目根目录是大写 "TomPanda"（P 大写）。
SERVER_PROJECT_ROOT = Path("/home/cwadmin/Tompanda/LegalDuet")
SERVER_FINE_TUNING_DIR = SERVER_PROJECT_ROOT / "Fine-Tuning"
# low_small 数据集目录（LADAN 风格 fact_cut 文本 + 稀疏 law/accu ID）。
# 实际位置：/root/TomPanda/LegalDuet/ljp_labels/low_small
# 如与你服务器实际路径不同，请用 --data-dir 覆盖。
SERVER_DATA_DIR = SERVER_PROJECT_ROOT / "ljp_labels" / "low_small"
SERVER_BERT_MODEL_DIR = SERVER_PROJECT_ROOT / "bert"
SERVER_OUTPUT_DIR = SERVER_FINE_TUNING_DIR / "checkpoints" / "semdr_twotower_low_small"

DEFAULT_DATA_DIR = os.environ.get("SEMDR_DATA_DIR", str(SERVER_DATA_DIR))
DEFAULT_BERT_MODEL = os.environ.get("SEMDR_BERT_MODEL", str(SERVER_BERT_MODEL_DIR))
DEFAULT_OUTPUT_DIR = os.environ.get("SEMDR_OUTPUT_DIR", str(SERVER_OUTPUT_DIR))

DEFAULT_TRAIN_FILE = "train_cs.json"
DEFAULT_VALID_FILE = "valid_cs.json"
DEFAULT_TEST_FILE = "test_cs.json"
# === 标签文本来源（新版）===
# 不再依赖手写 new_*.txt，改为从「修正版 pkl」+「cail2018 释义 csv」读取。
# 优先级：pkl 中的 law2def_byid/accu2def_byid/term2def_byid
#        > 由 pkl(law2num/accu2num) + csv 在运行时现场构建
#        > （向后兼容）旧的 new_*.txt
DEFAULT_MAPPING_PKL = "mappings_lowfreq_fixed.pkl"
DEFAULT_LAW_CSV = "cail2018law2text.csv"
DEFAULT_ACCU_CSV = "cail2018charge2text.csv"
DEFAULT_TERM_CSV = "cail2018term2text.csv"
# 以下 txt 默认值仅作为向后兼容项，正常不再使用。
DEFAULT_LAW_LABEL_FILE = "new_law.txt"
DEFAULT_ACCU_LABEL_FILE = "new_accu.txt"
DEFAULT_TERM_LABEL_FILE = "new_term.txt"

MAPPING_PKL_CANDIDATES = (
    "mappings_lowfreq_fixed.pkl",
    "mappings_lowfreq.pkl",
)

TRAIN_FILE_CANDIDATES = (
    "train_cs.json",
    "train_cs.jsonl",
    "train_cs_bert_small.json",
    "train_processed_bert.pkl",
    "train.pkl",
    "train.json",
    "train.jsonl",
)
VALID_FILE_CANDIDATES = (
    "valid_cs.json",
    "valid_cs.jsonl",
    "valid_cs_bert_small.json",
    "valid_processed_bert.pkl",
    "valid.pkl",
    "valid.json",
    "valid.jsonl",
    "val.json",
    "val.jsonl",
)
TEST_FILE_CANDIDATES = (
    "test_cs.json",
    "test_cs.jsonl",
    "test_cs_bert_small.json",
    "test_processed_bert.pkl",
    "test.pkl",
    "test.json",
    "test.jsonl",
)
LAW_LABEL_CANDIDATES = ("new_law.txt", "law.txt", "law_label.txt", "law_labels.txt")
ACCU_LABEL_CANDIDATES = ("new_accu.txt", "accu.txt", "accu_label.txt", "accu_labels.txt", "charge.txt")
TERM_LABEL_CANDIDATES = ("new_term.txt", "term.txt", "term_label.txt", "term_labels.txt")


# ============================================================================
# 2. 数据读取与字段兼容
# ============================================================================

FACT_FIELD_CANDIDATES = (
    "fact",
    "fact_cut",
    "facts",
    "content",
    "text",
    "case_fact",
    "criminal_fact",
)
LAW_FIELD_CANDIDATES = ("law", "law_label", "article", "article_label", "law_label_lists")
ACCU_FIELD_CANDIDATES = ("accu", "accu_label", "charge", "charge_label", "accu_label_lists")
TERM_FIELD_CANDIDATES = (
    "term",
    "time",
    "penalty",
    "term_label",
    "time_label",
    "imprisonment",
    "term_lists",
)

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

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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
        records.append(
            {
                "fact": fact,
                "law": int(data[law_key][i]),
                "accu": int(data[accu_key][i]),
                "term": int(data[term_key][i]),
            }
        )
    return records


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return _load_pkl(path)
    return _load_json_or_jsonl(path)


def _as_single_label(value: Any) -> int:
    """兼容 int、长度为 1 的 list/tuple/numpy array，以及 torch tensor。"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("标签列表为空。")
        return int(value[0])
    return int(value)


class CAILDataset(Dataset):
    """CAIL-small 多任务数据集，兼容 JSONL/JSON/pkl 与多种 fact 字段形式。"""

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
            name
            for name, value in (
                ("fact", self.fact_field),
                ("law", self.law_field),
                ("accu", self.accu_field),
                ("term", self.term_field),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{self.path} 无法识别字段：{missing}。"
                "请通过 --fact-field/--law-field/--accu-field/--term-field 显式指定。"
            )
        if self.fact_input_format not in {"auto", "bert_ids", "text", "pretokenized_words", "pretokenized_wordpieces"}:
            raise ValueError(f"未知 fact_input_format：{self.fact_input_format}")

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
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _encode_pretokenized_words(self, tokens: Sequence[str]) -> Tuple[List[int], List[int]]:
        encoded = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            add_special_tokens=self.add_special_tokens_for_pretokenized,
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _encode_pretokenized_wordpieces(self, tokens: Sequence[str]) -> Tuple[List[int], List[int]]:
        pieces = [str(x) for x in tokens]
        if self.add_special_tokens_for_pretokenized:
            pieces = [self.tokenizer.cls_token] + pieces + [self.tokenizer.sep_token]
        ids = self.tokenizer.convert_tokens_to_ids(pieces)
        return self._encode_ids(ids)

    def _encode_text(self, text: str, field_name: str) -> Tuple[List[int], List[int]]:
        text = text.strip()
        field_is_cut = field_name.endswith("cut")
        as_pretok = self.text_is_pretokenized or field_is_cut

        if self.fact_input_format == "text" or (self.fact_input_format == "auto" and not as_pretok):
            return self._encode_plain_text(text)

        tokens = text.split()
        if self.fact_input_format == "pretokenized_wordpieces":
            return self._encode_pretokenized_wordpieces(tokens)

        # 默认把 fact_cut 视为“词级预分词”，交给 BERT tokenizer 继续切 wordpiece。
        return self._encode_pretokenized_words(tokens)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        fact_value = item[self.fact_field]  # type: ignore[index]
        if hasattr(fact_value, "tolist"):
            fact_value = fact_value.tolist()

        if isinstance(fact_value, (list, tuple)) and all(isinstance(x, (int, float)) for x in fact_value):
            if self.fact_input_format in {"auto", "bert_ids"}:
                input_ids, attention_mask = self._encode_ids(fact_value)
            else:
                joined = " ".join(str(int(x)) for x in fact_value)
                input_ids, attention_mask = self._encode_text(joined, self.fact_field or "fact")
        elif isinstance(fact_value, (list, tuple)):
            tokens = [str(x) for x in fact_value]
            if self.fact_input_format == "pretokenized_wordpieces":
                input_ids, attention_mask = self._encode_pretokenized_wordpieces(tokens)
            elif self.fact_input_format in {"auto", "pretokenized_words"}:
                input_ids, attention_mask = self._encode_pretokenized_words(tokens)
            else:
                input_ids, attention_mask = self._encode_plain_text(" ".join(tokens))
        else:
            input_ids, attention_mask = self._encode_text(str(fact_value), self.fact_field or "fact")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "law": torch.tensor(_as_single_label(item[self.law_field]), dtype=torch.long),  # type: ignore[index]
            "accu": torch.tensor(_as_single_label(item[self.accu_field]), dtype=torch.long),  # type: ignore[index]
            "term": torch.tensor(_as_single_label(item[self.term_field]), dtype=torch.long),  # type: ignore[index]
        }


def scan_num_classes(paths: Iterable[str | Path], fields: Optional[FieldConfig] = None) -> Dict[str, int]:
    fields = fields or FieldConfig()
    max_ids = {"law": -1, "accu": -1, "term": -1}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        records = load_records(path)
        if not records:
            continue
        first = records[0]
        law_field = fields.law_field or _first_existing(first, LAW_FIELD_CANDIDATES)
        accu_field = fields.accu_field or _first_existing(first, ACCU_FIELD_CANDIDATES)
        term_field = fields.term_field or _first_existing(first, TERM_FIELD_CANDIDATES)
        if law_field is None or accu_field is None or term_field is None:
            continue
        for item in records:
            max_ids["law"] = max(max_ids["law"], _as_single_label(item[law_field]))
            max_ids["accu"] = max(max_ids["accu"], _as_single_label(item[accu_field]))
            max_ids["term"] = max(max_ids["term"], _as_single_label(item[term_field]))
    return {k: v + 1 for k, v in max_ids.items() if v >= 0}


# ============================================================================
# 3. 标签文本库：标签塔要编码这些候选标签文本
# ============================================================================

DEFAULT_TERM_TEXTS_12 = [
    "免予刑事处罚或者无期徒刑以上特殊刑期类别",
    "拘役、管制或者六个月以下有期徒刑",
    "六个月以上一年以下有期徒刑",
    "一年以上二年以下有期徒刑",
    "二年以上三年以下有期徒刑",
    "三年以上五年以下有期徒刑",
    "五年以上七年以下有期徒刑",
    "七年以上十年以下有期徒刑",
    "十年以上十五年以下有期徒刑",
    "十五年以上有期徒刑",
    "无期徒刑",
    "死刑",
]


@dataclass
class LabelStore:
    law_texts: List[str]
    accu_texts: List[str]
    term_texts: List[str]

    @property
    def num_law(self) -> int:
        return len(self.law_texts)

    @property
    def num_accu(self) -> int:
        return len(self.accu_texts)

    @property
    def num_term(self) -> int:
        return len(self.term_texts)

    def summary(self) -> Dict[str, int]:
        return {"law": self.num_law, "accu": self.num_accu, "term": self.num_term}


def read_label_file(path: Optional[str | Path], prefix: str = "") -> Optional[List[str]]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(f"{prefix}{line}" if prefix else line)
    return texts


def make_generic_labels(prefix: str, count: int, start: int = 0) -> List[str]:
    return [f"{prefix}{i}" for i in range(start, start + count)]


def make_term_texts(count: int) -> List[str]:
    if count == 12:
        return DEFAULT_TERM_TEXTS_12.copy()
    return [f"刑期类别{i}" for i in range(count)]


# ----------------------------------------------------------------------------
# 新版标签文本来源：从「修正版 pkl」+「cail2018 释义 csv」按 raw id 构建
# ----------------------------------------------------------------------------

def _read_csv_dict(path: Optional[str | Path], key_col: str, val_col: str) -> Dict[str, str]:
    """读取 cail2018 释义 csv，返回 {key: def}。文件不存在时返回空 dict。"""
    import csv as _csv
    result: Dict[str, str] = {}
    if path is None:
        return result
    path = Path(path)
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            key = (row.get(key_col) or "").strip()
            val = (row.get(val_col) or "").strip()
            if key:
                result[key] = val
    return result


def _format_article_display(article: str) -> str:
    """把 raw 法条键（如 '353'、'353_b'）转为可读的「第N条/第N条之一」。"""
    if "_" not in article:
        return f"{article}条"
    base, suffix = article.split("_", 1)
    suffix_map = {"b": "条之一", "c": "条之二", "d": "条之三"}
    return f"{base}{suffix_map.get(suffix, '条之' + suffix)}"


def build_label_store_from_pkl_csv(
    mapping_pkl: Optional[str | Path],
    law_csv: Optional[str | Path] = None,
    accu_csv: Optional[str | Path] = None,
    term_csv: Optional[str | Path] = None,
    num_law: Optional[int] = None,
    num_accu: Optional[int] = None,
    num_term: Optional[int] = None,
) -> Optional[LabelStore]:
    """从修正版 pkl（+ 可选 csv）按 raw id 构建标签释义库。

    对齐口径：标签塔第 i 行 = raw id i 的释义，与 train_cs.json 的
    law/accu/term 整数标签（稀疏 raw id）以及 F.cross_entropy 严格对齐。

    优先使用 pkl 内已离线构建好的 law2def_byid/accu2def_byid/term2def_byid；
    若没有，则用 pkl 的 law2num/accu2num 反查 + csv 现场构建；
    空洞 id 用占位文本「（未使用法条/罪名类别N）」。
    返回 None 表示无法从 pkl 构建（调用方应回退到 txt 逻辑）。
    """
    if mapping_pkl is None:
        return None
    mapping_pkl = Path(mapping_pkl)
    if not mapping_pkl.exists():
        return None
    with mapping_pkl.open("rb") as f:
        M = pickle.load(f)
    if not isinstance(M, dict):
        return None

    n_law = int(num_law or M.get("num_law") or 57)
    n_accu = int(num_accu or M.get("num_accu") or 65)
    n_term = int(num_term or M.get("num_term") or 11)

    # --- 路线 1：pkl 内已含 *_byid 离线释义，直接用 ---
    def _byid_to_list(byid: Dict[Any, str], count: int, kind: str) -> List[str]:
        out: List[str] = []
        for i in range(count):
            txt = byid.get(i, byid.get(str(i)))
            out.append(txt if txt else f"（未使用{kind}类别{i}）")
        return out

    law_byid = M.get("law2def_byid")
    accu_byid = M.get("accu2def_byid")
    term_byid = M.get("term2def_byid") or M.get("term2def")
    if law_byid and accu_byid and term_byid:
        return LabelStore(
            law_texts=_byid_to_list(law_byid, n_law, "法条"),
            accu_texts=_byid_to_list(accu_byid, n_accu, "罪名"),
            term_texts=_byid_to_list(term_byid, n_term, "刑期"),
        )

    # --- 路线 2：用 law2num/accu2num 反查 + csv 现场构建 ---
    law2num = M.get("law2num")
    accu2num = M.get("accu2num")
    if law2num is None or accu2num is None:
        return None
    law_csv_d = _read_csv_dict(law_csv, "law", "def")
    accu_csv_d = _read_csv_dict(accu_csv, "charge", "def")
    term_csv_d = _read_csv_dict(term_csv, "term", "def")

    id2article = {int(v): k for k, v in law2num.items()}
    id2charge = {int(v): k for k, v in accu2num.items()}

    law_texts: List[str] = []
    for i in range(n_law):
        if i in id2article:
            art = id2article[i]
            text = law_csv_d.get(art) or law_csv_d.get(art.split("_")[0]) or "（释义缺失）"
            law_texts.append(f"刑法第{_format_article_display(art)}：{text}")
        else:
            law_texts.append(f"（未使用法条类别{i}）")

    accu_texts: List[str] = []
    for i in range(n_accu):
        if i in id2charge:
            name = id2charge[i]
            text = accu_csv_d.get(name) or f"{name}罪。（释义缺失）"
            accu_texts.append(f"罪名：{name}。{text}")
        else:
            accu_texts.append(f"（未使用罪名类别{i}）")

    term_src = M.get("term2def") or {}
    term_texts: List[str] = []
    for i in range(n_term):
        text = term_csv_d.get(str(i)) or term_src.get(i) or f"刑期类别{i}"
        term_texts.append(text)

    return LabelStore(law_texts=law_texts, accu_texts=accu_texts, term_texts=term_texts)


def build_label_store(
    num_law: int,
    num_accu: int,
    num_term: int,
    law_label_file: Optional[str | Path] = None,
    accu_label_file: Optional[str | Path] = None,
    term_label_file: Optional[str | Path] = None,
) -> LabelStore:
    # low_small 适配：标签文件已含完整释义文本（如“刑法第158条：……”、
    # “罪名：盗窃、侮辱尸体。……”、“根据中华人民共和国刑法，判处被告人……”），
    # 因此这里不再追加任何前缀，直接按原文逐行读取，避免出现重复前缀。
    law_texts = read_label_file(law_label_file, prefix="")
    accu_texts = read_label_file(accu_label_file, prefix="")
    term_texts = read_label_file(term_label_file, prefix="")

    if law_texts is None:
        law_texts = make_generic_labels("法条类别", num_law)
    elif len(law_texts) < num_law:
        start = len(law_texts)
        law_texts.extend(make_generic_labels("法条类别", num_law - start, start=start))

    if accu_texts is None:
        accu_texts = make_generic_labels("罪名类别", num_accu)
    elif len(accu_texts) < num_accu:
        start = len(accu_texts)
        accu_texts.extend(make_generic_labels("罪名类别", num_accu - start, start=start))

    if term_texts is None:
        term_texts = make_term_texts(num_term)
    elif len(term_texts) < num_term:
        start = len(term_texts)
        term_texts.extend(make_generic_labels("刑期类别", num_term - start, start=start))

    return LabelStore(law_texts=law_texts, accu_texts=accu_texts, term_texts=term_texts)


class TokenizedLabelStore:
    def __init__(self, label_store: LabelStore, tokenizer, max_label_length: int = 128) -> None:
        self.label_store = label_store
        self.max_label_length = max_label_length
        self.law = self._encode(label_store.law_texts, tokenizer)
        self.accu = self._encode(label_store.accu_texts, tokenizer)
        self.term = self._encode(label_store.term_texts, tokenizer)

    def _encode(self, texts: List[str], tokenizer) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_label_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def to(self, device: torch.device | str) -> "TokenizedLabelStore":
        for pack in (self.law, self.accu, self.term):
            for key, value in list(pack.items()):
                pack[key] = value.to(device)
        return self


# ============================================================================
# 4. BERT 双塔模型：事实塔 + 标签塔 + 三任务相似度 logits
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


class SEMDRBaseTwoTower(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        projection_dim: int = 256,
        pooling: PoolingType = "cls",
        dropout: float = 0.1,
        similarity: SimilarityType = "dot",
        temperature: float = 1.0,
        share_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.similarity = similarity
        self.temperature = temperature
        normalize = similarity == "cosine"

        fact_bert = AutoModel.from_pretrained(model_name_or_path)
        label_bert = fact_bert if share_encoder else AutoModel.from_pretrained(model_name_or_path)

        self.fact_tower = BertTower(
            fact_bert,
            projection_dim=projection_dim,
            pooling=pooling,
            dropout=dropout,
            normalize=normalize,
        )
        self.label_tower = BertTower(
            label_bert,
            projection_dim=projection_dim,
            pooling=pooling,
            dropout=dropout,
            normalize=normalize,
        )

    def encode_fact(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.fact_tower(input_ids=input_ids, attention_mask=attention_mask)

    def encode_labels(self, labels: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        fact_repr = self.encode_fact(input_ids=input_ids, attention_mask=attention_mask)
        law_repr = self.encode_labels(law_labels)
        accu_repr = self.encode_labels(accu_labels)
        term_repr = self.encode_labels(term_labels)

        law_logits = self.similarity_logits(fact_repr, law_repr)
        accu_logits = self.similarity_logits(fact_repr, accu_repr)
        term_logits = self.similarity_logits(fact_repr, term_repr)

        output: Dict[str, torch.Tensor] = {
            "law_logits": law_logits,
            "accu_logits": accu_logits,
            "term_logits": term_logits,
            "fact_repr": fact_repr,
        }
        if targets is not None:
            weights = {"law": 1.0, "accu": 1.0, "term": 1.0}
            if loss_weights is not None:
                weights.update(loss_weights)
            law_loss = F.cross_entropy(law_logits, targets["law"])
            accu_loss = F.cross_entropy(accu_logits, targets["accu"])
            term_loss = F.cross_entropy(term_logits, targets["term"])
            output.update(
                {
                    "law_loss": law_loss,
                    "accu_loss": accu_loss,
                    "term_loss": term_loss,
                    "loss": weights["law"] * law_loss
                    + weights["accu"] * accu_loss
                    + weights["term"] * term_loss,
                }
            )
        return output


def build_optimizer(
    model: SEMDRBaseTwoTower,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight")
    encoder_params: List[Tuple[torch.nn.Parameter, float]] = []
    head_params: List[Tuple[torch.nn.Parameter, float]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = encoder_params if ".bert." in name else head_params
        decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        group.append((param, decay))

    optimizer_groups: List[Dict[str, Any]] = []
    for params, lr in ((encoder_params, encoder_lr), (head_params, head_lr)):
        decay_params = [p for p, decay in params if decay > 0]
        nodecay_params = [p for p, decay in params if decay == 0]
        if decay_params:
            optimizer_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
        if nodecay_params:
            optimizer_groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(optimizer_groups)


# ============================================================================
# 5. 评估指标
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


def multitask_metrics(targets: Dict[str, List[int]], preds: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    return {task: classification_report_dict(targets[task], preds[task]) for task in ("law", "accu", "term")}


def average_macro_f1(result: Dict[str, Dict[str, float]]) -> float:
    return float(np.mean([result[task]["macro_f1"] for task in ("law", "accu", "term")]))


# ============================================================================
# 6. 路径解析、训练与评估
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SEMDR BERT two-tower backbone for CAIL-small.")

    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="包含 train/valid/test 与标签文件的数据目录。")
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL, help="本地 BERT 预训练模型目录或 HuggingFace 模型名。")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="checkpoint 与日志输出目录。")

    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE, help="训练文件名或绝对路径；默认 train_cs_bert_small.json。")
    parser.add_argument("--valid-file", type=str, default=DEFAULT_VALID_FILE, help="验证文件名或绝对路径；默认 valid_cs_bert_small.json。")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE, help="测试文件名或绝对路径；默认 test_cs_bert_small.json。")
    # === 新版标签来源：pkl + csv（优先） ===
    parser.add_argument("--mapping-pkl", type=str, default=DEFAULT_MAPPING_PKL,
                        help="映射表 pkl；优先读其中按 raw id 对齐的释义。默认 mappings_lowfreq_fixed.pkl。")
    parser.add_argument("--law-csv", type=str, default=DEFAULT_LAW_CSV, help="法条释义 csv（law,def）。")
    parser.add_argument("--accu-csv", type=str, default=DEFAULT_ACCU_CSV, help="罪名释义 csv（charge,def）。")
    parser.add_argument("--term-csv", type=str, default=DEFAULT_TERM_CSV, help="刑期释义 csv（term,def）。")
    # === 旧版 txt（仅向后兼容，默认不再使用） ===
    parser.add_argument("--law-label-file", type=str, default=DEFAULT_LAW_LABEL_FILE, help="（已弃用）旧的法条标签 txt。")
    parser.add_argument("--accu-label-file", type=str, default=DEFAULT_ACCU_LABEL_FILE, help="（已弃用）旧的罪名标签 txt。")
    parser.add_argument("--term-label-file", type=str, default=DEFAULT_TERM_LABEL_FILE, help="（已弃用）旧的刑期标签 txt。")

    # low_small 默认类别数（来自 mappings_lowfreq.pkl：law 0..56、accu 0..64、term 0..10，含空洞占位）。
    parser.add_argument("--num-law", type=int, default=57, help="法条类别数；low_small 默认 57（ID 0..56）。")
    parser.add_argument("--num-accu", type=int, default=65, help="罪名类别数；low_small 默认 65（ID 0..64）。")
    parser.add_argument("--num-term", type=int, default=11, help="刑期类别数；low_small 默认 11（ID 0..10，含无罪）。")

    # low_small 数据字段：fact_cut(空格分词文本) / law / accu / term。
    parser.add_argument("--fact-field", type=str, default="fact_cut", help="案件事实字段名；low_small 默认 fact_cut。")
    parser.add_argument("--law-field", type=str, default="law", help="法条标签字段名；默认 law。")
    parser.add_argument("--accu-field", type=str, default="accu", help="罪名标签字段名；默认 accu。")
    parser.add_argument("--term-field", type=str, default="term", help="刑期标签字段名；默认 term（0..10，非 term_cate/time）。")
    parser.add_argument("--text-is-pretokenized", action="store_true", help="强制把事实字符串按空格分词 token 处理。")
    parser.add_argument(
        "--fact-input-format",
        type=str,
        choices=["auto", "bert_ids", "text", "pretokenized_words", "pretokenized_wordpieces"],
        default="pretokenized_words",
        help=(
            "案件事实输入格式。low_small 的 fact_cut 是空格分词的中文词序列，"
            "默认用 pretokenized_words（交由 BERT tokenizer 继续切 wordpiece）；"
            "若改用 LegalDuet 的 BERT id 可用 auto/bert_ids。"
        ),
    )
    # 注：argparse.BooleanOptionalAction 仅 Python>=3.9 可用；为兼容服务器上较老的
    # Python（3.7/3.8），这里改用一对互斥的 store_true/store_false 开关。
    # 默认为 True（加 [CLS]/[SEP]）；传 --no-add-special-tokens-for-pretokenized 可关闭。
    parser.add_argument(
        "--add-special-tokens-for-pretokenized",
        dest="add_special_tokens_for_pretokenized",
        action="store_true",
        default=True,
        help="为 fact_cut/预分词输入加入 [CLS]/[SEP]（默认开启）。",
    )
    parser.add_argument(
        "--no-add-special-tokens-for-pretokenized",
        dest="add_special_tokens_for_pretokenized",
        action="store_false",
        help="不为 fact_cut/预分词输入加入 [CLS]/[SEP]。",
    )

    parser.add_argument("--max-length", "--max-token", dest="max_length", type=int, default=300, help="案件事实最大 token 数，可调。")
    parser.add_argument("--max-label-length", type=int, default=256, help="标签文本最大 token 数。")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--similarity", type=str, choices=["dot", "cosine"], default="dot")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--share-encoder", action="store_true", help="事实塔和标签塔共享同一个 BERT，以降低显存。")

    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=16, help="训练 epoch 数，可调。")
    parser.add_argument("--batch-size", type=int, default=32, help="训练 batch size，可调。")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--law-loss-weight", type=float, default=1.0)
    parser.add_argument("--accu-loss-weight", type=float, default=1.0)
    parser.add_argument("--term-loss-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fp16", action="store_true", help="启用混合精度训练。")
    # （已废弃）保留此参数仅为兼容旧命令；现在无论是否传入，都只保存 best_model.pt。
    parser.add_argument("--save-every-epoch", action="store_true", help="（已废弃、不再生效）早期用于每 epoch 保存 checkpoint。")
    parser.add_argument("--do-test", action="store_true", help="训练结束后用 best_model 在 test 集上评估。")
    parser.add_argument(
        "--best-ckpt-name",
        type=str,
        default="finetuned_big_model_best.pth.tar",
        help="最佳 checkpoint 的保存文件名，默认 finetuned_big_model_best.pth.tar。",
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


def find_file(base_dir: Path, explicit: Optional[str], candidates: Sequence[str], required: bool = True) -> Optional[Path]:
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

    # 兼容数据目录中还有一层 cail_small_standard/data_processed 等子目录的情况。
    for name in candidates:
        matches = sorted(base_dir.rglob(name)) if base_dir.exists() else []
        if matches:
            return matches[0]

    if required:
        raise FileNotFoundError(
            f"在 {base_dir} 下没有找到候选文件：{', '.join(candidates)}。"
            "请用 --train-file/--valid-file/--test-file 显式指定。"
        )
    return None


def count_label_file(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _max_existing(*values: Optional[int]) -> Optional[int]:
    present = [int(v) for v in values if v is not None]
    return max(present) if present else None


def infer_num_classes(
    args: argparse.Namespace,
    data_paths: List[Path],
    fields: FieldConfig,
    law_label_path: Optional[Path],
    accu_label_path: Optional[Path],
    term_label_path: Optional[Path],
) -> Dict[str, int]:
    scanned = scan_num_classes(data_paths, fields=fields)
    law_label_count = count_label_file(law_label_path)
    accu_label_count = count_label_file(accu_label_path)
    term_label_count = count_label_file(term_label_path)
    return {
        "law": args.num_law or _max_existing(law_label_count, scanned.get("law")),
        "accu": args.num_accu or _max_existing(accu_label_count, scanned.get("accu")),
        "term": args.num_term or _max_existing(term_label_count, scanned.get("term")),
    }


def validate_label_space(data_paths: List[Path], fields: FieldConfig, label_summary: Dict[str, int]) -> None:
    scanned = scan_num_classes(data_paths, fields=fields)
    for task in ("law", "accu", "term"):
        needed = scanned.get(task)
        available = label_summary.get(task)
        if needed is not None and available is not None and needed > available:
            raise ValueError(
                f"{task} 标签空间不足：数据最大 ID 需要至少 {needed} 类，但候选标签只有 {available} 类。"
                f"请检查标签文件或显式设置 --num-{task}。"
            )


def make_loader(dataset: CAILDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_eval(
    model: SEMDRBaseTwoTower,
    loader: DataLoader,
    labels: TokenizedLabelStore,
    device: torch.device,
    loss_weights: Dict[str, float],
    fp16: bool = False,
) -> Dict[str, Any]:
    model.eval()
    preds: Dict[str, List[int]] = {"law": [], "accu": [], "term": []}
    targets: Dict[str, List[int]] = {"law": [], "accu": [], "term": []}
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
            task_logits = {
                "law": output["law_logits"],
                "accu": output["accu_logits"],
                "term": output["term_logits"],
            }
            for task in ("law", "accu", "term"):
                preds[task].extend(torch.argmax(task_logits[task], dim=-1).cpu().tolist())
                targets[task].extend(batch_targets[task].cpu().tolist())

    metric = multitask_metrics(targets, preds)
    return {"loss": total_loss / max(steps, 1), "metrics": metric, "avg_macro_f1": average_macro_f1(metric)}


def save_checkpoint(
    path: Path,
    model: SEMDRBaseTwoTower,
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


def print_path_summary(
    data_dir: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Optional[Path],
    law_label_path: Optional[Path],
    accu_label_path: Optional[Path],
    term_label_path: Optional[Path],
    bert_model: str,
    output_dir: Path,
) -> None:
    print("Resolved paths:")
    print(f"  script     : {Path(__file__).resolve()}")
    print(f"  data_dir   : {data_dir}")
    print(f"  train_file : {train_path}")
    print(f"  valid_file : {valid_path}")
    print(f"  test_file  : {test_path}")
    print(f"  law labels : {law_label_path}")
    print(f"  accu labels: {accu_label_path}")
    print(f"  term labels: {term_label_path}")
    print(f"  bert_model : {bert_model}")
    print(f"  output_dir : {output_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{data_dir}\n"
            "请检查截图中的实际路径，或运行时通过 --data-dir 指定。"
        )

    train_path = find_file(data_dir, args.train_file, TRAIN_FILE_CANDIDATES, required=True)
    valid_path = find_file(data_dir, args.valid_file, VALID_FILE_CANDIDATES, required=True)
    test_path = find_file(data_dir, args.test_file, TEST_FILE_CANDIDATES, required=False)
    law_label_path = find_file(data_dir, args.law_label_file, LAW_LABEL_CANDIDATES, required=False)
    accu_label_path = find_file(data_dir, args.accu_label_file, ACCU_LABEL_CANDIDATES, required=False)
    term_label_path = find_file(data_dir, args.term_label_file, TERM_LABEL_CANDIDATES, required=False)
    # 新版标签来源：映射表 pkl + 释义 csv（同时在 data_dir 与脚本目录处查找）
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
        # 再用递归候选查找
        return find_file(data_dir, name, candidates, required=False)
    mapping_pkl_path = _resolve_aux(args.mapping_pkl, MAPPING_PKL_CANDIDATES)
    law_csv_path = _resolve_aux(args.law_csv, (DEFAULT_LAW_CSV,))
    accu_csv_path = _resolve_aux(args.accu_csv, (DEFAULT_ACCU_CSV,))
    term_csv_path = _resolve_aux(args.term_csv, (DEFAULT_TERM_CSV,))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print_path_summary(
        data_dir=data_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        law_label_path=law_label_path,
        accu_label_path=accu_label_path,
        term_label_path=term_label_path,
        bert_model=args.bert_model,
        output_dir=output_dir,
    )

    fields = FieldConfig(
        fact_field=args.fact_field,
        law_field=args.law_field,
        accu_field=args.accu_field,
        term_field=args.term_field,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_dataset = CAILDataset(
        train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        fields=fields,
        text_is_pretokenized=args.text_is_pretokenized,
        add_special_tokens_for_pretokenized=args.add_special_tokens_for_pretokenized,
        fact_input_format=args.fact_input_format,
    )
    valid_dataset = CAILDataset(
        valid_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        fields=fields,
        text_is_pretokenized=args.text_is_pretokenized,
        add_special_tokens_for_pretokenized=args.add_special_tokens_for_pretokenized,
        fact_input_format=args.fact_input_format,
    )
    test_dataset = None
    if test_path is not None and test_path.exists():
        test_dataset = CAILDataset(
            test_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            fields=fields,
            text_is_pretokenized=args.text_is_pretokenized,
            add_special_tokens_for_pretokenized=args.add_special_tokens_for_pretokenized,
            fact_input_format=args.fact_input_format,
        )

    data_paths = [p for p in [train_path, valid_path, test_path] if p is not None and p.exists()]
    num_classes = infer_num_classes(args, data_paths, fields, law_label_path, accu_label_path, term_label_path)
    if any(value is None for value in num_classes.values()):
        raise ValueError(f"无法推断类别数：{num_classes}。请显式传入 --num-law/--num-accu/--num-term。")
    num_classes = {k: int(v) for k, v in num_classes.items()}

    # 优先从「修正版 pkl + csv」按 raw id 构建标签释义；失败才回退到旧 txt。
    label_store = build_label_store_from_pkl_csv(
        mapping_pkl=mapping_pkl_path,
        law_csv=law_csv_path,
        accu_csv=accu_csv_path,
        term_csv=term_csv_path,
        num_law=num_classes["law"],
        num_accu=num_classes["accu"],
        num_term=num_classes["term"],
    )
    if label_store is not None:
        print(f"  [label source] pkl+csv -> {mapping_pkl_path}")
    else:
        print("  [label source] pkl 不可用，回退到旧 txt 逻辑。")
        label_store = build_label_store(
            num_law=num_classes["law"],
            num_accu=num_classes["accu"],
            num_term=num_classes["term"],
            law_label_file=law_label_path,
            accu_label_file=accu_label_path,
            term_label_file=term_label_path,
        )
    validate_label_space(data_paths, fields, label_store.summary())

    print("Data summary:")
    print(f"  train: {len(train_dataset)} | valid: {len(valid_dataset)} | test: {len(test_dataset) if test_dataset is not None else 0}")
    print(
        "  fields: "
        f"fact={train_dataset.fact_field}, law={train_dataset.law_field}, "
        f"accu={train_dataset.accu_field}, term={train_dataset.term_field}"
    )
    print(f"  labels: {label_store.summary()}")
    print(f"  max_length={args.max_length}, epochs={args.epochs}, batch_size={args.batch_size}")

    device = get_device(args.device)
    labels = TokenizedLabelStore(label_store, tokenizer, max_label_length=args.max_label_length).to(device)

    model = SEMDRBaseTwoTower(
        model_name_or_path=args.bert_model,
        projection_dim=args.projection_dim,
        pooling=args.pooling,
        dropout=args.dropout,
        similarity=args.similarity,
        temperature=args.temperature,
        share_encoder=args.share_encoder,
    ).to(device)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_batch_size = args.eval_batch_size or args.batch_size
    valid_loader = make_loader(valid_dataset, eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, eval_batch_size, shuffle=False, num_workers=args.num_workers) if test_dataset is not None else None

    optimizer = build_optimizer(model, encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / max(args.gradient_accumulation_steps, 1))
    total_training_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    scaler = GradScaler(enabled=args.fp16 and device.type == "cuda")
    loss_weights = {"law": args.law_loss_weight, "accu": args.accu_loss_weight, "term": args.term_loss_weight}

    best_score = -1.0
    history: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
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
        valid_result = run_eval(model, valid_loader, labels, device, loss_weights=loss_weights, fp16=args.fp16)
        score = float(valid_result["avg_macro_f1"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid": valid_result,
            "global_step": global_step,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        history.append(epoch_record)

        print(
            f"\nEpoch {epoch} finished | train_loss={train_loss:.4f} | "
            f"valid_loss={valid_result['loss']:.4f} | avg_macro_f1={score:.4f}"
        )
        for task, result in valid_result["metrics"].items():
            print(
                f"  {task:4s}: acc={result['accuracy']:.4f} "
                f"macro_p={result['macro_precision']:.4f} "
                f"macro_r={result['macro_recall']:.4f} "
                f"macro_f1={result['macro_f1']:.4f}"
            )

        (output_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        # 按需求：只保存验证集表现最好的单个 checkpoint（best_model.pt），
        # 不再保存每个 epoch 的中间 checkpoint。
        if score > best_score:
            best_score = score
            save_checkpoint(
                output_dir / args.best_ckpt_name,
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                label_store.summary(),
                valid_result,
            )
            print(f"  Saved best checkpoint -> {args.best_ckpt_name} | avg_macro_f1={best_score:.4f}")

    if args.do_test and test_loader is not None:
        best_ckpt = output_dir / args.best_ckpt_name
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        test_result = run_eval(model, test_loader, labels, device, loss_weights=loss_weights, fp16=args.fp16)
        (output_dir / "test_result.json").write_text(json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Test avg_macro_f1={test_result['avg_macro_f1']:.4f}")

    print(f"Training done. Best avg_macro_f1={best_score:.4f}. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
