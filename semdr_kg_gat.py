#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEMDR · 法律判决推理知识图谱构建 + GAT 聚合模块（解耦版）
================================================================

本文件独立于训练脚本 `semdr_tt_ls_fixed.py`，仅依赖 PyTorch，便于维护与替换。
实现学位论文第 3 章公式 (3-10)~(3-13) 描述的「多事实推理」图注意力聚合：

  (3-10) H^L_{I/C/A} = BERT(L_{I/C/A})
         —— 法律判决结果节点（刑期 I / 罪名 C / 法条 A）用 BERT 编码其文本释义初始化。
            在本工程中，这一步由现有标签塔（label tower）完成，直接把其输出
            [num_law/num_accu/num_term, D] 作为标签节点初始表征传入本模块即可。

  (3-11) R_ij = LeakyReLU(ω_ij · [N_i ‖ N_j])
         —— 对任意相连节点 N_i、N_j，拼接表征后做线性变换 + LeakyReLU，得关系注意力分数。

  (3-12) α_ij = exp(R_ij) / Σ_{e∈O(i)} exp(R_ie)
         —— 对 N_i 的一跳邻居 O(i) 做 softmax 归一化。

  (3-13) H̃^L = σ( Σ_{j∈N_i} α_ij · N_j ),  σ = ELU
         —— 用归一化注意力对邻居加权聚合，得到增强后的标签动态表征 H̃^L，
            替代静态 H^L 参与最终判决预测。

设计约定（按需求确认）：
  * 邻居范围：one-hop（一跳）。
  * 图结构：三类判决结果节点（law/accu/term）与案件节点共享同一张图；
    一次聚合同时更新三类判决结果节点。
  * 边：案件节点 ↔ 它的真实 law/accu/term 标签节点（无向，含自环）。
    案件邻居取自当前 batch（在线动态图），与按 batch 训练天然衔接。

典型用法（在训练脚本 forward 中接入，不强制修改原文件）：

    from semdr_kg_gat import LegalKGReasoner

    reasoner = LegalKGReasoner(dim=256, num_heads=4, dropout=0.1)
    # fact_repr: [B, D]; law_repr/accu_repr/term_repr: [N_*, D]
    law_e, accu_e, term_e = reasoner(
        fact_repr, law_repr, accu_repr, term_repr,
        law_targets, accu_targets, term_targets,   # 各 [B]，batch 内每个案件的真实标签 id
    )
    # 用增强后的 law_e/accu_e/term_e 替换原标签向量去算 similarity_logits。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 单层图注意力（严格对应公式 3-11 ~ 3-13）
# ============================================================================


class GraphAttentionLayer(nn.Module):
    """单头图注意力层，按论文公式实现，使用稠密邻接掩码而非 dgl。

    与标准 GAT（Velickovic et al. 2018）一致，但拼接式注意力的写法严格对齐
    式(3-11)：先对每个节点做线性投影 W·h，再用 a=[a_src‖a_dst] 把
    LeakyReLU(aᵀ[Wh_i‖Wh_j]) 拆成 (a_src·Wh_i) + (a_dst·Wh_j) 高效计算。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat_activation: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat_activation = concat_activation

        # W：节点表征线性变换（式 3-11 中的可学习变换）
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # a：注意力向量，等价于式(3-11)的 ω_ij 作用在 [N_i‖N_j] 上
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.a_src)
        nn.init.zeros_(self.a_dst)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        h:        [V, in_dim]   全部节点表征
        adj_mask: [V, V]        邻接掩码，adj_mask[i, j] = True 表示 j ∈ O(i)（i 的一跳邻居）
        return:   [V, out_dim]  聚合更新后的节点表征
        """
        V = h.size(0)
        Wh = self.W(h)  # [V, out_dim]

        # 式(3-11)：R_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j)
        e_src = (Wh * self.a_src).sum(dim=-1, keepdim=True)  # [V, 1] -> 作为 i 行
        e_dst = (Wh * self.a_dst).sum(dim=-1, keepdim=True)  # [V, 1] -> 作为 j 列
        scores = e_src + e_dst.transpose(0, 1)               # [V, V], scores[i, j]
        scores = self.leaky_relu(scores)

        # 式(3-12)：仅在一跳邻居 O(i) 上做 softmax，非邻居置 -inf
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~adj_mask, neg_inf)
        # 处理孤立节点（整行无邻居）：softmax 会得 NaN，这里兜底为全 0 注意力
        has_neighbor = adj_mask.any(dim=-1, keepdim=True)    # [V, 1]
        alpha = torch.softmax(scores, dim=-1)                # [V, V]
        alpha = torch.where(has_neighbor, alpha, torch.zeros_like(alpha))
        alpha = self.dropout(alpha)

        # 式(3-13)：H̃ = σ( Σ_j α_ij · Wh_j )，σ = ELU
        h_prime = alpha @ Wh                                 # [V, out_dim]
        if self.concat_activation:
            h_prime = self.elu(h_prime)
        return h_prime


class MultiHeadGraphAttention(nn.Module):
    """多头 GAT：中间层 head 拼接，输出层 head 取平均（与常见 GAT 设置一致）。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        average_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.average_output = average_output
        # 若取平均，每个 head 输出 out_dim；若拼接，每个 head 输出 out_dim/num_heads
        head_out = out_dim if average_output else out_dim // num_heads
        if not average_output:
            assert out_dim % num_heads == 0, "拼接模式下 out_dim 必须能被 num_heads 整除"
        self.heads = nn.ModuleList([
            GraphAttentionLayer(
                in_dim, head_out, dropout=dropout,
                negative_slope=negative_slope, concat_activation=True,
            )
            for _ in range(num_heads)
        ])

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        outs = [head(h, adj_mask) for head in self.heads]
        if self.average_output:
            return torch.stack(outs, dim=0).mean(dim=0)  # [V, out_dim]
        return torch.cat(outs, dim=-1)                   # [V, out_dim]


# ============================================================================
# 2. 知识图谱构建：案件节点 ↔ 判决结果节点（三类共享一张图）
# ============================================================================


def build_shared_adjacency(
    num_case: int,
    num_law: int,
    num_accu: int,
    num_term: int,
    law_targets: torch.Tensor,
    accu_targets: torch.Tensor,
    term_targets: torch.Tensor,
    add_self_loop: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, dict]:
    """构建「案件 + law + accu + term」共享图的一跳邻接掩码（无向）。

    节点排列顺序（用于切片）：
        [0 : num_case)                         -> 案件节点
        [num_case : num_case+num_law)          -> 法条(law)节点
        [... : ...+num_accu)                   -> 罪名(accu)节点
        [... : ...+num_term)                   -> 刑期(term)节点

    边：每个案件 c 与其真实 law/accu/term 标签节点相连（双向）。
    返回 (adj_mask[V,V] bool, offsets dict)。
    """
    V = num_case + num_law + num_accu + num_term
    law_off = num_case
    accu_off = num_case + num_law
    term_off = num_case + num_law + num_accu

    adj = torch.zeros(V, V, dtype=torch.bool, device=device)

    case_idx = torch.arange(num_case, device=device)
    law_node = law_off + law_targets.to(device)
    accu_node = accu_off + accu_targets.to(device)
    term_node = term_off + term_targets.to(device)

    for lbl_node in (law_node, accu_node, term_node):
        # 案件 -> 标签 与 标签 -> 案件（无向）
        adj[case_idx, lbl_node] = True
        adj[lbl_node, case_idx] = True

    if add_self_loop:
        diag = torch.arange(V, device=device)
        adj[diag, diag] = True

    offsets = {
        "case": (0, num_case),
        "law": (law_off, law_off + num_law),
        "accu": (accu_off, accu_off + num_accu),
        "term": (term_off, term_off + num_term),
    }
    return adj, offsets


# ============================================================================
# 3. 推理器：把 H^F / H^L 喂入 GAT，输出增强后的 H̃^L（三类共同更新）
# ============================================================================


class LegalKGReasoner(nn.Module):
    """法律判决推理图：三类判决结果节点与案件节点共享一张图，做多事实推理聚合。

    输入：
        fact_repr  [B, D]   案件（犯罪事实）节点表征 H^F
        law_repr   [num_law, D]    法条标签节点初始表征 H^L_A（式 3-10 的 BERT 输出）
        accu_repr  [num_accu, D]   罪名标签节点初始表征 H^L_C
        term_repr  [num_term, D]   刑期标签节点初始表征 H^L_I
        *_targets  [B]      batch 内每个案件的真实 law/accu/term 标签 id（用于建边）
    输出：
        增强后的 (law_e, accu_e, term_e)，形状与各自输入一致，可替换静态标签向量。
    """

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 1,
        # num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.residual = residual
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(
                MultiHeadGraphAttention(
                    in_dim=dim, out_dim=dim, num_heads=num_heads,
                    dropout=dropout, negative_slope=negative_slope,
                    average_output=True,
                )
            )
        self.gat_layers = nn.ModuleList(layers)

    def forward(
        self,
        fact_repr: torch.Tensor,
        law_repr: torch.Tensor,
        accu_repr: torch.Tensor,
        term_repr: torch.Tensor,
        law_targets: torch.Tensor,
        accu_targets: torch.Tensor,
        term_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = fact_repr.device
        num_case = fact_repr.size(0)
        num_law, num_accu, num_term = law_repr.size(0), accu_repr.size(0), term_repr.size(0)

        # 拼成统一的节点表征矩阵 H = [案件; law; accu; term]
        h = torch.cat([fact_repr, law_repr, accu_repr, term_repr], dim=0)  # [V, D]

        adj_mask, off = build_shared_adjacency(
            num_case, num_law, num_accu, num_term,
            law_targets, accu_targets, term_targets,
            add_self_loop=True, device=device,
        )

        out = h
        for layer in self.gat_layers:
            updated = layer(out, adj_mask)
            out = out + updated if self.residual else updated  # 残差，稳住训练

        # 切片取回三类判决结果节点的增强表征 H̃^L
        l0, l1 = off["law"]
        a0, a1 = off["accu"]
        t0, t1 = off["term"]
        law_e = out[l0:l1]
        accu_e = out[a0:a1]
        term_e = out[t0:t1]
        return law_e, accu_e, term_e


# ============================================================================
# 4. 自测：随机张量跑通形状与公式正确性
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 256
    NUM_LAW, NUM_ACCU, NUM_TERM = 57, 65, 11

    fact = torch.randn(B, D)
    law = torch.randn(NUM_LAW, D)
    accu = torch.randn(NUM_ACCU, D)
    term = torch.randn(NUM_TERM, D)
    law_t = torch.randint(0, NUM_LAW, (B,))
    accu_t = torch.randint(0, NUM_ACCU, (B,))
    term_t = torch.randint(0, NUM_TERM, (B,))

    reasoner = LegalKGReasoner(dim=D, num_layers=1, num_heads=4, dropout=0.1)
    law_e, accu_e, term_e = reasoner(fact, law, accu, term, law_t, accu_t, term_t)

    print("[shape] law:", tuple(law_e.shape), "accu:", tuple(accu_e.shape), "term:", tuple(term_e.shape))
    assert law_e.shape == law.shape and accu_e.shape == accu.shape and term_e.shape == term.shape

    # 验证注意力归一化：单独测一层单头，检查每个有邻居节点的 α 行和为 1
    layer = GraphAttentionLayer(D, D)
    adj, _ = build_shared_adjacency(B, NUM_LAW, NUM_ACCU, NUM_TERM, law_t, accu_t, term_t)
    Wh = layer.W(torch.cat([fact, law, accu, term], 0))
    e = layer.leaky_relu((Wh * layer.a_src).sum(-1, keepdim=True) + (Wh * layer.a_dst).sum(-1, keepdim=True).t())
    e = e.masked_fill(~adj, torch.finfo(e.dtype).min)
    alpha = torch.softmax(e, -1)
    row_sums = alpha.sum(-1)
    print("[attn] min/max row-sum over nodes with neighbors:",
          float(row_sums.min()), float(row_sums.max()))

    # 反向传播可用性
    loss = law_e.sum() + accu_e.sum() + term_e.sum()
    loss.backward()
    print("[backward] OK, grad on W:", reasoner.gat_layers[0].heads[0].W.weight.grad is not None)
    print("ALL_OK")
