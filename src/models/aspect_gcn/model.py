from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNAspectRecommender(nn.Module):
    """
    LightGCN backbone + ABSA item semantic encoder + gated residual fusion.

    LightGCN part follows the core idea of LightGCN: only neighborhood propagation
    over the normalized user-item graph, no nonlinear transformation inside graph layers.

    The semantic branch does not replace collaborative filtering. It adds an item-side
    residual representation learned from ABSA features.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        item_feature_dim: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        aspect_hidden_dim: int = 128,
        dropout: float = 0.1,
        fusion: str = "gated_residual",
        use_user_bias: bool = True,
        use_item_bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.n_layers = int(n_layers)
        self.fusion = fusion
        self.use_user_bias = bool(use_user_bias)
        self.use_item_bias = bool(use_item_bias)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.aspect_encoder = nn.Sequential(
            nn.Linear(item_feature_dim, aspect_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aspect_hidden_dim, embedding_dim),
        )

        self.user_semantic_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        if fusion == "gated_residual":
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 3 + 1, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid(),
            )
        elif fusion == "add":
            self.gate = None
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

        if self.use_user_bias:
            self.user_bias = nn.Embedding(n_users, 1)
        else:
            self.register_parameter("user_bias", None)

        if self.use_item_bias:
            self.item_bias = nn.Embedding(n_items, 1)
        else:
            self.register_parameter("item_bias", None)

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        if self.use_user_bias:
            nn.init.zeros_(self.user_bias.weight)
        if self.use_item_bias:
            nn.init.zeros_(self.item_bias.weight)

    def computer(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return propagated LightGCN user and item embeddings."""
        all_emb = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        users, items = torch.split(final_emb, [self.n_users, self.n_items], dim=0)
        return users, items

    def encode_item_features(self, item_features: torch.Tensor) -> torch.Tensor:
        return self.aspect_encoder(item_features)

    def fuse_item_embedding(
        self,
        user_emb: torch.Tensor,
        item_cf_emb: torch.Tensor,
        item_sem_emb: torch.Tensor,
        item_feature_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if item_feature_mask.ndim == 1:
            item_feature_mask = item_feature_mask.unsqueeze(-1)

        item_sem_emb = item_sem_emb * item_feature_mask

        if self.fusion == "add":
            gate_value = item_feature_mask
            return item_cf_emb + item_sem_emb, gate_value

        gate_input = torch.cat(
            [user_emb, item_cf_emb, item_sem_emb, item_feature_mask], dim=-1
        )
        gate_value = self.gate(gate_input)
        return item_cf_emb + gate_value * item_sem_emb, gate_value

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        norm_adj: torch.Tensor,
        item_features: torch.Tensor,
        item_feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        user_all, item_cf_all = self.computer(norm_adj)
        item_sem_all = self.encode_item_features(item_features)
        return self.score_pairs_from_embeddings(
            users, items, user_all, item_cf_all, item_sem_all, item_feature_mask
        )

    def score_pairs_from_embeddings(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        user_all: torch.Tensor,
        item_cf_all: torch.Tensor,
        item_sem_all: torch.Tensor,
        item_feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        u = user_all[users]
        i_cf = item_cf_all[items]
        i_sem = item_sem_all[items]
        mask = item_feature_mask[items]

        i_final, _ = self.fuse_item_embedding(u, i_cf, i_sem, mask)
        cf_score = (u * i_final).sum(dim=-1)

        # Semantic preference term: lets users interact with aspect representation directly.
        u_sem = self.user_semantic_projection(u)
        sem_score = (u_sem * (i_sem * mask.unsqueeze(-1))).sum(dim=-1)

        score = cf_score + sem_score + self.global_bias
        if self.use_user_bias:
            score = score + self.user_bias(users).squeeze(-1)
        if self.use_item_bias:
            score = score + self.item_bias(items).squeeze(-1)
        return score

    @torch.no_grad()
    def full_sort_scores(
        self,
        users: torch.Tensor,
        candidate_items: torch.Tensor,
        norm_adj: torch.Tensor,
        item_features: torch.Tensor,
        item_feature_mask: torch.Tensor,
        batch_items: int = 50000,
    ) -> torch.Tensor:
        """Score a user batch against candidate items. Used for evaluation."""
        self.eval()
        user_all, item_cf_all = self.computer(norm_adj)
        item_sem_all = self.encode_item_features(item_features)
        all_scores = []
        for start in range(0, len(candidate_items), batch_items):
            cand = candidate_items[start : start + batch_items]
            # Shape: [B, C, D]
            u = user_all[users]
            i_cf = item_cf_all[cand]
            i_sem = item_sem_all[cand]
            mask = item_feature_mask[cand]

            B, C = users.numel(), cand.numel()
            u_exp = u[:, None, :].expand(B, C, -1).reshape(B * C, -1)
            i_cf_exp = i_cf[None, :, :].expand(B, C, -1).reshape(B * C, -1)
            i_sem_exp = i_sem[None, :, :].expand(B, C, -1).reshape(B * C, -1)
            mask_exp = mask[None, :].expand(B, C).reshape(B * C)
            item_final, _ = self.fuse_item_embedding(
                u_exp, i_cf_exp, i_sem_exp, mask_exp
            )
            scores = (u_exp * item_final).sum(dim=-1)
            u_sem = self.user_semantic_projection(u_exp)
            scores = scores + (u_sem * (i_sem_exp * mask_exp.unsqueeze(-1))).sum(dim=-1)
            scores = scores + self.global_bias
            if self.use_user_bias:
                scores = scores + self.user_bias(users).squeeze(-1).repeat_interleave(C)
            if self.use_item_bias:
                scores = scores + self.item_bias(cand).squeeze(-1).repeat(B)
            all_scores.append(scores.view(B, C))
        return torch.cat(all_scores, dim=1)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def embedding_l2_loss(
    model: LightGCNAspectRecommender,
    users: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> torch.Tensor:
    u = model.user_embedding(users).pow(2).sum(dim=1)
    p = model.item_embedding(pos).pow(2).sum(dim=1)
    n = model.item_embedding(neg).pow(2).sum(dim=1)
    return (u + p + n).mean() / 2.0
