import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    """
    Traditional Matrix Factorization baseline.
    Input:
        user_idx, item_idx
    Output:
        predicted overall rating
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        global_mean: float = 3.5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        interaction = torch.sum(user_vec * item_vec, dim=1)

        user_b = self.user_bias(user_idx).squeeze(-1)
        item_b = self.item_bias(item_idx).squeeze(-1)

        pred = self.global_bias + user_b + item_b + interaction

        # pred = torch.clamp(pred, self.min_rating, self.max_rating)

        return pred


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering baseline.

    Input:
        user_idx, item_idx

    Output:
        predicted overall rating
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims=None,
        dropout: float = 0.2,
        global_mean: float = 3.5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))

        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        x = torch.cat([user_vec, item_vec], dim=1)

        pred = self.mlp(x).squeeze(-1)

        user_b = self.user_bias(user_idx).squeeze(-1)
        item_b = self.item_bias(item_idx).squeeze(-1)

        pred = self.global_bias + user_b + item_b + pred
        # pred = torch.clamp(pred, self.min_rating, self.max_rating)

        return pred


class MultiCriteriaMatrixFactorizationBaseline(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_criteria=5,
        embedding_dim=64,
        global_mean=3.5,
        min_rating=1.0,
        max_rating=5.0,
    ):
        super().__init__()

        self.n_criteria = n_criteria
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(n_users, embedding_dim) for _ in range(n_criteria)]
        )

        self.item_embeddings = nn.ModuleList(
            [nn.Embedding(n_items, embedding_dim) for _ in range(n_criteria)]
        )

        self.user_biases = nn.ModuleList(
            [nn.Embedding(n_users, 1) for _ in range(n_criteria)]
        )

        self.item_biases = nn.ModuleList(
            [nn.Embedding(n_items, 1) for _ in range(n_criteria)]
        )

        self.criteria_bias = nn.Parameter(torch.ones(n_criteria) * global_mean)

        self.overall_head = nn.Linear(n_criteria, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))

        self._init_weights()

    def _init_weights(self):
        for emb in self.user_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        for emb in self.item_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        for b in self.user_biases:
            nn.init.zeros_(b.weight)

        for b in self.item_biases:
            nn.init.zeros_(b.weight)

    def forward(self, user_idx, item_idx):
        criteria_preds = []

        for c in range(self.n_criteria):
            u = self.user_embeddings[c](user_idx)
            i = self.item_embeddings[c](item_idx)

            dot = torch.sum(u * i, dim=1)

            ub = self.user_biases[c](user_idx).squeeze(-1)
            ib = self.item_biases[c](item_idx).squeeze(-1)

            pred_c = dot + ub + ib + self.criteria_bias[c]
            criteria_preds.append(pred_c)

        criteria_preds = torch.stack(criteria_preds, dim=1)
        # #criteria_preds = torch.clamp(
        #     criteria_preds,
        #     self.min_rating,
        #     self.max_rating,
        # )

        overall = self.overall_head(criteria_preds).squeeze(-1)
        overall = overall + self.global_bias
        # overall = torch.clamp(overall, self.min_rating, self.max_rating)

        return {
            "overall": overall,
            "criteria": criteria_preds,
        }


class MaskedMultiCriteriaLoss(nn.Module):
    def __init__(
        self,
        lambda_overall=1.0,
        lambda_criteria=0.3,
    ):
        super().__init__()
        self.lambda_overall = lambda_overall
        self.lambda_criteria = lambda_criteria

    def forward(
        self,
        outputs,
        rating,
        criteria_targets,
        criteria_masks,
    ):
        overall_pred = outputs["overall"]
        criteria_pred = outputs["criteria"]

        overall_loss = F.mse_loss(overall_pred, rating)

        squared_error = (criteria_pred - criteria_targets) ** 2
        masked_error = squared_error * criteria_masks

        denom = criteria_masks.sum().clamp(min=1.0)
        criteria_loss = masked_error.sum() / denom

        total_loss = (
            self.lambda_overall * overall_loss + self.lambda_criteria * criteria_loss
        )

        return total_loss, {
            "total": float(total_loss.detach().cpu()),
            "overall": float(overall_loss.detach().cpu()),
            "criteria": float(criteria_loss.detach().cpu()),
        }


class DeepFMRecommender(nn.Module):
    """
    DeepFM baseline for rating prediction.

    Inputs:
        user_idx, item_idx, dense_features

    dense_features gồm:
        item_quality, item_value, item_design, item_usability, item_durability
        mask_quality, mask_value, mask_design, mask_usability, mask_durability
        support_quality, support_value, support_design, support_usability, support_durability
        item_support
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dense_dim: int = 16,
        embedding_dim: int = 32,
        hidden_dims=None,
        dropout: float = 0.2,
        global_mean: float = 3.5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_linear = nn.Embedding(n_users, 1)
        self.item_linear = nn.Embedding(n_items, 1)
        self.dense_linear = nn.Linear(dense_dim, 1)

        self.user_fm = nn.Embedding(n_users, embedding_dim)
        self.item_fm = nn.Embedding(n_items, embedding_dim)
        self.dense_fm = nn.Linear(dense_dim, embedding_dim)

        deep_input_dim = embedding_dim * 3

        layers = []
        input_dim = deep_input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h

        layers.append(nn.Linear(input_dim, 1))

        self.deep = nn.Sequential(*layers)

        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.user_fm.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_fm.weight, mean=0.0, std=0.01)

    def forward(self, user_idx, item_idx, dense_features):
        # ---------- Linear part ----------
        linear_out = (
            self.user_linear(user_idx).squeeze(-1)
            + self.item_linear(item_idx).squeeze(-1)
            + self.dense_linear(dense_features).squeeze(-1)
        )

        # ---------- FM part ----------
        user_emb = self.user_fm(user_idx)
        item_emb = self.item_fm(item_idx)
        dense_emb = self.dense_fm(dense_features)

        fm_stack = torch.stack([user_emb, item_emb, dense_emb], dim=1)

        square_of_sum = torch.sum(fm_stack, dim=1) ** 2
        sum_of_square = torch.sum(fm_stack**2, dim=1)

        fm_out = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)

        # ---------- Deep part ----------
        deep_input = torch.cat([user_emb, item_emb, dense_emb], dim=1)
        deep_out = self.deep(deep_input).squeeze(-1)

        pred = self.global_bias + linear_out + fm_out + deep_out
        # pred = torch.clamp(pred, self.min_rating, self.max_rating)

        return pred


class MCDCF(nn.Module):
    """
    Proposed Model: NCF-enhanced Multi-Criteria Deep Collaborative Filtering.

    Components:
    - shared NCF branch for direct overall user-item interaction
    - criterion-specific user/item embeddings
    - criterion-specific deep interaction networks
    - attention-based criteria fusion
    - multi-task output: criteria + overall
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_criteria: int = 5,
        embedding_dim: int = 64,
        hidden_dims=None,
        ncf_hidden_dims=None,
        attention_dim: int = 64,
        dropout: float = 0.3,
        global_mean: float = 3.5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        if ncf_hidden_dims is None:
            ncf_hidden_dims = [128, 64, 32]

        self.n_criteria = n_criteria
        self.embedding_dim = embedding_dim
        self.min_rating = min_rating
        self.max_rating = max_rating

        # ======================================================
        # Direct NCF overall branch
        # ======================================================
        self.ncf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.ncf_item_embedding = nn.Embedding(n_items, embedding_dim)

        ncf_layers = []
        input_dim = embedding_dim * 2

        for h in ncf_hidden_dims:
            ncf_layers.append(nn.Linear(input_dim, h))
            ncf_layers.append(nn.ReLU())
            ncf_layers.append(nn.Dropout(dropout))
            input_dim = h

        self.ncf_mlp = nn.Sequential(*ncf_layers)
        self.ncf_output_dim = ncf_hidden_dims[-1]

        # ======================================================
        # Criteria-specific branches
        # ======================================================
        self.user_embeddings = nn.ModuleList(
            [nn.Embedding(n_users, embedding_dim) for _ in range(n_criteria)]
        )

        self.item_embeddings = nn.ModuleList(
            [nn.Embedding(n_items, embedding_dim) for _ in range(n_criteria)]
        )

        self.criterion_networks = nn.ModuleList()

        for _ in range(n_criteria):
            layers = []
            input_dim = embedding_dim * 2

            for h in hidden_dims:
                layers.append(nn.Linear(input_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = h

            self.criterion_networks.append(nn.Sequential(*layers))

        self.criteria_heads = nn.ModuleList(
            [nn.Linear(hidden_dims[-1], 1) for _ in range(n_criteria)]
        )

        # ======================================================
        # Attention-based criteria fusion
        # ======================================================
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        # ======================================================
        # Final overall prediction
        # ======================================================
        overall_input_dim = self.ncf_output_dim + hidden_dims[-1] + n_criteria

        self.overall_head = nn.Sequential(
            nn.Linear(overall_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.ncf_user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.ncf_item_embedding.weight, mean=0.0, std=0.01)

        for emb in self.user_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

        for emb in self.item_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def _rating_scale(self, x):
        return self.min_rating + (self.max_rating - self.min_rating) * torch.sigmoid(x)

    def forward(self, user_idx, item_idx):
        # ======================================================
        # Direct NCF branch
        # ======================================================
        ncf_user = self.ncf_user_embedding(user_idx)
        ncf_item = self.ncf_item_embedding(item_idx)

        ncf_input = torch.cat([ncf_user, ncf_item], dim=1)
        ncf_hidden = self.ncf_mlp(ncf_input)

        # ======================================================
        # Criteria-specific branches
        # ======================================================
        criterion_hidden = []
        criterion_scores = []

        for c in range(self.n_criteria):
            u = self.user_embeddings[c](user_idx)
            i = self.item_embeddings[c](item_idx)

            x = torch.cat([u, i], dim=1)
            h = self.criterion_networks[c](x)

            raw_score = self.criteria_heads[c](h).squeeze(-1)
            score = self._rating_scale(raw_score)

            criterion_hidden.append(h)
            criterion_scores.append(score)

        hidden_stack = torch.stack(criterion_hidden, dim=1)
        criteria_scores = torch.stack(criterion_scores, dim=1)

        # ======================================================
        # Attention fusion
        # ======================================================
        attn_logits = self.attention_layer(hidden_stack).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)

        fused_hidden = torch.sum(
            hidden_stack * attn_weights.unsqueeze(-1),
            dim=1,
        )

        # ======================================================
        # Final overall prediction
        # ======================================================
        overall_input = torch.cat(
            [
                ncf_hidden,
                fused_hidden,
                criteria_scores,
            ],
            dim=1,
        )

        raw_overall = self.overall_head(overall_input).squeeze(-1)
        overall = self._rating_scale(raw_overall)

        return {
            "overall": overall,
            "criteria": criteria_scores,
            "attention": attn_weights,
        }


class MCDCFLoss(nn.Module):
    def __init__(
        self,
        lambda_overall: float = 1.0,
        lambda_criteria: float = 0.3,
    ):
        super().__init__()
        self.lambda_overall = lambda_overall
        self.lambda_criteria = lambda_criteria

    def forward(
        self,
        outputs,
        rating,
        criteria_targets,
        criteria_masks,
    ):
        overall_pred = outputs["overall"]
        criteria_pred = outputs["criteria"]

        overall_loss = F.mse_loss(overall_pred, rating)

        criteria_se = (criteria_pred - criteria_targets) ** 2
        criteria_se = criteria_se * criteria_masks

        denom = criteria_masks.sum().clamp(min=1.0)
        criteria_loss = criteria_se.sum() / denom

        total_loss = (
            self.lambda_overall * overall_loss + self.lambda_criteria * criteria_loss
        )

        return total_loss, {
            "total": float(total_loss.detach().cpu()),
            "overall": float(overall_loss.detach().cpu()),
            "criteria": float(criteria_loss.detach().cpu()),
        }


class CriteriaResidualNeuMF(nn.Module):
    """
    Proposed v4: User-Conditioned Criteria-Residual NeuMF.

    - GMF branch: memorization
    - MLP branch: nonlinear user-item interaction
    - Criteria encoder: item criteria features
    - User-conditioned criteria weighting: user-specific criteria preference
    - Residual criteria signal: small additive correction
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dense_dim: int = 16,
        n_criteria: int = 5,
        embedding_dim: int = 32,
        gmf_dim: int = 32,
        mlp_hidden_dims=None,
        criteria_hidden_dim: int = 32,
        dropout: float = 0.4,
        global_mean: float = 3.5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
        alpha_init: float = 0.01,
    ):
        super().__init__()

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [128, 64]

        self.n_criteria = n_criteria
        self.min_rating = min_rating
        self.max_rating = max_rating

        # GMF branch
        self.gmf_user_embedding = nn.Embedding(n_users, gmf_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, gmf_dim)

        # MLP branch
        self.mlp_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, embedding_dim)

        mlp_layers = []
        input_dim = embedding_dim * 2

        for h in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = h

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp_output_dim = mlp_hidden_dims[-1]

        # User-conditioned criteria preference
        self.user_criteria_pref = nn.Sequential(
            nn.Linear(embedding_dim, criteria_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(criteria_hidden_dim, n_criteria),
        )

        # Criteria encoder
        self.criteria_encoder = nn.Sequential(
            nn.Linear(dense_dim, criteria_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(criteria_hidden_dim, self.mlp_output_dim),
            nn.ReLU(),
        )

        self.criteria_projection = nn.Linear(
            self.mlp_output_dim,
            self.mlp_output_dim,
        )

        # Bounded alpha: max influence = 0.02
        self.criteria_alpha_raw = nn.Parameter(torch.tensor(-4.0))

        # Auxiliary criteria head
        self.criteria_head = nn.Sequential(
            nn.Linear(self.mlp_output_dim, criteria_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(criteria_hidden_dim, n_criteria),
        )

        # Bias
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))

        # Final head
        # + 1 vì thêm weighted_criteria_score
        final_input_dim = gmf_dim + self.mlp_output_dim + self.mlp_output_dim + 1

        self.output_head = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def _rating_scale(self, x):
        return self.min_rating + (self.max_rating - self.min_rating) * torch.sigmoid(x)

    def forward(self, user_idx, item_idx, dense_features):
        # GMF
        gmf_user = self.gmf_user_embedding(user_idx)
        gmf_item = self.gmf_item_embedding(item_idx)
        gmf_vector = gmf_user * gmf_item

        # MLP
        mlp_user = self.mlp_user_embedding(user_idx)
        mlp_item = self.mlp_item_embedding(item_idx)

        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_hidden = self.mlp(mlp_input)

        # Criteria features
        criteria_values = dense_features[:, : self.n_criteria]
        criteria_masks = dense_features[:, self.n_criteria : self.n_criteria * 2]

        # criteria_values_norm = (criteria_values - self.min_rating) / (
        #     self.max_rating - self.min_rating
        # )
        criteria_values_norm = (criteria_values + 1.0) / 2.0

        # criteria_values_norm = torch.clamp(criteria_values_norm, 0.0, 1.0)

        # User-conditioned criteria weights
        criteria_logits = self.user_criteria_pref(mlp_user)

        masked_logits = criteria_logits.masked_fill(
            criteria_masks <= 0,
            -1e9,
        )

        weights_masked = torch.softmax(masked_logits, dim=-1)
        weights_unmasked = torch.softmax(criteria_logits, dim=-1)

        has_any_criteria = criteria_masks.sum(dim=1, keepdim=True) > 0

        criteria_weights = torch.where(
            has_any_criteria,
            weights_masked,
            weights_unmasked,
        )

        weighted_criteria_score = (criteria_values_norm * criteria_weights).sum(
            dim=1, keepdim=True
        )

        # Criteria residual
        criteria_hidden = self.criteria_encoder(dense_features)
        criteria_residual = self.criteria_projection(criteria_hidden)

        alpha = 0.02 * torch.sigmoid(self.criteria_alpha_raw)

        enhanced_mlp = mlp_hidden + alpha * criteria_residual

        # Auxiliary criteria prediction
        raw_criteria = self.criteria_head(criteria_hidden)
        criteria_pred = self._rating_scale(raw_criteria)

        # Overall prediction
        final_input = torch.cat(
            [
                gmf_vector,
                enhanced_mlp,
                criteria_hidden,
                weighted_criteria_score,
            ],
            dim=1,
        )

        raw_overall = self.output_head(final_input).squeeze(-1)

        user_b = self.user_bias(user_idx).squeeze(-1)
        item_b = self.item_bias(item_idx).squeeze(-1)

        overall = raw_overall + user_b + item_b + self.global_bias
        # overall = torch.clamp(overall, self.min_rating, self.max_rating)

        return {
            "overall": overall,
            "criteria": criteria_pred,
            "criteria_alpha": alpha,
            "criteria_weights": criteria_weights,
            "weighted_criteria_score": weighted_criteria_score,
        }


class CriteriaResidualNeuMFLoss(nn.Module):
    def __init__(
        self,
        lambda_overall: float = 1.0,
        lambda_criteria: float = 0.01,
    ):
        super().__init__()
        self.lambda_overall = lambda_overall
        self.lambda_criteria = lambda_criteria

    def forward(
        self,
        outputs,
        rating,
        criteria_targets,
        criteria_masks,
    ):
        overall_pred = outputs["overall"]
        criteria_pred = outputs["criteria"]

        overall_loss = F.mse_loss(overall_pred, rating)

        criteria_se = (criteria_pred - criteria_targets) ** 2
        masked_criteria_se = criteria_se * criteria_masks

        denom = criteria_masks.sum().clamp(min=1.0)
        criteria_loss = masked_criteria_se.sum() / denom

        total_loss = (
            self.lambda_overall * overall_loss + self.lambda_criteria * criteria_loss
        )

        return total_loss, {
            "total": float(total_loss.detach().cpu()),
            "overall": float(overall_loss.detach().cpu()),
            "criteria": float(criteria_loss.detach().cpu()),
        }


class BiasOnlyRecommender(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        global_mean: float = 3.5,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ):
        super().__init__()

        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        user_b = self.user_bias(user_idx).squeeze(-1)
        item_b = self.item_bias(item_idx).squeeze(-1)

        pred = self.global_bias + user_b + item_b
        return pred


print("done")
