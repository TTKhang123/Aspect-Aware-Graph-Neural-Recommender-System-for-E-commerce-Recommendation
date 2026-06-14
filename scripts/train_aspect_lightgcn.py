from __future__ import annotations
import copy
import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.aspect_gcn.data import BPRDataset, load_data_bundle
from src.models.aspect_gcn.evaluate import evaluate_split
from src.models.aspect_gcn.graph import build_normalized_adj
from src.models.aspect_gcn.model import (
    LightGCNAspectRecommender,
    bpr_loss,
    embedding_l2_loss,
)
from src.models.aspect_gcn.utils import (
    choose_device,
    ensure_dir,
    load_yaml,
    log,
    save_json,
    set_seed,
)


def run_one_experiment(cfg: dict, ablation_name: str) -> Dict[str, dict]:
    set_seed(int(cfg.get("seed", 42)))

    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    device = choose_device(cfg["training"].get("device", "cuda"))

    log("=" * 80)
    log(f"RUNNING ABLATION: {ablation_name}")
    log("=" * 80)

    log(f"Using device: {device}")
    log("Loading data bundle...")
    bundle = load_data_bundle(cfg)

    log(f"n_users={bundle.n_users:,} | n_items={bundle.n_items:,}")
    log(f"train interactions={len(bundle.train.users):,}")
    log(
        f"item feature dim={bundle.item_features.shape[1]} | "
        f"features={bundle.feature_names}"
    )

    log("Building normalized LightGCN graph...")
    norm_adj = build_normalized_adj(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        users=bundle.train.users,
        items=bundle.train.items,
        device=device,
    )

    item_features_t = torch.tensor(
        bundle.item_features, dtype=torch.float32, device=device
    )
    item_feature_mask_t = torch.tensor(
        bundle.item_feature_mask, dtype=torch.float32, device=device
    )

    model_cfg = cfg["model"]
    model = LightGCNAspectRecommender(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        item_feature_dim=bundle.item_features.shape[1],
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        n_layers=int(model_cfg.get("n_layers", 3)),
        aspect_hidden_dim=int(model_cfg.get("aspect_hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        fusion=model_cfg.get("fusion", "gated_residual"),
        use_user_bias=bool(model_cfg.get("use_user_bias", True)),
        use_item_bias=bool(model_cfg.get("use_item_bias", True)),
    ).to(device)

    # train_dataset = BPRDataset(
    #     users=bundle.train.users,
    #     pos_items=bundle.train.items,
    #     train_user_pos=bundle.train_user_pos,
    #     n_items=bundle.n_items,
    #     num_negatives=int(cfg["training"].get("num_negatives_train", 1)),
    #     seed=int(cfg.get("seed", 42)),
    # )

    train_dataset = BPRDataset(
        users=bundle.train.users,
        pos_items=bundle.train.items,
        train_user_pos=bundle.train_user_pos,
        n_items=bundle.n_items,
        num_negatives=int(cfg["training"].get("num_negatives_train", 1)),
        seed=int(cfg.get("seed", 42)),
        negative_items=bundle.warm_items,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"].get("batch_size", 8192)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )

    k_values = cfg["metrics"].get("top_k", [5, 10, 20])
    max_eval_users = cfg["data"].get("max_eval_users", 2000)
    num_neg_eval = int(cfg["data"].get("num_negatives_eval", 100))
    filter_seen = bool(cfg["data"].get("filter_seen_items", True))

    best_metric = -1.0
    best_path = out_dir / "best_model.pt"
    patience = int(cfg["training"].get("patience", 5))
    bad_epochs = 0
    history = []

    for epoch in range(1, int(cfg["training"].get("epochs", 30)) + 1):
        log(f"========== [{ablation_name}] Epoch {epoch} ==========")

        train_metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            norm_adj=norm_adj,
            item_features_t=item_features_t,
            item_feature_mask_t=item_feature_mask_t,
            device=device,
            cfg=cfg,
        )
        log(f"train: {train_metrics}")

        val_metrics = evaluate_split(
            model=model,
            split=bundle.val,
            bundle=bundle,
            norm_adj=norm_adj,
            item_features_t=item_features_t,
            item_feature_mask_t=item_feature_mask_t,
            device=device,
            k_values=k_values,
            max_eval_users=max_eval_users,
            num_negatives=num_neg_eval,
            filter_seen_items=filter_seen,
            seed=int(cfg.get("seed", 42)) + epoch,
        )
        log(f"val: {val_metrics}")

        monitor = float(val_metrics.get("ndcg@10", val_metrics.get("recall@10", 0.0)))
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        save_json({"history": history}, out_dir / "history.json")

        if monitor > best_metric:
            best_metric = monitor
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "ablation_name": ablation_name,
                    "n_users": bundle.n_users,
                    "n_items": bundle.n_items,
                    "feature_names": bundle.feature_names,
                    "best_val_metric": best_metric,
                },
                best_path,
            )
            log(f"Saved best model: {best_path} | monitor={best_metric:.6f}")
        else:
            bad_epochs += 1
            log(f"No improvement. bad_epochs={bad_epochs}/{patience}")
            if bad_epochs >= patience:
                log("Early stopping triggered.")
                break

    log("Loading best model for final test...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    final = {}

    final["val_warm"] = evaluate_split(
        model,
        bundle.val,
        bundle,
        norm_adj,
        item_features_t,
        item_feature_mask_t,
        device,
        k_values,
        max_eval_users,
        num_neg_eval,
        filter_seen,
        seed=int(cfg.get("seed", 42)) + 1000,
    )

    final["test_warm"] = evaluate_split(
        model,
        bundle.test,
        bundle,
        norm_adj,
        item_features_t,
        item_feature_mask_t,
        device,
        k_values,
        max_eval_users,
        num_neg_eval,
        filter_seen,
        seed=int(cfg.get("seed", 42)) + 2000,
    )

    if bundle.val_full is not None:
        final["val_full"] = evaluate_split(
            model,
            bundle.val_full,
            bundle,
            norm_adj,
            item_features_t,
            item_feature_mask_t,
            device,
            k_values,
            max_eval_users,
            num_neg_eval,
            filter_seen,
            seed=int(cfg.get("seed", 42)) + 3000,
        )

    if bundle.test_full is not None:
        final["test_full"] = evaluate_split(
            model,
            bundle.test_full,
            bundle,
            norm_adj,
            item_features_t,
            item_feature_mask_t,
            device,
            k_values,
            max_eval_users,
            num_neg_eval,
            filter_seen,
            seed=int(cfg.get("seed", 42)) + 4000,
        )

    save_json(final, out_dir / "metrics.json")
    log(f"Final metrics saved to {out_dir / 'metrics.json'}")

    return final


def train_one_epoch(
    model: LightGCNAspectRecommender,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm_adj: torch.Tensor,
    item_features_t: torch.Tensor,
    item_feature_mask_t: torch.Tensor,
    device: torch.device,
    cfg: dict,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bpr = 0.0
    total_l2 = 0.0
    n_batches = 0

    l2_weight = float(cfg["training"].get("l2_reg_weight", 0.0))
    grad_clip = cfg["training"].get("grad_clip_norm")

    for users, pos, neg in tqdm(loader, desc="train"):
        users = users.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        neg = neg.to(device, non_blocking=True)

        user_all, item_cf_all = model.computer(norm_adj)
        item_sem_all = model.encode_item_features(item_features_t)

        pos_scores = model.score_pairs_from_embeddings(
            users, pos, user_all, item_cf_all, item_sem_all, item_feature_mask_t
        )
        neg_scores = model.score_pairs_from_embeddings(
            users, neg, user_all, item_cf_all, item_sem_all, item_feature_mask_t
        )

        loss_bpr = bpr_loss(pos_scores, neg_scores)
        loss_l2 = embedding_l2_loss(model, users, pos, neg)
        loss = loss_bpr + l2_weight * loss_l2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_bpr += float(loss_bpr.detach().cpu())
        total_l2 += float(loss_l2.detach().cpu())
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "bpr_loss": total_bpr / max(n_batches, 1),
        "l2_loss": total_l2 / max(n_batches, 1),
    }


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="config/config.yaml",
#     )
#     args = parser.parse_args()
#     root_cfg = load_yaml(args.config)
#     cfg = root_cfg["aspect_lightgcn"]

#     set_seed(int(cfg.get("seed", 42)))
#     out_dir = ensure_dir(cfg["paths"]["output_dir"])
#     device = choose_device(cfg["training"].get("device", "cuda"))

#     log(f"Using device: {device}")
#     log("Loading data bundle...")
#     bundle = load_data_bundle(cfg)
#     log(f"n_users={bundle.n_users:,} | n_items={bundle.n_items:,}")
#     log(f"train interactions={len(bundle.train.users):,}")
#     log(
#         f"item feature dim={bundle.item_features.shape[1]} | features={bundle.feature_names}"
#     )

#     log("Building normalized LightGCN graph...")
#     norm_adj = build_normalized_adj(
#         n_users=bundle.n_users,
#         n_items=bundle.n_items,
#         users=bundle.train.users,
#         items=bundle.train.items,
#         device=device,
#     )

#     item_features_t = torch.tensor(
#         bundle.item_features, dtype=torch.float32, device=device
#     )
#     item_feature_mask_t = torch.tensor(
#         bundle.item_feature_mask, dtype=torch.float32, device=device
#     )

#     model_cfg = cfg["model"]
#     model = LightGCNAspectRecommender(
#         n_users=bundle.n_users,
#         n_items=bundle.n_items,
#         item_feature_dim=bundle.item_features.shape[1],
#         embedding_dim=int(model_cfg.get("embedding_dim", 64)),
#         n_layers=int(model_cfg.get("n_layers", 3)),
#         aspect_hidden_dim=int(model_cfg.get("aspect_hidden_dim", 128)),
#         dropout=float(model_cfg.get("dropout", 0.1)),
#         fusion=model_cfg.get("fusion", "gated_residual"),
#         use_user_bias=bool(model_cfg.get("use_user_bias", True)),
#         use_item_bias=bool(model_cfg.get("use_item_bias", True)),
#     ).to(device)

#     train_dataset = BPRDataset(
#         users=bundle.train.users,
#         pos_items=bundle.train.items,
#         train_user_pos=bundle.train_user_pos,
#         n_items=bundle.n_items,
#         num_negatives=int(cfg["training"].get("num_negatives_train", 1)),
#         seed=int(cfg.get("seed", 42)),
#     )
#     loader = DataLoader(
#         train_dataset,
#         batch_size=int(cfg["training"].get("batch_size", 8192)),
#         shuffle=True,
#         num_workers=int(cfg["training"].get("num_workers", 0)),
#         pin_memory=(device.type == "cuda"),
#     )

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=float(cfg["training"].get("learning_rate", 1e-3)),
#         weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
#     )

#     k_values = cfg["metrics"].get("top_k", [5, 10, 20])
#     max_eval_users = cfg["data"].get("max_eval_users", 2000)
#     num_neg_eval = int(cfg["data"].get("num_negatives_eval", 100))
#     filter_seen = bool(cfg["data"].get("filter_seen_items", True))

#     best_metric = -1.0
#     best_path = out_dir / "best_model.pt"
#     patience = int(cfg["training"].get("patience", 5))
#     bad_epochs = 0
#     history = []

#     for epoch in range(1, int(cfg["training"].get("epochs", 30)) + 1):
#         log(f"========== Epoch {epoch} ==========")
#         train_metrics = train_one_epoch(
#             model=model,
#             loader=loader,
#             optimizer=optimizer,
#             norm_adj=norm_adj,
#             item_features_t=item_features_t,
#             item_feature_mask_t=item_feature_mask_t,
#             device=device,
#             cfg=cfg,
#         )
#         log(f"train: {train_metrics}")

#         val_metrics = evaluate_split(
#             model=model,
#             split=bundle.val,
#             bundle=bundle,
#             norm_adj=norm_adj,
#             item_features_t=item_features_t,
#             item_feature_mask_t=item_feature_mask_t,
#             device=device,
#             k_values=k_values,
#             max_eval_users=max_eval_users,
#             num_negatives=num_neg_eval,
#             filter_seen_items=filter_seen,
#             seed=int(cfg.get("seed", 42)) + epoch,
#         )
#         log(f"val: {val_metrics}")

#         monitor = float(val_metrics.get("ndcg@10", val_metrics.get("recall@10", 0.0)))
#         history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
#         save_json({"history": history}, out_dir / "history.json")

#         if monitor > best_metric:
#             best_metric = monitor
#             bad_epochs = 0
#             torch.save(
#                 {
#                     "model_state_dict": model.state_dict(),
#                     "config": cfg,
#                     "n_users": bundle.n_users,
#                     "n_items": bundle.n_items,
#                     "feature_names": bundle.feature_names,
#                     "best_val_metric": best_metric,
#                 },
#                 best_path,
#             )
#             log(f"Saved best model: {best_path} | monitor={best_metric:.6f}")
#         else:
#             bad_epochs += 1
#             log(f"No improvement. bad_epochs={bad_epochs}/{patience}")
#             if bad_epochs >= patience:
#                 log("Early stopping triggered.")
#                 break

#     log("Loading best model for final test...")
#     ckpt = torch.load(best_path, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])

#     final = {}
#     final["val_warm"] = evaluate_split(
#         model,
#         bundle.val,
#         bundle,
#         norm_adj,
#         item_features_t,
#         item_feature_mask_t,
#         device,
#         k_values,
#         max_eval_users,
#         num_neg_eval,
#         filter_seen,
#         seed=int(cfg.get("seed", 42)) + 1000,
#     )
#     final["test_warm"] = evaluate_split(
#         model,
#         bundle.test,
#         bundle,
#         norm_adj,
#         item_features_t,
#         item_feature_mask_t,
#         device,
#         k_values,
#         max_eval_users,
#         num_neg_eval,
#         filter_seen,
#         seed=int(cfg.get("seed", 42)) + 2000,
#     )
#     if bundle.val_full is not None:
#         final["val_full"] = evaluate_split(
#             model,
#             bundle.val_full,
#             bundle,
#             norm_adj,
#             item_features_t,
#             item_feature_mask_t,
#             device,
#             k_values,
#             max_eval_users,
#             num_neg_eval,
#             filter_seen,
#             seed=int(cfg.get("seed", 42)) + 3000,
#         )
#     if bundle.test_full is not None:
#         final["test_full"] = evaluate_split(
#             model,
#             bundle.test_full,
#             bundle,
#             norm_adj,
#             item_features_t,
#             item_feature_mask_t,
#             device,
#             k_values,
#             max_eval_users,
#             num_neg_eval,
#             filter_seen,
#             seed=int(cfg.get("seed", 42)) + 4000,
#         )

#     save_json(final, out_dir / "metrics.json")
#     log(f"Final metrics saved to {out_dir / 'metrics.json'}")
#     log(str(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        # lightgcn_only (done) , lightgcn_aspects (done) , lightgcn_aspects_masks (done), lightgcn_aspects_masks_support
        help="Ablation name to run, or 'all' to run every ablation.",
    )
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    root_cfg = load_yaml(args.config)
    base_cfg = root_cfg["aspect_lightgcn"]

    ablations = base_cfg.get("ablation")

    if not ablations:
        cfg = copy.deepcopy(base_cfg)
        cfg["active_ablation"] = "default"
        result = run_one_experiment(cfg, "default")
        save_json(
            {"default": result}, Path(cfg["paths"]["output_dir"]) / "all_metrics.json"
        )
        return

    if args.ablation == "all":
        selected_ablations = list(ablations.keys())
    else:
        if args.ablation not in ablations:
            raise KeyError(
                f"Unknown ablation '{args.ablation}'. "
                f"Available: {list(ablations.keys())}"
            )
        selected_ablations = [args.ablation]

    all_results = {}

    base_output_dir = Path(base_cfg["paths"]["output_dir"])

    for ablation_name in selected_ablations:
        cfg = copy.deepcopy(base_cfg)

        cfg["active_ablation"] = ablation_name
        cfg["paths"]["output_dir"] = str(base_output_dir / ablation_name)

        result = run_one_experiment(cfg, ablation_name)
        all_results[ablation_name] = result

        save_json(all_results, base_output_dir / "ablation_summary_metrics.json")

    log("=" * 80)
    log("ABLATION SUMMARY")
    log("=" * 80)

    for ablation_name, result in all_results.items():
        test_warm = result.get("test_warm", {})
        val_warm = result.get("val_warm", {})

        log(
            f"{ablation_name} | "
            f"val_ndcg@10={val_warm.get('ndcg@10')} | "
            f"test_ndcg@10={test_warm.get('ndcg@10')} | "
            f"test_recall@10={test_warm.get('recall@10')} | "
            f"test_hit@10={test_warm.get('hit@10')}"
        )


if __name__ == "__main__":
    main()
