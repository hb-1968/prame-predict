"""
Train attention-based MIL classifier with optional regularization.

Extends 04_train_mil.py with three lightweight regularization options:
  - Attention entropy regularization (penalizes concentrated attention)
  - Gradient clipping (stabilizes training across variable bag sizes)
  - Label smoothing (reduces overconfidence for downstream routing)

All regularization defaults to disabled (0.0), preserving baseline behavior.

Includes two light-empirical evaluation modes:
  --compare : trains baseline vs. fully-regularized on fold 1 (bundled view)
  --ablation: trains baseline + each regularizer in isolation on fold 1

Both produce side-by-side diagnostic plots — a cheap directional signal
before committing to full 5-fold CV.

Usage:
    # Full 5-fold with the recommended production config (post-ablation heuristic)
    python 04_train_mil_reg.py --model uni --grad-clip 1.0 --label-smoothing 0.05

    # Ablation: isolate each regularizer's effect (use for new datasets)
    python 04_train_mil_reg.py --model uni --ablation --entropy-lambda 1e-3 --grad-clip 1.0 --label-smoothing 0.05

    # Bundled comparison: baseline vs. fully-regularized
    python 04_train_mil_reg.py --model uni --compare --grad-clip 1.0 --label-smoothing 0.05

    # Baseline only (identical to 04_train_mil.py)
    python 04_train_mil_reg.py --model uni

Heuristic (from fold-1 ablation on 100 slides):
    - grad-clip 1.0 : reduces overfitting gap, zero cost function impact. KEEP.
    - label-smoothing 0.05 : only regularizer with positive AUC delta; produces
                             calibrated confidence scores for downstream routing. KEEP.
    - entropy-lambda : inert at gentle coefficients; the unregularized attention
                       already sits near max entropy. DROP from production config.
"""

import argparse
import json
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SlideDataset(Dataset):
    """Loads pre-extracted patch embeddings per slide."""

    def __init__(self, slide_paths, labels):
        self.slide_paths = slide_paths
        self.labels = labels

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        with h5py.File(self.slide_paths[idx], "r") as f:
            features = f["features"][:].astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)


def collate_bags(batch):
    """Variable-size bags — batch size is always 1 for MIL."""
    features, labels = zip(*batch)
    return features[0], labels[0]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttentionMIL(nn.Module):
    """
    Gated Attention-based MIL (Ilse et al. 2018).

    Learns per-patch attention weights via a gated mechanism, aggregates
    the patch embeddings into a fixed-size slide representation, then
    classifies the slide as high or low PRAME.
    """

    def __init__(self, feat_dim, hidden_dim=256, attn_dim=128, dropout=0.25):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention: element-wise product of tanh and sigmoid branches
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attn_dim, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (N, feat_dim) patch embeddings for one slide
        Returns:
            logit: scalar prediction (pre-sigmoid)
            attention: (N,) softmax-normalized attention weights
        """
        h = self.feature_net(x)  # (N, hidden_dim)

        a_V = self.attention_V(h)  # (N, attn_dim)
        a_U = self.attention_U(h)  # (N, attn_dim)
        a = self.attention_w(a_V * a_U)  # (N, 1)
        attention = torch.softmax(a, dim=0).squeeze(-1)  # (N,)

        slide_repr = (attention.unsqueeze(-1) * h).sum(dim=0)  # (hidden_dim,)
        logit = self.classifier(slide_repr).squeeze()  # scalar

        return logit, attention


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def compute_attention_entropy(attention):
    """Shannon entropy of the attention distribution. Higher = more spread."""
    return -(attention * torch.log(attention + 1e-8)).sum()


def train_one_epoch(model, loader, optimizer, criterion, device,
                    entropy_lambda=0.0, grad_clip=0.0, label_smoothing=0.0):
    model.train()
    total_loss = 0
    total_entropy = 0
    preds, truths = [], []

    for features, label in loader:
        features, label = features.to(device), label.to(device)
        true_label = label.item()

        target = label
        if label_smoothing > 0:
            target = label * (1 - 2 * label_smoothing) + label_smoothing

        logit, attention = model(features)
        loss = criterion(logit, target)

        # Attention entropy regularization: maximize entropy (penalize concentration)
        if entropy_lambda > 0:
            entropy = compute_attention_entropy(attention)
            loss = loss - entropy_lambda * entropy
            total_entropy += entropy.item()

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        preds.append(torch.sigmoid(logit).item())
        truths.append(true_label)

    n = len(loader)
    auc = roc_auc_score(truths, preds)
    mean_entropy = total_entropy / n if entropy_lambda > 0 else 0.0
    return total_loss / n, auc, mean_entropy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_entropy = 0
    preds, truths = [], []

    with torch.inference_mode():
        for features, label in loader:
            features, label = features.to(device), label.to(device)

            logit, attention = model(features)
            loss = criterion(logit, label)

            total_loss += loss.item()
            total_entropy += compute_attention_entropy(attention).item()
            preds.append(torch.sigmoid(logit).item())
            truths.append(label.item())

    n = len(loader)
    auc = roc_auc_score(truths, preds)
    acc = accuracy_score(truths, [int(p > 0.5) for p in preds])
    mean_entropy = total_entropy / n
    return total_loss / n, auc, acc, preds, truths, mean_entropy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(fold_results, all_preds, all_labels, val_indices, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    aucs = [r["val_auc"] for r in fold_results]
    n_folds = len(fold_results)

    ax = axes[0]
    ax.bar(range(1, n_folds + 1), aucs, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(aucs), color="red", linestyle="--",
               label=f"Mean: {np.mean(aucs):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title(f"{model_name.upper()} — Validation AUC per Fold")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1]
    pool_labels = all_labels[val_indices]
    pool_preds = all_preds[val_indices]
    fpr, tpr, _ = roc_curve(pool_labels, pool_preds)
    pooled_auc = roc_auc_score(pool_labels, pool_preds)
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"Pooled AUC = {pooled_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name.upper()} — Pooled ROC Curve")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "cv_results.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(all_train_losses, all_val_losses, all_train_aucs, all_val_aucs,
                         model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for fold_i, (tl, vl) in enumerate(zip(all_train_losses, all_val_losses)):
        axes[0].plot(tl, alpha=0.5, label=f"Fold {fold_i+1} train")
        axes[0].plot(vl, alpha=0.5, linestyle="--", label=f"Fold {fold_i+1} val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} — Loss Curves")
    axes[0].legend(fontsize=7)

    for fold_i, (ta, va) in enumerate(zip(all_train_aucs, all_val_aucs)):
        axes[1].plot(ta, alpha=0.5, label=f"Fold {fold_i+1} train")
        axes[1].plot(va, alpha=0.5, linestyle="--", label=f"Fold {fold_i+1} val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title(f"{model_name.upper()} — AUC Curves")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(baseline, regularized, model_name, out_dir):
    """Side-by-side diagnostic plots for --compare mode."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Validation AUC curves
    ax = axes[0, 0]
    ax.plot(baseline["val_aucs"], label="Baseline", color="steelblue", lw=2)
    ax.plot(regularized["val_aucs"], label="Regularized", color="coral", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC Over Training")
    ax.legend()
    ax.set_ylim(0, 1)

    # (0,1) Training loss curves
    ax = axes[0, 1]
    ax.plot(baseline["train_losses"], label="Baseline train", color="steelblue", lw=1.5)
    ax.plot(baseline["val_losses"], label="Baseline val", color="steelblue", lw=1.5, linestyle="--")
    ax.plot(regularized["train_losses"], label="Reg train", color="coral", lw=1.5)
    ax.plot(regularized["val_losses"], label="Reg val", color="coral", lw=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves (solid=train, dashed=val)")
    ax.legend(fontsize=8)

    # (1,0) Train-val AUC gap (overfitting indicator)
    ax = axes[1, 0]
    bl_gap = [t - v for t, v in zip(baseline["train_aucs"], baseline["val_aucs"])]
    rg_gap = [t - v for t, v in zip(regularized["train_aucs"], regularized["val_aucs"])]
    ax.plot(bl_gap, label="Baseline", color="steelblue", lw=2)
    ax.plot(rg_gap, label="Regularized", color="coral", lw=2)
    ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train AUC − Val AUC")
    ax.set_title("Overfitting Gap (lower = less overfitting)")
    ax.legend()

    # (1,1) Attention entropy over training
    ax = axes[1, 1]
    if any(e > 0 for e in regularized["train_entropies"]):
        ax.plot(baseline["val_entropies"], label="Baseline (val)", color="steelblue", lw=2)
        ax.plot(regularized["val_entropies"], label="Regularized (val)", color="coral", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Attention Entropy")
        ax.set_title("Attention Spread (higher = more distributed)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Entropy regularization\nnot enabled",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.set_title("Attention Entropy")

    plt.suptitle(f"{model_name.upper()} — Baseline vs. Regularized (Fold 1)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "compare_baseline_vs_reg.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation(histories, model_name, out_dir):
    """2x2 diagnostic grid for --ablation mode. histories is dict[name] -> history."""
    colors = {
        "baseline": "steelblue",
        "entropy_only": "coral",
        "grad_clip_only": "seagreen",
        "label_smooth_only": "mediumpurple",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Validation AUC curves
    ax = axes[0, 0]
    for name, h in histories.items():
        ax.plot(h["val_aucs"], label=name, color=colors.get(name, "gray"), lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC Over Training")
    ax.legend()
    ax.set_ylim(0, 1)

    # (0,1) Overfitting gap (train AUC - val AUC)
    ax = axes[0, 1]
    for name, h in histories.items():
        gap = [t - v for t, v in zip(h["train_aucs"], h["val_aucs"])]
        ax.plot(gap, label=name, color=colors.get(name, "gray"), lw=2)
    ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train AUC \u2212 Val AUC")
    ax.set_title("Overfitting Gap (lower = less overfitting)")
    ax.legend()

    # (1,0) Attention entropy over training (val)
    ax = axes[1, 0]
    for name, h in histories.items():
        ax.plot(h["val_entropies"], label=name, color=colors.get(name, "gray"), lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Attention Entropy (val)")
    ax.set_title("Attention Spread (higher = more distributed)")
    ax.legend()

    # (1,1) Final metrics bar chart
    ax = axes[1, 1]
    metric_names = ["Val AUC", "Val Acc", "Sensitivity", "Specificity"]
    metric_keys = ["final_val_auc", "final_val_acc",
                   "final_sensitivity", "final_specificity"]
    config_names = list(histories.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(config_names)
    for i, name in enumerate(config_names):
        values = [histories[name][k] for k in metric_keys]
        offset = (i - (len(config_names) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name,
               color=colors.get(name, "gray"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Final Metrics (best epoch)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    plt.suptitle(f"{model_name.upper()} — Regularization Ablation (Fold 1)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Single-fold training (used by both full CV and compare mode)
# ---------------------------------------------------------------------------

def train_fold(model, train_loader, val_loader, optimizer, scheduler, criterion,
               device, epochs, patience, entropy_lambda=0.0, grad_clip=0.0,
               label_smoothing=0.0, verbose=True, fold_label=""):
    """Train one fold, return metrics history and best model state."""
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {
        "train_losses": [], "val_losses": [],
        "train_aucs": [], "val_aucs": [],
        "train_entropies": [], "val_entropies": [],
    }

    for epoch in range(epochs):
        train_loss, train_auc, train_ent = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            entropy_lambda=entropy_lambda, grad_clip=grad_clip,
            label_smoothing=label_smoothing)
        val_loss, val_auc, val_acc, _, _, val_ent = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_aucs"].append(train_auc)
        history["val_aucs"].append(val_auc)
        history["train_entropies"].append(train_ent)
        history["val_entropies"].append(val_ent)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            ent_str = f" Ent: {val_ent:.1f}" if entropy_lambda > 0 else ""
            print(f"  {fold_label}Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} AUC: {train_auc:.3f} | "
                  f"Val Loss: {val_loss:.4f} AUC: {val_auc:.3f} "
                  f"Acc: {val_acc:.3f}{ent_str}")

        if patience_counter >= patience:
            if verbose:
                print(f"  {fold_label}Early stopping at epoch {epoch+1} "
                      f"(best: epoch {best_epoch})")
            break

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc
    return history, best_state


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def run_compare(args, slides, labels, patients, feat_dim, device):
    """Train baseline and regularized on fold 1, produce comparison plots."""
    print("\n" + "=" * 60)
    print("COMPARE MODE: Baseline vs. Regularized (Fold 1)")
    print("=" * 60)
    print(f"  Regularization: entropy_lambda={args.entropy_lambda}, "
          f"grad_clip={args.grad_clip}, label_smoothing={args.label_smoothing}")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.seed)
    train_idx, val_idx = next(iter(
        skf.split(slides, labels, groups=patients)))

    print(f"  Train: {len(train_idx)} slides | Val: {len(val_idx)} slides")

    train_patients = set(patients[train_idx])
    val_patients = set(patients[val_idx])
    assert train_patients.isdisjoint(val_patients), "Patient leakage!"

    train_dataset = SlideDataset(slides[train_idx], labels[train_idx])
    val_dataset = SlideDataset(slides[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=collate_bags, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_bags, num_workers=0)

    results_dir = Path(args.results_dir) / args.model / "compare"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "baseline": {
            "entropy_lambda": 0.0, "grad_clip": 0.0, "label_smoothing": 0.0,
        },
        "regularized": {
            "entropy_lambda": args.entropy_lambda,
            "grad_clip": args.grad_clip,
            "label_smoothing": args.label_smoothing,
        },
    }

    run_results = {}
    for name, cfg in configs.items():
        print(f"\n--- Training: {name} ---")

        # Reset seeds so both configs see the same shuffle order
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = AttentionMIL(feat_dim, args.hidden_dim, args.attn_dim,
                             args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        history, best_state = train_fold(
            model, train_loader, val_loader, optimizer, scheduler, criterion,
            device, args.epochs, args.patience,
            entropy_lambda=cfg["entropy_lambda"],
            grad_clip=cfg["grad_clip"],
            label_smoothing=cfg["label_smoothing"],
            fold_label=f"[{name}] ")

        # Final eval with best model
        model.load_state_dict(best_state)
        val_loss, val_auc, val_acc, val_preds, val_labels, val_ent = evaluate(
            model, val_loader, criterion, device)

        val_binary = [int(p > 0.5) for p in val_preds]
        tn, fp, fn, tp = confusion_matrix(val_labels, val_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        history["final_val_auc"] = val_auc
        history["final_val_acc"] = val_acc
        history["final_sensitivity"] = sensitivity
        history["final_specificity"] = specificity
        history["final_val_entropy"] = val_ent
        run_results[name] = history

        print(f"  Best epoch: {history['best_epoch']} | "
              f"Val AUC: {val_auc:.3f} Acc: {val_acc:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}")
        if val_ent > 0:
            print(f"  Mean attention entropy: {val_ent:.2f}")

    # Print comparison summary
    bl = run_results["baseline"]
    rg = run_results["regularized"]
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':>10} {'Regularized':>12} {'Delta':>10}")
    print(f"{'-'*60}")
    for metric, label in [
        ("final_val_auc", "Val AUC"),
        ("final_val_acc", "Val Accuracy"),
        ("final_sensitivity", "Sensitivity"),
        ("final_specificity", "Specificity"),
        ("best_epoch", "Best Epoch"),
        ("final_val_entropy", "Attn Entropy"),
    ]:
        bv = bl[metric]
        rv = rg[metric]
        delta = rv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<23} {bv:>10.3f} {rv:>12.3f} {sign}{delta:>9.3f}")

    # Save comparison
    plot_comparison(bl, rg, args.model, results_dir)

    comparison = {
        "model": args.model,
        "mode": "compare",
        "seed": args.seed,
        "epochs": args.epochs,
        "regularization": {
            "entropy_lambda": args.entropy_lambda,
            "grad_clip": args.grad_clip,
            "label_smoothing": args.label_smoothing,
        },
        "baseline": {k: v for k, v in bl.items()
                     if k.startswith("final_") or k == "best_epoch"},
        "regularized": {k: v for k, v in rg.items()
                        if k.startswith("final_") or k == "best_epoch"},
    }
    with open(results_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {results_dir}/")
    print(f"  compare_baseline_vs_reg.png — side-by-side diagnostic plots")
    print(f"  comparison.json             — numeric summary")


# ---------------------------------------------------------------------------
# Ablation mode
# ---------------------------------------------------------------------------

def run_ablation(args, slides, labels, patients, feat_dim, device):
    """Train baseline + each regularizer in isolation on fold 1."""
    print("\n" + "=" * 60)
    print("ABLATION MODE: Baseline + each regularizer in isolation (Fold 1)")
    print("=" * 60)
    print(f"  entropy_lambda={args.entropy_lambda}, "
          f"grad_clip={args.grad_clip}, label_smoothing={args.label_smoothing}")

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.seed)
    train_idx, val_idx = next(iter(
        skf.split(slides, labels, groups=patients)))

    print(f"  Train: {len(train_idx)} slides | Val: {len(val_idx)} slides")

    train_patients = set(patients[train_idx])
    val_patients = set(patients[val_idx])
    assert train_patients.isdisjoint(val_patients), "Patient leakage!"

    train_dataset = SlideDataset(slides[train_idx], labels[train_idx])
    val_dataset = SlideDataset(slides[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=collate_bags, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_bags, num_workers=0)

    results_dir = Path(args.results_dir) / args.model / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "baseline": {
            "entropy_lambda": 0.0, "grad_clip": 0.0, "label_smoothing": 0.0,
        },
        "entropy_only": {
            "entropy_lambda": args.entropy_lambda, "grad_clip": 0.0,
            "label_smoothing": 0.0,
        },
        "grad_clip_only": {
            "entropy_lambda": 0.0, "grad_clip": args.grad_clip,
            "label_smoothing": 0.0,
        },
        "label_smooth_only": {
            "entropy_lambda": 0.0, "grad_clip": 0.0,
            "label_smoothing": args.label_smoothing,
        },
    }

    histories = {}
    for name, cfg in configs.items():
        print(f"\n--- Training: {name} ---")
        print(f"    (entropy_lambda={cfg['entropy_lambda']}, "
              f"grad_clip={cfg['grad_clip']}, label_smoothing={cfg['label_smoothing']})")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = AttentionMIL(feat_dim, args.hidden_dim, args.attn_dim,
                             args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        history, best_state = train_fold(
            model, train_loader, val_loader, optimizer, scheduler, criterion,
            device, args.epochs, args.patience,
            entropy_lambda=cfg["entropy_lambda"],
            grad_clip=cfg["grad_clip"],
            label_smoothing=cfg["label_smoothing"],
            fold_label=f"[{name}] ")

        model.load_state_dict(best_state)
        val_loss, val_auc, val_acc, val_preds, val_labels_fold, val_ent = evaluate(
            model, val_loader, criterion, device)

        val_binary = [int(p > 0.5) for p in val_preds]
        tn, fp, fn, tp = confusion_matrix(val_labels_fold, val_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        history["final_val_auc"] = val_auc
        history["final_val_acc"] = val_acc
        history["final_sensitivity"] = sensitivity
        history["final_specificity"] = specificity
        history["final_val_entropy"] = val_ent
        history["config"] = cfg
        histories[name] = history

        print(f"  Best epoch: {history['best_epoch']} | "
              f"Val AUC: {val_auc:.3f} Acc: {val_acc:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}")
        print(f"  Mean attention entropy: {val_ent:.2f}")

    # Summary table with deltas from baseline
    bl = histories["baseline"]
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY (deltas from baseline)")
    print(f"{'='*80}")
    header = f"{'Metric':<18}" + "".join(f"{name:>16}" for name in histories)
    print(header)
    print(f"{'-'*80}")
    for metric, label in [
        ("final_val_auc", "Val AUC"),
        ("final_val_acc", "Val Accuracy"),
        ("final_sensitivity", "Sensitivity"),
        ("final_specificity", "Specificity"),
        ("best_epoch", "Best Epoch"),
        ("final_val_entropy", "Attn Entropy"),
    ]:
        row = f"  {label:<16}"
        for name, h in histories.items():
            val = h[metric]
            if name == "baseline":
                row += f"{val:>16.3f}"
            else:
                delta = val - bl[metric]
                sign = "+" if delta >= 0 else ""
                row += f"  {val:>6.3f} ({sign}{delta:.3f})"
        print(row)

    plot_ablation(histories, args.model, results_dir)

    ablation_out = {
        "model": args.model,
        "mode": "ablation",
        "seed": args.seed,
        "epochs": args.epochs,
        "configs": {
            name: {
                **h["config"],
                "best_epoch": h["best_epoch"],
                "final_val_auc": h["final_val_auc"],
                "final_val_acc": h["final_val_acc"],
                "final_sensitivity": h["final_sensitivity"],
                "final_specificity": h["final_specificity"],
                "final_val_entropy": h["final_val_entropy"],
            }
            for name, h in histories.items()
        },
    }
    with open(results_dir / "ablation.json", "w") as f:
        json.dump(ablation_out, f, indent=2)

    print(f"\nAblation saved to {results_dir}/")
    print(f"  ablation_comparison.png — 2x2 diagnostic grid")
    print(f"  ablation.json           — per-config summary")


# ---------------------------------------------------------------------------
# Full cross-validation
# ---------------------------------------------------------------------------

def run_full_cv(args, slides, labels, patients, feat_dim, device):
    """Standard k-fold CV with optional regularization."""
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.seed)

    results_dir = Path(args.results_dir) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    all_val_preds = np.zeros(len(slides))
    all_val_labels = np.zeros(len(slides))
    val_indices_all = []

    all_train_losses, all_val_losses = [], []
    all_train_aucs, all_val_aucs = [], []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(slides, labels, groups=patients)):

        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{args.folds}")
        print(f"  Train: {len(train_idx)} slides | Val: {len(val_idx)} slides")

        train_patients = set(patients[train_idx])
        val_patients = set(patients[val_idx])
        assert train_patients.isdisjoint(val_patients), "Patient leakage!"

        train_dataset = SlideDataset(slides[train_idx], labels[train_idx])
        val_dataset = SlideDataset(slides[val_idx], labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                  collate_fn=collate_bags, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_bags, num_workers=0)

        model = AttentionMIL(feat_dim, args.hidden_dim, args.attn_dim,
                             args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        history, best_state = train_fold(
            model, train_loader, val_loader, optimizer, scheduler, criterion,
            device, args.epochs, args.patience,
            entropy_lambda=args.entropy_lambda,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing)

        all_train_losses.append(history["train_losses"])
        all_val_losses.append(history["val_losses"])
        all_train_aucs.append(history["train_aucs"])
        all_val_aucs.append(history["val_aucs"])

        # Evaluate best model
        model.load_state_dict(best_state)
        _, val_auc, val_acc, val_preds, val_labels_fold, _ = evaluate(
            model, val_loader, criterion, device)

        val_binary = [int(p > 0.5) for p in val_preds]
        tn, fp, fn, tp = confusion_matrix(val_labels_fold, val_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"  Best epoch: {history['best_epoch']} | "
              f"Val AUC: {val_auc:.3f} Acc: {val_acc:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f} | "
              f"Specificity: {specificity:.3f}")

        fold_results.append({
            "fold": fold + 1,
            "best_epoch": history["best_epoch"],
            "val_auc": val_auc,
            "val_acc": val_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
        })

        for i, idx in enumerate(val_idx):
            all_val_preds[idx] = val_preds[i]
            all_val_labels[idx] = val_labels_fold[i]
        val_indices_all.extend(val_idx.tolist())

        torch.save(best_state, results_dir / f"fold{fold+1}_model.pt")

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"AGGREGATE RESULTS — {args.model.upper()}")
    print(f"{'='*50}")

    aucs = [r["val_auc"] for r in fold_results]
    accs = [r["val_acc"] for r in fold_results]
    sens = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]

    val_idx_arr = np.array(val_indices_all)
    pooled_auc = roc_auc_score(all_val_labels[val_idx_arr],
                               all_val_preds[val_idx_arr])

    print(f"AUC:         {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"Accuracy:    {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"Sensitivity: {np.mean(sens):.3f} +/- {np.std(sens):.3f}")
    print(f"Specificity: {np.mean(specs):.3f} +/- {np.std(specs):.3f}")
    print(f"Pooled AUC:  {pooled_auc:.3f}")

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_dir / "cv_results.csv", index=False)

    summary = {
        "model": args.model,
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "entropy_lambda": args.entropy_lambda,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "num_slides": len(slides),
        "num_patients": len(set(patients)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_sensitivity": float(np.mean(sens)),
        "std_sensitivity": float(np.std(sens)),
        "mean_specificity": float(np.mean(specs)),
        "std_specificity": float(np.std(specs)),
        "pooled_auc": float(pooled_auc),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(fold_results, all_val_preds, all_val_labels,
                 val_idx_arr, args.model, results_dir)
    plot_training_curves(all_train_losses, all_val_losses,
                        all_train_aucs, all_val_aucs,
                        args.model, results_dir)

    print(f"\nResults saved to {results_dir}/")
    print(f"  cv_results.csv    — per-fold metrics")
    print(f"  summary.json      — aggregate metrics + hyperparameters")
    print(f"  cv_results.png    — AUC bar chart + pooled ROC curve")
    print(f"  training_curves.png — loss and AUC curves per fold")
    print(f"  fold*_model.pt    — best model weights per fold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train attention-based MIL for PRAME prediction (with optional regularization)")
    parser.add_argument("--model", required=True, choices=["uni", "conch"],
                        help="Foundation model embeddings to use")
    parser.add_argument("--emb-dir", default="embeddings",
                        help="Root directory for embeddings (default: embeddings)")
    parser.add_argument("--manifest", default="data/expression/slide_manifest.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without val AUC improvement)")
    parser.add_argument("--seed", type=int, default=42)
    # Regularization
    parser.add_argument("--entropy-lambda", type=float, default=0.0,
                        help="Attention entropy regularization coefficient (0=disabled). "
                             "Fold-1 ablation found this inert on PRAME data at gentle "
                             "coefficients; retained for experimentation.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient clipping max norm (0=disabled, try 1.0)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0=disabled, try 0.05)")
    parser.add_argument("--max-slides", type=int, default=0,
                        help="Limit total slides (stratified subsample, 0=use all)")
    # Evaluation modes
    parser.add_argument("--compare", action="store_true",
                        help="Run single-fold baseline vs. fully-regularized comparison instead of full CV")
    parser.add_argument("--ablation", action="store_true",
                        help="Run single-fold ablation: baseline + each regularizer in isolation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load manifest and match to embeddings
    # ------------------------------------------------------------------
    manifest = pd.read_csv(args.manifest)
    emb_dir = Path(args.emb_dir) / args.model
    feat_dim = {"uni": 1024, "conch": 768}[args.model]

    slides, labels, patients = [], [], []
    for _, row in manifest.iterrows():
        h5_path = emb_dir / row["file_name"].replace(".svs", ".h5")
        if h5_path.exists():
            slides.append(str(h5_path))
            labels.append(row["prame_label"])
            patients.append(row["submitter_id"])

    slides = np.array(slides)
    labels = np.array(labels)
    patients = np.array(patients)

    # Stratified subsample if requested
    if args.max_slides and 0 < args.max_slides < len(slides):
        rng = np.random.RandomState(args.seed)
        high_idx = np.where(labels == 1)[0]
        low_idx = np.where(labels == 0)[0]
        n_high = args.max_slides // 2
        n_low = args.max_slides - n_high
        keep = np.concatenate([
            rng.choice(high_idx, min(n_high, len(high_idx)), replace=False),
            rng.choice(low_idx, min(n_low, len(low_idx)), replace=False),
        ])
        slides, labels, patients = slides[keep], labels[keep], patients[keep]
        print(f"Subsampled to {len(slides)} slides (--max-slides {args.max_slides})")

    print(f"Model: {args.model.upper()}")
    print(f"Slides: {len(slides)} (high={labels.sum()}, low={len(labels) - labels.sum()})")
    print(f"Unique patients: {len(set(patients))}")
    print(f"Feature dim: {feat_dim}")

    if args.ablation:
        run_ablation(args, slides, labels, patients, feat_dim, device)
    elif args.compare:
        run_compare(args, slides, labels, patients, feat_dim, device)
    else:
        reg_active = (args.entropy_lambda > 0 or args.grad_clip > 0
                      or args.label_smoothing > 0)
        if reg_active:
            print(f"Regularization: entropy_lambda={args.entropy_lambda}, "
                  f"grad_clip={args.grad_clip}, label_smoothing={args.label_smoothing}")
        run_full_cv(args, slides, labels, patients, feat_dim, device)


if __name__ == "__main__":
    main()
