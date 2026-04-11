"""
Train attention-based MIL classifier for slide-level PRAME prediction.

Uses pre-extracted patch embeddings (UNI or CONCH) to predict binary
PRAME expression (high vs. low) at the slide level. Patient-level
stratified k-fold cross-validation prevents data leakage from
multi-slide patients.

Usage:
    python 04_train_mil.py --model uni
    python 04_train_mil.py --model conch
    python 04_train_mil.py --model uni --lr 1e-4 --epochs 100
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

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, truths = [], []

    for features, label in loader:
        features, label = features.to(device), label.to(device)

        logit, _ = model(features)
        loss = criterion(logit, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.append(torch.sigmoid(logit).item())
        truths.append(label.item())

    auc = roc_auc_score(truths, preds)
    return total_loss / len(loader), auc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, truths = [], []

    with torch.inference_mode():
        for features, label in loader:
            features, label = features.to(device), label.to(device)

            logit, _ = model(features)
            loss = criterion(logit, label)

            total_loss += loss.item()
            preds.append(torch.sigmoid(logit).item())
            truths.append(label.item())

    auc = roc_auc_score(truths, preds)
    acc = accuracy_score(truths, [int(p > 0.5) for p in preds])
    return total_loss / len(loader), auc, acc, preds, truths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(fold_results, all_preds, all_labels, val_indices, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    aucs = [r["val_auc"] for r in fold_results]
    n_folds = len(fold_results)

    # Per-fold AUC bar chart
    ax = axes[0]
    ax.bar(range(1, n_folds + 1), aucs, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(aucs), color="red", linestyle="--",
               label=f"Mean: {np.mean(aucs):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title(f"{model_name.upper()} — Validation AUC per Fold")
    ax.set_ylim(0, 1)
    ax.legend()

    # Pooled ROC curve
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train attention-based MIL for PRAME prediction")
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

    print(f"Model: {args.model.upper()}")
    print(f"Slides: {len(slides)} (high={labels.sum()}, low={len(labels) - labels.sum()})")
    print(f"Unique patients: {len(set(patients))}")
    print(f"Feature dim: {feat_dim}")

    # ------------------------------------------------------------------
    # Patient-level stratified k-fold
    # ------------------------------------------------------------------
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.seed)

    results_dir = Path(args.results_dir) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    all_val_preds = np.zeros(len(slides))
    all_val_labels = np.zeros(len(slides))
    val_indices_all = []

    # For training curve plots
    all_train_losses, all_val_losses = [], []
    all_train_aucs, all_val_aucs = [], []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(slides, labels, groups=patients)):

        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{args.folds}")
        print(f"  Train: {len(train_idx)} slides | Val: {len(val_idx)} slides")

        # Verify no patient leakage
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

        best_val_auc = 0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        fold_train_losses, fold_val_losses = [], []
        fold_train_aucs, fold_val_aucs = [], []

        for epoch in range(args.epochs):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)
            val_loss, val_auc, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device)
            scheduler.step()

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_aucs.append(train_auc)
            fold_val_aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} AUC: {train_auc:.3f} | "
                      f"Val Loss: {val_loss:.4f} AUC: {val_auc:.3f} "
                      f"Acc: {val_acc:.3f}")

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best: epoch {best_epoch})")
                break

        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_train_aucs.append(fold_train_aucs)
        all_val_aucs.append(fold_val_aucs)

        # Evaluate best model
        model.load_state_dict(best_state)
        _, val_auc, val_acc, val_preds, val_labels_fold = evaluate(
            model, val_loader, criterion, device)

        val_binary = [int(p > 0.5) for p in val_preds]
        tn, fp, fn, tp = confusion_matrix(val_labels_fold, val_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"  Best epoch: {best_epoch} | "
              f"Val AUC: {val_auc:.3f} Acc: {val_acc:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f} | "
              f"Specificity: {specificity:.3f}")

        fold_results.append({
            "fold": fold + 1,
            "best_epoch": best_epoch,
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

        # Save best model
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

    # Save CSV and JSON
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

    # Plots
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


if __name__ == "__main__":
    main()
