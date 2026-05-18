"""
Train the Component-2 PRAME-conditioned diagnostic MIL classifier.

Component 2 fuses H&E patch features and per-slide PRAME (measured for
SKCM/GTEx, pseudobulked for HEST, predicted by Component 1 for COBRA)
into a single multimodal MIL classifier. The PRAME scalar is projected
to the hidden representation space and appended to the patch bag as
one extra MIL instance; gated-attention pooling then co-attends over
the patch instances and the PRAME instance.

Three modes selected via --mode let the user ablate the PRAME signal:

    full          : PRAME projection + PRAME instance for has_prame=True rows
    no_predicted  : same as full, but rows with prame_source==
                    "component1_predicted" (COBRA) are silenced
                    (has_prame forced to False)
    no_prame      : PRAME projection branch is not built; pure visual MIL
                    baseline. Architecturally identical to Component 1's
                    AttentionMIL.

--compare runs all three modes through the full N-fold CV (--folds,
default 5) with the same deterministic patient-level split and emits
a bundled comparison plot + JSON summary on top of the per-mode CV
artifacts. Use `--compare --folds 1` for a single-fold sanity check.

Hyperparameters can be loaded from a tuning run via `--config PATH`
(produced by 09_tune_component2.py). CLI flags explicitly passed on the
command line always win over the loaded config. Regularizers
`--entropy-lambda`, `--grad-clip`, and `--label-smoothing` (all default 0,
meaning disabled) are wired into the train loop and selectable by the
tuner.

Usage:
    python 10_train_component2.py --mode full
    python 10_train_component2.py --mode no_predicted
    python 10_train_component2.py --mode no_prame
    python 10_train_component2.py --compare
    python 10_train_component2.py --mode full --epochs 100 --folds 5
    python 10_train_component2.py --compare --config results/uni/component2_tune/best_config.json
"""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# AttentionMIL is loaded from 04_train_mil.py (numeric prefix is not a valid
# Python module name; project convention is SourceFileLoader / spec_from_file).
# ---------------------------------------------------------------------------

def _load_attention_mil():
    here = Path(__file__).resolve().parent
    spec = spec_from_file_location("_mil_04", str(here / "04_train_mil.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AttentionMIL


AttentionMIL = _load_attention_mil()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEAT_DIMS = {"uni": 1024, "conch": 768}

# Per-cohort embedding subdirectory under --emb-dir. Component-1 SKCM lives
# at {emb_root}/{model}; the Component-2 cohorts get their own subdirs so
# they don't collide with the SKCM run.
SOURCE_EMB_SUBDIR = {
    "skcm_melanoma": "{model}",
    "skcm_normal":   "{model}",
    "gtex_normal":   "{model}_gtex",
    "cobra_bcc":     "{model}_cobra",
    "hest_visium":   "{model}_hest",
}

PREDICTED_PRAME_SOURCE = "component1_predicted"

MODES = ("full", "no_predicted", "no_prame")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Component2MIL(AttentionMIL):
    """PRAME-as-MIL-instance classifier.

    Inherits AttentionMIL's feature_net, gated attention, and classifier
    head. When use_prame=True, an additional 1->hidden_dim projection
    encodes the per-slide PRAME scalar; the projected vector is appended
    as one extra instance to the post-feature_net bag, so gated-attention
    pooling co-attends over patch instances and the PRAME instance.

    When use_prame=False, the prame_proj layer is not constructed and the
    model is architecturally identical to AttentionMIL.
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim=256,
        attn_dim=128,
        dropout=0.25,
        use_prame=True,
    ):
        super().__init__(feat_dim, hidden_dim, attn_dim, dropout)
        self.use_prame = use_prame
        if use_prame:
            # Tanh-bounded so the PRAME instance can't dominate the bag at init.
            self.prame_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
            )

    def forward(self, x, prame=None, has_prame=False):
        h = self.feature_net(x)  # (N, hidden_dim)

        if self.use_prame and bool(has_prame):
            prame_scalar = prame.view(1, 1).to(h.dtype)
            prame_h = self.prame_proj(prame_scalar)  # (1, hidden_dim)
            h = torch.cat([h, prame_h], dim=0)        # (N+1, hidden_dim)

        a_V = self.attention_V(h)
        a_U = self.attention_U(h)
        a = self.attention_w(a_V * a_U)
        attention = torch.softmax(a, dim=0).squeeze(-1)

        slide_repr = (attention.unsqueeze(-1) * h).sum(dim=0)
        logit = self.classifier(slide_repr).squeeze()
        return logit, attention, slide_repr


# ---------------------------------------------------------------------------
# DANN (domain-adversarial training) primitives
#
# The manifest correlates source_group with melanoma_label almost perfectly
# (all label=1 are TCGA-SKCM; all label=0 are GTEx/HEST/SKCM-tumor-free), so
# a vanilla MIL classifier learns scanner/cohort identity, not melanoma
# morphology. The gradient-reversal adversary forces the post-attention
# slide_repr to be cohort-invariant: the adversary tries to predict
# source_group from slide_repr, the GRL flips the gradient on the way into
# the backbone, so the backbone has to produce slide reprs that defeat the
# adversary. Lambda ramps via the DANN paper schedule.
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def dann_lambda(epoch, max_epochs, gamma=10.0, max_lambda=1.0):
    """DANN paper schedule: 2/(1+exp(-gamma*p)) - 1, p in [0,1]."""
    p = epoch / max(1, max_epochs - 1)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))


class CohortAdversary(nn.Module):
    def __init__(self, hidden_dim, adv_hidden_dim, num_cohorts, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, adv_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adv_hidden_dim, num_cohorts),
        )

    def forward(self, slide_repr):
        return self.net(slide_repr)


class Component2MILWithAdversary(nn.Module):
    """Backbone Component2MIL + GRL + CohortAdversary.

    forward returns (logit, attention, cohort_logits, slide_repr). Set
    self.grl.lambda_ per epoch from the caller before iterating the loader.
    """

    def __init__(self, backbone, num_cohorts, adv_hidden_dim, dropout):
        super().__init__()
        self.backbone = backbone
        self.grl = GradientReversal(lambda_=0.0)
        hidden_dim = backbone.classifier.in_features
        self.adversary = CohortAdversary(
            hidden_dim, adv_hidden_dim, num_cohorts, dropout,
        )

    def forward(self, x, prame=None, has_prame=False):
        logit, attention, slide_repr = self.backbone(x, prame, has_prame)
        cohort_logits = self.adversary(self.grl(slide_repr))
        return logit, attention, cohort_logits, slide_repr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SlideDataset(Dataset):
    """Per-slide patch features + PRAME + has_prame + cohort_idx + label."""

    def __init__(self, slide_paths, prames, has_prames, cohort_idxs, labels):
        self.slide_paths = slide_paths
        self.prames = prames
        self.has_prames = has_prames
        self.cohort_idxs = cohort_idxs
        self.labels = labels

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        with h5py.File(self.slide_paths[idx], "r") as f:
            features = f["features"][:].astype(np.float32)
        return (
            torch.from_numpy(features),
            torch.tensor(self.prames[idx], dtype=torch.float32),
            bool(self.has_prames[idx]),
            torch.tensor(self.cohort_idxs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def collate_bag(batch):
    # Batch size is always 1 (variable-size bags); skip the default stack.
    return batch[0]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _resolve_emb_path(row, emb_root, model_name):
    """Find the .h5 embedding for a manifest row.

    Tries cohort-specific subdirs (per SOURCE_EMB_SUBDIR) before falling
    back to Component-1's flat SKCM location. Tries multiple filename
    conventions because the SKCM run saves embeddings by .svs basename
    while the COBRA Colab notebook saves them by file_id stem.
    """
    file_id = str(row.get("file_id", "") or "")
    file_name = str(row.get("file_name", "") or "")
    source = str(row.get("source_group", "") or "")

    subdirs = []
    if source in SOURCE_EMB_SUBDIR:
        subdirs.append(SOURCE_EMB_SUBDIR[source].format(model=model_name))
    subdirs.append(model_name)

    names = []
    if file_name.endswith(".svs"):
        names.append(file_name.replace(".svs", ".h5"))
    if file_id:
        names.append(f"{file_id}.h5")
    if file_name and not file_name.endswith(".h5"):
        names.append(f"{Path(file_name).stem}.h5")
    if file_name.endswith(".h5"):
        names.append(file_name)

    for sd in subdirs:
        for n in names:
            p = Path(emb_root) / sd / n
            if p.exists():
                return p
    return None


def _preprocess_prame(prames, sources, norm):
    prames = np.asarray(prames, dtype=np.float32)
    sources = np.asarray(sources)
    if norm == "raw":
        return prames
    if norm == "log":
        return np.log1p(np.maximum(prames, 0.0))
    if norm == "zscore_per_source":
        out = np.zeros_like(prames)
        for s in np.unique(sources):
            mask = sources == s
            if mask.sum() > 1:
                vals = prames[mask]
                mu = float(vals.mean())
                sigma = float(vals.std() + 1e-8)
                out[mask] = (vals - mu) / sigma
        return out
    raise ValueError(f"unknown --prame-norm: {norm!r}")


def _build_cohort_mapping(df, encoding):
    """Add a `cohort_idx` int column to df and return (df, num_cohorts, names).

    Encoding choices:
        multiclass  - one class per source_group (4 by default in this project:
                      gtex_normal, hest_visium, skcm_melanoma, skcm_normal).
        binary_tcga - is_TCGA vs not (skcm_* -> 1; gtex/hest -> 0). Coarser
                      cohort scrubbing; lets the model still discriminate
                      GTEx-vs-HEST scanner signatures.
    """
    if encoding == "binary_tcga":
        df = df.copy()
        df["cohort_idx"] = df["source_group"].isin(
            ["skcm_melanoma", "skcm_normal"]
        ).astype(int)
        return df, 2, ["non_tcga", "tcga"]

    # multiclass: stable ordering by sorted source_group name
    sources = sorted(df["source_group"].unique().tolist())
    mapping = {s: i for i, s in enumerate(sources)}
    df = df.copy()
    df["cohort_idx"] = df["source_group"].map(mapping).astype(int)
    return df, len(sources), sources


def load_manifest(manifest_path, emb_root, model_name, mode, prame_norm):
    """Read manifest, resolve embeddings, apply mode-specific has_prame masking.

    Returns a DataFrame with columns: h5_path, prame, has_prame, label,
    patient, source_group, prame_source. PRAME has already been preprocessed
    via --prame-norm at this point.
    """
    raw = pd.read_csv(manifest_path)
    print(f"Manifest: {len(raw)} rows from {manifest_path}")

    # Defensive: older manifests may lack has_prame / prame_source.
    if "has_prame" not in raw.columns:
        raw["has_prame"] = raw["prame_tpm"].notna()
    if "prame_source" not in raw.columns:
        raw["prame_source"] = ""
    if "source_group" not in raw.columns:
        raw["source_group"] = ""

    rows = []
    missing = 0
    for _, r in raw.iterrows():
        p = _resolve_emb_path(r, emb_root, model_name)
        if p is None:
            missing += 1
            continue
        has_prame = bool(r["has_prame"]) and pd.notna(r.get("prame_tpm"))
        prame_source = str(r.get("prame_source", "") or "")

        if mode == "no_predicted" and prame_source == PREDICTED_PRAME_SOURCE:
            has_prame = False

        rows.append({
            "h5_path": str(p),
            "prame_raw": float(r["prame_tpm"]) if pd.notna(r.get("prame_tpm")) else 0.0,
            "has_prame": has_prame,
            "label": int(r["melanoma_label"]),
            "patient": str(r["submitter_id"]),
            "source_group": str(r.get("source_group", "") or ""),
            "prame_source": prame_source,
        })

    df = pd.DataFrame(rows)
    if missing:
        print(f"  [warn] {missing} rows skipped (no .h5 embedding found)")
    print(f"  {len(df)} slides resolved")

    if len(df) == 0:
        raise RuntimeError("No embeddings resolved; check --emb-dir and manifest paths.")

    # Per-source breakdown
    for s in sorted(df["source_group"].unique()):
        sub = df[df["source_group"] == s]
        n_pos = int(sub["label"].sum())
        n_prame = int(sub["has_prame"].sum())
        print(f"    {s:18s}  n={len(sub):4d}  pos={n_pos:4d}  has_prame={n_prame:4d}")

    df["prame"] = _preprocess_prame(
        df["prame_raw"].values, df["source_group"].values, prame_norm,
    )
    return df


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def compute_attention_entropy(attention):
    """Shannon entropy of a post-softmax attention vector. Higher = more spread."""
    return -(attention * torch.log(attention + 1e-8)).sum()


def _has_adv(model):
    return hasattr(model, "grl")


def train_one_epoch(model, loader, optimizer, criterion, device,
                    entropy_lambda=0.0, grad_clip=0.0, label_smoothing=0.0,
                    adv_lambda=0.0, cohort_criterion=None, amp=False):
    """Train one epoch.

    When the model is a Component2MILWithAdversary, sets `model.grl.lambda_`
    to adv_lambda before iterating, and adds the cohort-prediction CE to
    the main BCE. GRL flips the sign of that gradient on the way into the
    backbone, so the backbone is implicitly maximizing adv_loss.

    Returns (mean_main_loss, main_auc, mean_adv_loss, adv_acc). Without an
    adversary, mean_adv_loss/adv_acc are NaN.
    """
    model.train()
    if _has_adv(model):
        model.grl.lambda_ = adv_lambda
    total_main_loss = 0.0
    total_adv_loss = 0.0
    preds, truths = [], []
    adv_preds, adv_truths = [], []
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if amp and device.type == "cuda"
                    else _NullCtx())
    for features, prame, has_prame, cohort_idx, label in loader:
        features = features.to(device)
        prame = prame.to(device)
        cohort_idx = cohort_idx.to(device)
        label = label.to(device)

        target = label
        if label_smoothing > 0:
            target = label * (1 - 2 * label_smoothing) + label_smoothing

        with autocast_ctx:
            if _has_adv(model):
                logit, attention, cohort_logits, _ = model(features, prame, has_prame)
            else:
                logit, attention, _ = model(features, prame, has_prame)
                cohort_logits = None

            main_loss = criterion(logit, target)
            if entropy_lambda > 0:
                main_loss = main_loss - entropy_lambda * compute_attention_entropy(attention)

            if cohort_logits is not None and cohort_criterion is not None:
                # GRL handles the sign flip in backward; here we just add CE.
                # Backbone gradient becomes -lambda * d(adv_CE)/d(backbone),
                # adversary gradient is +d(adv_CE)/d(adversary).
                adv_loss = cohort_criterion(
                    cohort_logits.view(1, -1), cohort_idx.view(1),
                )
                loss = main_loss + adv_loss
            else:
                adv_loss = torch.tensor(0.0, device=device)
                loss = main_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_main_loss += main_loss.item()
        total_adv_loss += adv_loss.item()
        preds.append(torch.sigmoid(logit.detach().float()).item())
        truths.append(label.item())
        if cohort_logits is not None:
            adv_preds.append(int(cohort_logits.detach().argmax().item()))
            adv_truths.append(int(cohort_idx.item()))

    auc = _safe_auc(truths, preds)
    if adv_truths:
        adv_acc = float(accuracy_score(adv_truths, adv_preds))
    else:
        adv_acc = float("nan")
    return (
        total_main_loss / max(1, len(loader)),
        auc,
        total_adv_loss / max(1, len(loader)),
        adv_acc,
    )


def evaluate(model, loader, criterion, device, cohort_criterion=None, amp=False):
    """Evaluate. Returns (loss, auc, acc, preds, truths, adv_acc).

    The adversary's eval accuracy is the harder test of cohort-invariance:
    we want it converging to chance (1/num_cohorts) on held-out folds.
    Without an adversary, adv_acc is NaN.
    """
    model.eval()
    total_loss = 0.0
    preds, truths = [], []
    adv_preds, adv_truths = [], []
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if amp and device.type == "cuda"
                    else _NullCtx())
    with torch.inference_mode():
        for features, prame, has_prame, cohort_idx, label in loader:
            features = features.to(device)
            prame = prame.to(device)
            cohort_idx = cohort_idx.to(device)
            label = label.to(device)

            with autocast_ctx:
                if _has_adv(model):
                    logit, _, cohort_logits, _ = model(features, prame, has_prame)
                else:
                    logit, _, _ = model(features, prame, has_prame)
                    cohort_logits = None
                loss = criterion(logit, label)

            total_loss += loss.item()
            preds.append(torch.sigmoid(logit.float()).item())
            truths.append(label.item())
            if cohort_logits is not None:
                adv_preds.append(int(cohort_logits.argmax().item()))
                adv_truths.append(int(cohort_idx.item()))

    auc = _safe_auc(truths, preds)
    acc = accuracy_score(truths, [int(p > 0.5) for p in preds])
    if adv_truths:
        adv_acc = float(accuracy_score(adv_truths, adv_preds))
    else:
        adv_acc = float("nan")
    return total_loss / max(1, len(loader)), auc, acc, preds, truths, adv_acc


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def _safe_auc(truths, preds):
    if len(set(truths)) < 2:
        return float("nan")
    return roc_auc_score(truths, preds)


def _build_model(feat_dim, args, mode, num_cohorts, device):
    use_prame = mode in ("full", "no_predicted")
    backbone = Component2MIL(
        feat_dim,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
        use_prame=use_prame,
    ).to(device)
    if args.adv_disable or num_cohorts < 2:
        return backbone
    model = Component2MILWithAdversary(
        backbone,
        num_cohorts=num_cohorts,
        adv_hidden_dim=args.adv_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    return model


def train_one_fold(
    fold_idx,
    train_idx,
    val_idx,
    df,
    feat_dim,
    args,
    mode,
    num_cohorts,
    device,
):
    print(f"\n{'=' * 50}")
    print(f"Fold {fold_idx + 1}/{args.folds}  (mode={mode})")
    print(f"  Train: {len(train_idx)} slides | Val: {len(val_idx)} slides")

    train_patients = set(df.iloc[train_idx]["patient"])
    val_patients = set(df.iloc[val_idx]["patient"])
    assert train_patients.isdisjoint(val_patients), "Patient leakage!"

    train_ds = SlideDataset(
        df.iloc[train_idx]["h5_path"].tolist(),
        df.iloc[train_idx]["prame"].tolist(),
        df.iloc[train_idx]["has_prame"].tolist(),
        df.iloc[train_idx]["cohort_idx"].tolist(),
        df.iloc[train_idx]["label"].tolist(),
    )
    val_ds = SlideDataset(
        df.iloc[val_idx]["h5_path"].tolist(),
        df.iloc[val_idx]["prame"].tolist(),
        df.iloc[val_idx]["has_prame"].tolist(),
        df.iloc[val_idx]["cohort_idx"].tolist(),
        df.iloc[val_idx]["label"].tolist(),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_bag, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_bag, num_workers=0)

    model = _build_model(feat_dim, args, mode, num_cohorts, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    cohort_criterion = nn.CrossEntropyLoss() if _has_adv(model) else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None

    history = {
        "train_loss": [], "val_loss": [],
        "train_auc": [], "val_auc": [],
        "train_adv_acc": [], "val_adv_acc": [],
        "lambda_per_epoch": [],
    }

    effective_max_epochs = max(1, args.epochs - args.adv_warmup_epochs)

    for epoch in range(args.epochs):
        if _has_adv(model):
            if epoch < args.adv_warmup_epochs:
                current_lambda = 0.0
            else:
                current_lambda = dann_lambda(
                    epoch - args.adv_warmup_epochs,
                    effective_max_epochs,
                    max_lambda=args.adv_lambda,
                )
        else:
            current_lambda = 0.0

        train_loss, train_auc, _train_adv_loss, train_adv_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            entropy_lambda=args.entropy_lambda,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            adv_lambda=current_lambda,
            cohort_criterion=cohort_criterion,
            amp=args.amp,
        )
        val_loss, val_auc, val_acc, _, _, val_adv_acc = evaluate(
            model, val_loader, criterion, device,
            cohort_criterion=cohort_criterion, amp=args.amp,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_adv_acc"].append(train_adv_acc)
        history["val_adv_acc"].append(val_adv_acc)
        history["lambda_per_epoch"].append(current_lambda)

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            adv_tag = (f" | AdvAcc tr {train_adv_acc:.3f} va {val_adv_acc:.3f} "
                       f"lam {current_lambda:.3f}"
                       if _has_adv(model) else "")
            print(f"  Epoch {epoch + 1:3d} | "
                  f"Train Loss {train_loss:.4f} AUC {train_auc:.3f} | "
                  f"Val Loss {val_loss:.4f} AUC {val_auc:.3f} Acc {val_acc:.3f}"
                  f"{adv_tag}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch + 1} (best: epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_auc, val_acc, val_preds, val_truths, final_adv_acc = evaluate(
        model, val_loader, criterion, device,
        cohort_criterion=cohort_criterion, amp=args.amp,
    )

    val_binary = [int(p > 0.5) for p in val_preds]
    if len(set(val_truths)) >= 2:
        tn, fp, fn, tp = confusion_matrix(val_truths, val_binary).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sens = spec = float("nan")

    fold_result = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "val_auc": val_auc,
        "val_acc": val_acc,
        "sensitivity": sens,
        "specificity": spec,
        "final_adv_acc": final_adv_acc,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
    }
    return fold_result, history, best_state, val_preds, val_truths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cv_results(fold_results, pool_truths, pool_preds, model_name, mode, out_dir):
    aucs = [r["val_auc"] for r in fold_results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(range(1, len(aucs) + 1), aucs, color="steelblue", alpha=0.8)
    ax.axhline(np.nanmean(aucs), color="red", linestyle="--",
               label=f"Mean: {np.nanmean(aucs):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title(f"{model_name.upper()} {mode} - Val AUC per Fold")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1]
    if len(set(pool_truths)) >= 2:
        fpr, tpr, _ = roc_curve(pool_truths, pool_preds)
        pooled_auc = roc_auc_score(pool_truths, pool_preds)
        ax.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"Pooled AUC = {pooled_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name.upper()} {mode} - Pooled ROC")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"cv_results_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(histories_per_fold, model_name, mode, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, h in enumerate(histories_per_fold):
        axes[0].plot(h["train_loss"], alpha=0.5, label=f"Fold {i+1} train")
        axes[0].plot(h["val_loss"], alpha=0.5, linestyle="--", label=f"Fold {i+1} val")
        axes[1].plot(h["train_auc"], alpha=0.5, label=f"Fold {i+1} train")
        axes[1].plot(h["val_auc"], alpha=0.5, linestyle="--", label=f"Fold {i+1} val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} {mode} - Loss")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title(f"{model_name.upper()} {mode} - AUC")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / f"training_curves_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_compare(per_mode, model_name, n_folds, num_cohorts, out_dir):
    """3x2 grid summarizing an N-fold CV ablation across three PRAME modes.

    Designed so RELATIVE differences are readable even when absolute val AUCs
    saturate near 1.0 under cohort confound.

    (0,0) Val AUC per epoch: every fold as a thin trace + bold mean line per mode.
    (0,1) Pooled ROC across all folds, one curve per mode.
    (1,0) Paired delta per fold: full-no_prame, no_predicted-no_prame.
    (1,1) Per-cohort-pair AUC per mode (the real leakage diagnostic).
    (2,0) Adversary accuracy per epoch (train+val) per mode, chance line dashed.
    (2,1) Paired-difference summary with Wilcoxon p-value asterisks.
    """
    colors = {
        "full":         "steelblue",
        "no_predicted": "coral",
        "no_prame":     "seagreen",
    }
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # (0,0) Per-fold val AUC traces + mean line per mode
    ax = axes[0, 0]
    for mode in MODES:
        c = colors[mode]
        histories = per_mode[mode]["histories"]
        for h in histories:
            ax.plot(h["val_auc"], color=c, alpha=0.25, lw=1)
        max_ep = max(len(h["val_auc"]) for h in histories) if histories else 0
        means = []
        for ep in range(max_ep):
            vals = [h["val_auc"][ep] for h in histories if ep < len(h["val_auc"])]
            means.append(float(np.nanmean(vals)) if vals else np.nan)
        ax.plot(means, color=c, lw=2.5, label=mode)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Val AUC per Fold (light) + Mean (bold)")
    ax.legend()
    ax.set_ylim(0, 1)

    # (0,1) Pooled ROC across folds per mode
    ax = axes[0, 1]
    for mode in MODES:
        c = colors[mode]
        truths = per_mode[mode]["pool_truths"]
        preds = per_mode[mode]["pool_preds"]
        if len(set(truths)) >= 2:
            fpr, tpr, _ = roc_curve(truths, preds)
            auc = roc_auc_score(truths, preds)
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{mode} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC (held-out predictions, all folds)")
    ax.legend()

    # (1,0) Paired delta per fold: (full-no_prame) and (no_predicted-no_prame).
    # Bars can be negative; that's informative. This is the panel that stays
    # readable when absolutes are at the ceiling.
    ax = axes[1, 0]
    base_aucs = np.array([r["val_auc"] for r in per_mode["no_prame"]["fold_results"]])
    full_aucs = np.array([r["val_auc"] for r in per_mode["full"]["fold_results"]])
    nopred_aucs = np.array([r["val_auc"] for r in per_mode["no_predicted"]["fold_results"]])
    delta_full = full_aucs - base_aucs
    delta_nopred = nopred_aucs - base_aucs
    x = np.arange(n_folds)
    width = 0.4
    ax.bar(x - width / 2, delta_full,   width,
           color=colors["full"],         alpha=0.85, label="full - no_prame")
    ax.bar(x + width / 2, delta_nopred, width,
           color=colors["no_predicted"], alpha=0.85, label="no_predicted - no_prame")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {k + 1}" for k in range(n_folds)])
    ax.set_ylabel("Delta Val AUC")
    ax.set_title("Per-Fold Paired Delta vs no_prame baseline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    # (1,1) Per-cohort-pair AUC per mode. The REAL diagnostic for cohort
    # leakage: if skcm_vs_gtex >> skcm_vs_hest, the model is still using
    # scanner identity, not melanoma morphology.
    ax = axes[1, 1]
    pair_keys = list(per_mode["full"].get("pairwise_auc", {}).keys())
    if pair_keys:
        x = np.arange(len(pair_keys))
        width = 0.8 / len(MODES)
        for i, mode in enumerate(MODES):
            pw = per_mode[mode].get("pairwise_auc", {})
            vals = [pw.get(k, float("nan")) for k in pair_keys]
            offset = (i - (len(MODES) - 1) / 2) * width
            ax.bar(x + offset, vals, width, color=colors[mode], alpha=0.85,
                   label=mode)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [k.replace("_vs_", "\nvs\n") for k in pair_keys],
            fontsize=7,
        )
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="chance")
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1.05)
        ax.set_title("Pairwise AUC by Cohort Pair (Pooled Val)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "(no pairwise AUC available)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Pairwise AUC by Cohort Pair (Pooled Val)")

    # (2,0) Adversary accuracy over training, per mode. If DANN is working,
    # train_adv_acc and val_adv_acc converge to chance (1/num_cohorts).
    ax = axes[2, 0]
    any_adv = False
    for mode in MODES:
        c = colors[mode]
        histories = per_mode[mode]["histories"]
        max_ep = max((len(h["val_adv_acc"]) for h in histories), default=0)
        if max_ep == 0:
            continue
        train_means, val_means = [], []
        for ep in range(max_ep):
            tr = [h["train_adv_acc"][ep] for h in histories
                  if ep < len(h["train_adv_acc"]) and not np.isnan(h["train_adv_acc"][ep])]
            va = [h["val_adv_acc"][ep] for h in histories
                  if ep < len(h["val_adv_acc"]) and not np.isnan(h["val_adv_acc"][ep])]
            train_means.append(float(np.mean(tr)) if tr else np.nan)
            val_means.append(float(np.mean(va)) if va else np.nan)
        if any(not np.isnan(v) for v in train_means):
            any_adv = True
            ax.plot(train_means, color=c, lw=1.5, linestyle="--",
                    label=f"{mode} train")
            ax.plot(val_means, color=c, lw=2.5, label=f"{mode} val")
    if any_adv:
        chance = 1.0 / num_cohorts if num_cohorts > 0 else 0.5
        ax.axhline(chance, color="black", linestyle=":", lw=1.5,
                   label=f"chance ({chance:.2f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Adversary Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Cohort-Adversary Accuracy (DANN diagnostic)")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "(DANN disabled - no adversary trained)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cohort-Adversary Accuracy (DANN diagnostic)")

    # (2,1) Paired-difference summary with Wilcoxon sign-test asterisks.
    ax = axes[2, 1]
    pairs_to_plot = [("full", "no_prame"), ("no_predicted", "no_prame")]
    labels = []
    means = []
    stds = []
    stars = []
    for tgt, base in pairs_to_plot:
        a = np.array([r["val_auc"] for r in per_mode[tgt]["fold_results"]])
        b = np.array([r["val_auc"] for r in per_mode[base]["fold_results"]])
        delta = a - b
        means.append(float(np.mean(delta)))
        stds.append(float(np.std(delta)))
        p = float("nan")
        try:
            from scipy.stats import wilcoxon
            if np.any(delta != 0):
                _, p = wilcoxon(delta)
        except Exception:
            pass
        stars.append(_sig_stars(p))
        labels.append(f"{tgt}\n−\n{base}")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6,
           color=[colors["full"], colors["no_predicted"]], alpha=0.85)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Delta Val AUC (+/- std)")
    ax.set_title("Paired Differences (Wilcoxon: * p<.05, ** p<.01, *** p<.001)")
    # Print stars above bars
    for i, (m, sd, st) in enumerate(zip(means, stds, stars)):
        y = m + sd + 0.005 * max(abs(np.array(means)).max(), 0.01)
        if st:
            ax.text(i, y, st, ha="center", va="bottom", fontsize=14,
                    fontweight="bold")

    plt.suptitle(
        f"{model_name.upper()} - PRAME Ablation ({n_folds}-fold CV)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_dir / "compare_variants.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _cv_split(df, args):
    skf = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed,
    )
    return list(skf.split(
        df["h5_path"].values,
        df["label"].values,
        groups=df["patient"].values,
    ))


_PAIRWISE_PAIRS = (
    ("skcm_melanoma", "gtex_normal"),
    ("skcm_melanoma", "hest_visium"),
    ("skcm_melanoma", "skcm_normal"),
    ("hest_visium",   "gtex_normal"),
)


def _per_source_breakdown(val_sources, val_truths, val_preds):
    """Return (per_source_group, pairwise_auc) dicts for diagnostic output."""
    per_source = {}
    for s in sorted(set(val_sources)):
        mask = np.array([v == s for v in val_sources])
        sub_truths = val_truths[mask]
        sub_preds = val_preds[mask]
        per_source[s] = {
            "n": int(mask.sum()),
            "mean_pred": float(np.mean(sub_preds)) if len(sub_preds) else float("nan"),
            "mean_label": float(np.mean(sub_truths)) if len(sub_truths) else float("nan"),
        }

    pairwise_auc = {}
    for a, b in _PAIRWISE_PAIRS:
        key = f"{a}_vs_{b}"
        mask = np.array([v in (a, b) for v in val_sources])
        if mask.sum() < 4:
            pairwise_auc[key] = float("nan")
            continue
        sub_truths = val_truths[mask]
        sub_preds = val_preds[mask]
        if len(set(sub_truths.tolist())) < 2:
            pairwise_auc[key] = float("nan")
            continue
        pairwise_auc[key] = float(roc_auc_score(sub_truths, sub_preds))
    return per_source, pairwise_auc


def run_full_cv(df, feat_dim, args, mode, num_cohorts, cohort_names,
                device, results_dir):
    splits = _cv_split(df, args)
    fold_results = []
    histories = []
    all_val_preds = np.zeros(len(df))
    all_val_labels = np.zeros(len(df))
    val_indices_all = []
    val_sources_all = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        result, history, best_state, val_preds, val_truths = train_one_fold(
            fold_i, train_idx, val_idx, df, feat_dim, args, mode,
            num_cohorts, device,
        )
        fold_results.append(result)
        histories.append(history)

        if best_state is not None:
            torch.save(
                best_state,
                results_dir / f"fold{fold_i + 1}_{mode}_model.pt",
            )

        for i, idx in enumerate(val_idx):
            all_val_preds[idx] = val_preds[i]
            all_val_labels[idx] = val_truths[i]
        val_indices_all.extend(int(x) for x in val_idx.tolist())
        val_sources_all.extend(df.iloc[val_idx]["source_group"].tolist())

        adv_msg = (f" | AdvAcc(final) {result['final_adv_acc']:.3f}"
                   if not np.isnan(result.get("final_adv_acc", np.nan))
                   else "")
        print(f"  Best epoch {result['best_epoch']} | "
              f"Val AUC {result['val_auc']:.3f} Acc {result['val_acc']:.3f} | "
              f"Sens {result['sensitivity']:.3f} Spec {result['specificity']:.3f}"
              f"{adv_msg}")

    # Aggregate
    print(f"\n{'=' * 50}\nAGGREGATE - {args.model.upper()} mode={mode}\n{'=' * 50}")
    aucs = [r["val_auc"] for r in fold_results]
    accs = [r["val_acc"] for r in fold_results]
    sens = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]
    adv_accs = [r.get("final_adv_acc", float("nan")) for r in fold_results]

    val_idx_arr = np.array(val_indices_all)
    if len(set(all_val_labels[val_idx_arr].tolist())) >= 2:
        pooled_auc = float(roc_auc_score(
            all_val_labels[val_idx_arr], all_val_preds[val_idx_arr],
        ))
    else:
        pooled_auc = float("nan")

    # Per-source-group + pairwise AUC diagnostics. This is THE diagnostic for
    # whether the DANN fix worked. If skcm_melanoma_vs_gtex_normal is far above
    # skcm_melanoma_vs_hest_visium, the model is still cohort-detecting.
    per_source, pairwise_auc = _per_source_breakdown(
        val_sources_all,
        all_val_labels[val_idx_arr],
        all_val_preds[val_idx_arr],
    )

    print(f"AUC         {np.nanmean(aucs):.3f} +/- {np.nanstd(aucs):.3f}")
    print(f"Acc         {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    print(f"Sensitivity {np.nanmean(sens):.3f} +/- {np.nanstd(sens):.3f}")
    print(f"Specificity {np.nanmean(specs):.3f} +/- {np.nanstd(specs):.3f}")
    print(f"Pooled AUC  {pooled_auc:.3f}")
    if not np.all(np.isnan(adv_accs)):
        chance = 1.0 / num_cohorts
        print(f"Adv acc     {np.nanmean(adv_accs):.3f} +/- "
              f"{np.nanstd(adv_accs):.3f}  (chance={chance:.3f})")
    print("Pairwise AUC (per cohort pair on pooled val preds):")
    for k, v in pairwise_auc.items():
        print(f"  {k:42s} = {v:.3f}" if not np.isnan(v)
              else f"  {k:42s} = n/a")

    # Save per-fold CSV
    pd.DataFrame(fold_results).to_csv(
        results_dir / f"cv_results_{mode}.csv", index=False,
    )

    chance_acc = 1.0 / num_cohorts if num_cohorts > 0 else float("nan")

    # Summary JSON
    summary = {
        "model": args.model,
        "mode": mode,
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "prame_norm": args.prame_norm,
        "entropy_lambda": args.entropy_lambda,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "adv_lambda_max": (0.0 if args.adv_disable else float(args.adv_lambda)),
        "adv_disabled": bool(args.adv_disable),
        "adv_warmup_epochs": int(args.adv_warmup_epochs),
        "adv_hidden_dim": int(args.adv_hidden_dim),
        "cohort_encoding": args.cohort_encoding,
        "num_cohorts": int(num_cohorts),
        "cohort_names": list(cohort_names),
        "adversary_chance_acc": float(chance_acc),
        "amp": bool(args.amp),
        "device": device.type,
        "num_slides": len(df),
        "num_patients": int(df["patient"].nunique()),
        "mean_auc": float(np.nanmean(aucs)),
        "std_auc": float(np.nanstd(aucs)),
        "mean_acc": float(np.nanmean(accs)),
        "std_acc": float(np.nanstd(accs)),
        "mean_sensitivity": float(np.nanmean(sens)),
        "std_sensitivity": float(np.nanstd(sens)),
        "mean_specificity": float(np.nanmean(specs)),
        "std_specificity": float(np.nanstd(specs)),
        "pooled_auc": pooled_auc,
        "adversary_final_acc_per_fold": [float(a) for a in adv_accs],
        "adversary_final_acc_mean": float(np.nanmean(adv_accs)) if not np.all(np.isnan(adv_accs)) else float("nan"),
        "per_source_group": per_source,
        "pairwise_auc": pairwise_auc,
    }
    with open(results_dir / f"summary_{mode}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_cv_results(
        fold_results,
        all_val_labels[val_idx_arr].tolist(),
        all_val_preds[val_idx_arr].tolist(),
        args.model, mode, results_dir,
    )
    plot_training_curves(histories, args.model, mode, results_dir)

    print(f"\nSaved to {results_dir}/")
    print(f"  cv_results_{mode}.csv")
    print(f"  summary_{mode}.json")
    print(f"  cv_results_{mode}.png")
    print(f"  training_curves_{mode}.png")
    print(f"  fold*_{mode}_model.pt")

    return {
        "fold_results": fold_results,
        "histories": histories,
        "summary": summary,
        "pool_truths": all_val_labels[val_idx_arr].tolist() if len(val_idx_arr) else [],
        "pool_preds":  all_val_preds[val_idx_arr].tolist()  if len(val_idx_arr) else [],
        "pool_sources": val_sources_all,
        "per_source_group": per_source,
        "pairwise_auc": pairwise_auc,
        "num_cohorts": num_cohorts,
        "cohort_names": cohort_names,
    }


def _apply_mode_masking(df, mode):
    """Return a (copy of) df with mode-specific has_prame masking applied."""
    if mode == "no_predicted":
        out = df.copy()
        out.loc[out["prame_source"] == PREDICTED_PRAME_SOURCE, "has_prame"] = False
        return out
    return df


def run_compare(df, feat_dim, args, num_cohorts, cohort_names, device, results_dir):
    """Run a full N-fold CV (--folds, default 5) for each of the three modes
    and emit a bundled comparison plot + JSON on top of the per-mode CV
    artifacts that run_full_cv already writes.

    The same deterministic patient-level split (seed=42) is used across all
    three modes so per-fold metrics are directly comparable.
    """
    out_dir = results_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[compare] Running {args.folds}-fold CV for each of {len(MODES)} modes")

    per_mode = {}
    for mode in MODES:
        df_mode = _apply_mode_masking(df, mode)
        print(f"\n{'=' * 60}")
        print(f"[compare] mode = {mode}")
        print(f"{'=' * 60}")
        per_mode[mode] = run_full_cv(
            df_mode, feat_dim, args, mode, num_cohorts, cohort_names,
            device, results_dir,
        )

    # Bundled artifacts on top of the per-mode CV outputs
    plot_compare(per_mode, args.model, args.folds, num_cohorts, out_dir)

    summary_json = {}
    for mode in MODES:
        s = per_mode[mode]["summary"]
        per_fold = per_mode[mode]["fold_results"]
        summary_json[mode] = {
            "val_auc_per_fold":     [r["val_auc"] for r in per_fold],
            "val_acc_per_fold":     [r["val_acc"] for r in per_fold],
            "sensitivity_per_fold": [r["sensitivity"] for r in per_fold],
            "specificity_per_fold": [r["specificity"] for r in per_fold],
            "mean_val_auc":     s["mean_auc"],
            "std_val_auc":      s["std_auc"],
            "mean_val_acc":     s["mean_acc"],
            "std_val_acc":      s["std_acc"],
            "mean_sensitivity": s["mean_sensitivity"],
            "std_sensitivity":  s["std_sensitivity"],
            "mean_specificity": s["mean_specificity"],
            "std_specificity":  s["std_specificity"],
            "pooled_auc":       s["pooled_auc"],
            "adversary_final_acc_per_fold": s.get("adversary_final_acc_per_fold", []),
            "adversary_final_acc_mean":     s.get("adversary_final_acc_mean", float("nan")),
            "adversary_chance_acc":         s.get("adversary_chance_acc", float("nan")),
            "per_source_group":             s.get("per_source_group", {}),
            "pairwise_auc":                 s.get("pairwise_auc", {}),
        }

    # Paired-difference deltas: full vs no_prame and no_predicted vs no_prame.
    # Wilcoxon signed-rank p-values surface "is the PRAME branch actually
    # contributing real signal, or is the apparent gap noise?"
    def _paired(target, baseline):
        a = np.array([r["val_auc"]
                      for r in per_mode[target]["fold_results"]])
        b = np.array([r["val_auc"]
                      for r in per_mode[baseline]["fold_results"]])
        delta = a - b
        out = {
            "deltas_per_fold": delta.tolist(),
            "mean_delta": float(np.mean(delta)),
            "std_delta":  float(np.std(delta)),
        }
        try:
            from scipy.stats import wilcoxon
            if np.any(delta != 0):
                stat, p = wilcoxon(delta)
                out["wilcoxon_stat"] = float(stat)
                out["wilcoxon_p"]    = float(p)
            else:
                out["wilcoxon_stat"] = float("nan")
                out["wilcoxon_p"]    = float("nan")
        except Exception:
            out["wilcoxon_stat"] = float("nan")
            out["wilcoxon_p"]    = float("nan")
        return out

    summary_json["paired_deltas"] = {
        "full_vs_no_prame":          _paired("full", "no_prame"),
        "no_predicted_vs_no_prame":  _paired("no_predicted", "no_prame"),
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # Headline table
    print(f"\n{'=' * 60}")
    print(f"COMPARISON SUMMARY ({args.folds}-fold CV per mode)")
    print(f"{'=' * 60}")
    print(f"  {'mode':14s} {'val_auc':>17s} {'val_acc':>17s} "
          f"{'sens':>17s} {'spec':>17s} {'pooled':>8s}")
    for mode in MODES:
        s = summary_json[mode]
        def _fmt(m, sd):
            return f"{m:.3f} +/- {sd:.3f}"
        print(f"  {mode:14s} "
              f"{_fmt(s['mean_val_auc'], s['std_val_auc']):>17s} "
              f"{_fmt(s['mean_val_acc'], s['std_val_acc']):>17s} "
              f"{_fmt(s['mean_sensitivity'], s['std_sensitivity']):>17s} "
              f"{_fmt(s['mean_specificity'], s['std_specificity']):>17s} "
              f"{s['pooled_auc']:>8.3f}")
    print(f"\n[compare] Saved to {out_dir}/")
    print(f"  comparison.json")
    print(f"  compare_variants.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Hyperparameter keys that 09_tune_component2.py writes into best_config.json.
# CLI flags explicitly passed on the command line still win over the config.
_CONFIG_HYPERPARAMS = (
    "lr", "weight_decay", "hidden_dim", "attn_dim", "dropout", "patience",
    "prame_norm", "entropy_lambda", "grad_clip", "label_smoothing",
    "adv_lambda", "adv_warmup_epochs", "adv_hidden_dim", "cohort_encoding",
)


def _build_parser():
    p = argparse.ArgumentParser(
        description="Train Component-2 PRAME-conditioned diagnostic MIL.",
    )
    p.add_argument("--model", choices=list(FEAT_DIMS.keys()), default="uni",
                   help="Foundation model embeddings (default: uni)")
    p.add_argument("--mode", choices=list(MODES), default="full",
                   help="PRAME ablation mode (default: full)")
    p.add_argument("--manifest",
                   default="data/expression/diagnostic_manifest.csv")
    p.add_argument("--emb-dir", default="embeddings",
                   help="Root for embeddings; cohort subdirs resolved per source_group")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--attn-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prame-norm",
                   choices=("log", "raw", "zscore_per_source"),
                   default="zscore_per_source",
                   help="PRAME preprocessing (default: zscore_per_source). "
                        "z-score-per-source centers each cohort at 0/1, so "
                        "the PRAME instance can't encode cohort-conditional "
                        "magnitude differences. Use 'log' to revert.")
    p.add_argument("--entropy-lambda", type=float, default=0.0,
                   help="Attention entropy regularizer weight (0 = disabled)")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Max gradient norm for clip_grad_norm_ (0 = disabled)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="BCE label smoothing factor (0 = disabled)")
    p.add_argument("--adv-lambda", type=float, default=1.0,
                   help="Max GRL lambda for the DANN cohort-adversary. "
                        "0 has no effect because the schedule ramps from 0; "
                        "use --adv-disable to turn DANN off cleanly.")
    p.add_argument("--adv-warmup-epochs", type=int, default=0,
                   help="Epochs before the adversary engages (lambda=0).")
    p.add_argument("--adv-hidden-dim", type=int, default=64,
                   help="Hidden dim for the CohortAdversary MLP.")
    p.add_argument("--adv-disable", action="store_true",
                   help="Force DANN off (overrides --adv-lambda). Useful for "
                        "A/B comparison vs the adversarial fix.")
    p.add_argument("--cohort-encoding",
                   choices=("binary_tcga", "multiclass"), default="multiclass",
                   help="Cohort labels for the adversary. multiclass = one "
                        "class per source_group (default, scrubs all "
                        "scanner identities). binary_tcga = is_TCGA only "
                        "(coarser).")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast for forward pass (CUDA only)")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                   help="Computation device (default: auto)")
    p.add_argument("--config", default="",
                   help="Path to best_config.json from 09_tune_component2.py. "
                        "Hyperparameters in the JSON populate defaults; "
                        "explicit CLI flags still win.")
    p.add_argument("--compare", action="store_true",
                   help="Run all three modes through the full --folds CV and "
                        "emit bundled comparison artifacts on top of per-mode CV")
    return p


def parse_args():
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"--config not found: {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)
        hp = cfg.get("hyperparameters", {}) or {}
        # Detect which flags appeared on the command line; those override config.
        explicit = _explicit_cli_flags()
        for key in _CONFIG_HYPERPARAMS:
            cli_name = "--" + key.replace("_", "-")
            if cli_name in explicit:
                continue
            if key in hp:
                setattr(args, key, type(getattr(args, key))(hp[key]))
        print(f"Loaded config from {cfg_path}")
        for key in _CONFIG_HYPERPARAMS:
            print(f"  {key} = {getattr(args, key)}")
    return args


def _explicit_cli_flags():
    """Return the set of `--flag` tokens that appear in sys.argv."""
    import sys as _sys
    return {tok.split("=", 1)[0] for tok in _sys.argv[1:] if tok.startswith("--")}


def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("  [warn] --device cuda requested but CUDA not available; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Model:  {args.model.upper()}  (feat_dim={FEAT_DIMS[args.model]})")
    print(f"Mode:   {args.mode if not args.compare else f'compare (all three modes, {args.folds}-fold)'}")
    print(f"PRAME norm: {args.prame_norm}")
    if args.adv_disable:
        print(f"DANN:   disabled (--adv-disable)")
    else:
        print(f"DANN:   adv_lambda={args.adv_lambda}  "
              f"warmup={args.adv_warmup_epochs}  "
              f"hidden={args.adv_hidden_dim}  encoding={args.cohort_encoding}")

    feat_dim = FEAT_DIMS[args.model]

    # For --compare, load manifest in "full" mode (no_predicted masking is applied
    # per-mode inside run_compare).
    load_mode = args.mode if not args.compare else "full"
    df = load_manifest(args.manifest, args.emb_dir, args.model, load_mode, args.prame_norm)

    # Build cohort_idx column for the DANN adversary. Even when adversary is
    # disabled, the loader still threads a cohort_idx through (harmless) so
    # SlideDataset's signature is stable.
    df, num_cohorts, cohort_names = _build_cohort_mapping(df, args.cohort_encoding)
    print(f"Cohorts ({num_cohorts}): {cohort_names}")

    results_dir = Path(args.results_dir) / args.model / "component2"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        run_compare(df, feat_dim, args, num_cohorts, cohort_names, device, results_dir)
    else:
        run_full_cv(
            df, feat_dim, args, args.mode, num_cohorts, cohort_names,
            device, results_dir,
        )


if __name__ == "__main__":
    main()
