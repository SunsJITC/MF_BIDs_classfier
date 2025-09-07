# -*- coding: utf-8 -*-
"""
cross_attention_unimodal_model.py
=================================
Unimodal classifier with two pathology branches (UNI & MAE-adapted UNI)
fused via bidirectional cross-attention, followed by a linear head.

- Inputs:
  * UNI_XLS: Excel with columns [ID, WSI_Name, Group, Label1, F0..F*]
  * MAE_XLS: Excel with columns [ID, WSI_Name, F0..F*] (will be renamed to F*_mae)
  * Group must include: 'train', 'internal', 'test'

- Early stopping:
  * Done on a 9:1 split **within train** (NOT using 'internal').

- Outputs:
  * Saves best CV config and final metrics to OUT_DIR.
"""

import os
import json
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


# ------------------------ Paths & Globals ------------------------
# ------------------------ Paths & Globals ------------------------
UNI_XLS     = "/Users/yatedisihao/Desktop/UNI.xlsx"
MAE_XLS     = "/Users/yatedisihao/Desktop/MAE.xlsx"
OUT_DIR     = "/Users/yatedisihao/Desktop"
os.makedirs(OUT_DIR, exist_ok=True)

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid search (keep moderate; widen if needed)
PCA_GRID   = [32, 48, 64]        # PCA dim for UNI/MAE (should be divisible by HEADS)
BATCH_GRID = [8, 16, 32]
HEADS_GRID = [2, 4]              # num_heads (ensure pca_dim % heads == 0)
DROP_GRID  = [0.0, 0.1, 0.2]
LR_GRID    = [5e-4, 1e-3]
WD_GRID    = [1e-2, 5e-3]
EPCV_GRID  = [150, 300]

MAX_EPOCHS_FINAL    = 300
EARLY_STOP_PATIENCE = 10


# --------------------- Reproducibility ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- Data I/O & Merge ---------------------
def load_and_merge(uni_xls: str, mae_xls: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load UNI & MAE tables; align and merge feature columns.

    Expected columns:
      - shared: ID, WSI_Name, Group, Label1 (Group ∈ {train, internal, test})
      - UNI features: F0..F*
      - MAE features: F0..F* (will be renamed -> F*_mae)
    """
    df_uni = pd.read_excel(uni_xls)
    df_mae = pd.read_excel(mae_xls)
    df_uni.columns = df_uni.columns.astype(str)
    df_mae.columns = df_mae.columns.astype(str)

    meta_cols = ["ID", "WSI_Name", "Group", "Label1"]
    for col in meta_cols:
        if col not in df_uni.columns:
            raise ValueError(f"UNI file must contain column: {col}")

    feat_uni_cols = [c for c in df_uni.columns if c.startswith("F")]
    if not feat_uni_cols:
        raise ValueError("UNI file has no feature columns starting with 'F'.")

    mae_feats = [c for c in df_mae.columns if c.startswith("F")]
    if not mae_feats:
        raise ValueError("MAE file has no feature columns starting with 'F'.")

    # rename MAE features and align order with UNI columns
    df_mae = df_mae.rename(columns={c: f"{c}_mae" for c in mae_feats})
    feat_mae_cols = [f"{c}_mae" for c in feat_uni_cols if f"{c}_mae" in df_mae.columns]
    if len(feat_mae_cols) != len(feat_uni_cols):
        raise ValueError("UNI and MAE features are misaligned; ensure same F ordering/shape.")

    # merge
    df = (
        df_uni[meta_cols + feat_uni_cols]
          .merge(df_mae[["ID", "WSI_Name"] + feat_mae_cols], on=["ID", "WSI_Name"], how="left")
    )
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].reset_index(drop=True)
    return df, feat_uni_cols, feat_mae_cols


# --------------------- Dataset ---------------------
class WsiDS(Dataset):
    def __init__(self, Xu: np.ndarray, Xm: np.ndarray, y: np.ndarray):
        self.Xu = Xu.astype(np.float32)
        self.Xm = Xm.astype(np.float32)
        self.y  = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xu[i], self.Xm[i], self.y[i]


# --------------------- Model ---------------------
class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention on single-token embeddings:
      - Path1: Q = Xm (MAE-UNI), K/V = Xu (UNI)
      - Path2: Q = Xu,            K/V = Xm (if bidirectional=True)
      Concatenate and project back to 'dim'.
    """
    def __init__(self, dim: int, heads: int = 4, p_drop: float = 0.1, bidirectional: bool = True):
        super().__init__()
        assert dim % heads == 0, "embed_dim must be divisible by num_heads"
        self.bidirectional = bidirectional
        self.attn_m2u = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=p_drop, batch_first=True)
        if bidirectional:
            self.attn_u2m = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=p_drop, batch_first=True)
            out_dim = dim * 2
        else:
            self.attn_u2m = None
            out_dim = dim
        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, dim),
            nn.GELU(),
            nn.Dropout(p_drop)
        )

    def forward(self, x_u: torch.Tensor, x_m: torch.Tensor) -> torch.Tensor:
        # x_u/x_m: [B, D] -> [B, 1, D]
        q1 = x_m.unsqueeze(1); k1 = x_u.unsqueeze(1); v1 = x_u.unsqueeze(1)
        z1, _ = self.attn_m2u(q1, k1, v1)                 # [B,1,D]
        if self.bidirectional:
            q2 = x_u.unsqueeze(1); k2 = x_m.unsqueeze(1); v2 = x_m.unsqueeze(1)
            z2, _ = self.attn_u2m(q2, k2, v2)             # [B,1,D]
            z = torch.cat([z1, z2], dim=-1).squeeze(1)    # [B,2D]
        else:
            z = z1.squeeze(1)                              # [B,D]
        return self.proj(z)                                # [B,D]

class LinearHead(nn.Module):
    def __init__(self, dim: int, nclass: int):
        super().__init__()
        self.fc = nn.Linear(dim, nclass)
    def forward(self, x): return self.fc(x)


# --------------------- Metrics ---------------------
def macro_auc(model: nn.Module,
              fusion: nn.Module,
              loader: DataLoader,
              nclass: int,
              device: torch.device):
    model.eval(); fusion.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xu, xm, y in loader:
            xu, xm = xu.to(device), xm.to(device)
            prob = torch.softmax(model(fusion(xu, xm)), dim=1).cpu().numpy()
            ys.append(y.numpy()); ps.append(prob)
    y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
    y_bin  = label_binarize(y_true, classes=list(range(nclass)))
    aucs   = [roc_auc_score(y_bin[:, i], y_prob[:, i]) for i in range(nclass)]
    return float(np.mean(aucs)), aucs


# --------------------- Helpers ---------------------
def impute_means(train_df: pd.DataFrame, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    means = train_df[cols].mean()
    out = df.copy()
    out[cols] = out[cols].fillna(means)
    return out

def fit_pca(train_df: pd.DataFrame, cols: List[str], dim: int, seed: int) -> PCA:
    return PCA(dim, random_state=seed).fit(train_df[cols].values)

def transform_pca(df: pd.DataFrame, cols: List[str], pca: PCA) -> np.ndarray:
    return pca.transform(df[cols].values)


# --------------------- CV (per setting) ---------------------
def cv_one_setting(train_df: pd.DataFrame,
                   feat_uni_cols: List[str],
                   feat_mae_cols: List[str],
                   nclass: int,
                   device: torch.device,
                   pca_dim: int,
                   batch_size: int,
                   heads: int,
                   dropout: float,
                   lr: float,
                   wd: float,
                   epochs: int,
                   seed: int = 42,
                   folds: int = 5) -> float:

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []

    for tr_idx, va_idx in kf.split(train_df, train_df["y_enc"]):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)

        # impute by train means
        tr_i = impute_means(tr, tr, feat_uni_cols)
        tr_i = impute_means(tr_i, tr_i, feat_mae_cols)
        va_i = impute_means(tr, va, feat_uni_cols)
        va_i = impute_means(tr, va_i, feat_mae_cols)

        # PCA fit on fold-train only
        p_u = fit_pca(tr_i, feat_uni_cols, pca_dim, seed)
        p_m = fit_pca(tr_i, feat_mae_cols, pca_dim, seed)

        Xu_tr = transform_pca(tr_i, feat_uni_cols, p_u)
        Xm_tr = transform_pca(tr_i, feat_mae_cols, p_m)
        y_tr  = tr_i["y_enc"].values

        Xu_va = transform_pca(va_i, feat_uni_cols, p_u)
        Xm_va = transform_pca(va_i, feat_mae_cols, p_m)
        y_va  = va_i["y_enc"].values

        # class weights & sampler
        cw = compute_class_weight("balanced", classes=np.arange(nclass), y=y_tr)
        samp_w = np.array([1.0/float(cw[int(c)]) for c in y_tr], dtype=np.float32)
        sampler = WeightedRandomSampler(samp_w, num_samples=len(y_tr), replacement=True)

        ld_tr = DataLoader(WsiDS(Xu_tr, Xm_tr, y_tr), batch_size=batch_size, sampler=sampler, drop_last=True)
        ld_va = DataLoader(WsiDS(Xu_va, Xm_va, y_va), batch_size=256, shuffle=False)

        fusion = CrossAttentionFusion(dim=pca_dim, heads=heads, p_drop=dropout, bidirectional=True).to(device)
        head   = LinearHead(pca_dim, nclass).to(device)
        opt    = torch.optim.AdamW(list(fusion.parameters())+list(head.parameters()), lr=lr, weight_decay=wd)
        lossfn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

        fusion.train(); head.train()
        for _ in range(epochs):
            for xu, xm, y in ld_tr:
                xu, xm, y = xu.to(device), xm.to(device), y.to(device)
                loss = lossfn(head(fusion(xu, xm)), y)
                opt.zero_grad(); loss.backward(); opt.step()

        macro, _ = macro_auc(head, fusion, ld_va, nclass, device)
        scores.append(macro)

    return float(np.mean(scores))


# --------------------- Final train (early stop within train) ---------------------
def final_train(train_df: pd.DataFrame,
                internal_df: pd.DataFrame,
                test_df: pd.DataFrame,
                feat_uni_cols: List[str],
                feat_mae_cols: List[str],
                nclass: int,
                device: torch.device,
                pca_dim: int,
                batch_size: int,
                heads: int,
                dropout: float,
                lr: float,
                wd: float,
                max_epochs: int,
                early_stop: int,
                seed: int = 42):

    # impute using the entire train set means
    train_i = impute_means(train_df, train_df, feat_uni_cols)
    train_i = impute_means(train_df, train_i,  feat_mae_cols)
    internal_i = impute_means(train_df, internal_df, feat_uni_cols)
    internal_i = impute_means(train_df, internal_i,  feat_mae_cols)
    test_i = impute_means(train_df, test_df, feat_uni_cols)
    test_i = impute_means(train_df, test_i,  feat_mae_cols)

    # PCA fit on full train
    p_u = fit_pca(train_i, feat_uni_cols, pca_dim, seed)
    p_m = fit_pca(train_i, feat_mae_cols, pca_dim, seed)

    # split 9:1 within train for early stopping
    tr_sub, va_sub = train_test_split(train_i, test_size=0.1, stratify=train_i["y_enc"], random_state=seed)

    def TF(df_):
        return (transform_pca(df_, feat_uni_cols, p_u),
                transform_pca(df_, feat_mae_cols, p_m),
                df_["y_enc"].values)

    Xu_tr, Xm_tr, y_tr = TF(tr_sub)
    Xu_va, Xm_va, y_va = TF(va_sub)
    Xu_in, Xm_in, y_in = TF(internal_i)
    Xu_te, Xm_te, y_te = TF(test_i)

    cw = compute_class_weight("balanced", classes=np.arange(nclass), y=y_tr)
    samp_w = np.array([1.0/float(cw[int(c)]) for c in y_tr], dtype=np.float32)
    sampler = WeightedRandomSampler(samp_w, num_samples=len(y_tr), replacement=True)

    ld_tr = DataLoader(WsiDS(Xu_tr, Xm_tr, y_tr), batch_size=batch_size, sampler=sampler, drop_last=True)
    ld_va = DataLoader(WsiDS(Xu_va, Xm_va, y_va), batch_size=256)
    ld_in = DataLoader(WsiDS(Xu_in, Xm_in, y_in), batch_size=256)
    ld_te = DataLoader(WsiDS(Xu_te, Xm_te, y_te), batch_size=256)

    fusion = CrossAttentionFusion(dim=pca_dim, heads=heads, p_drop=dropout, bidirectional=True).to(device)
    head   = LinearHead(pca_dim, nclass).to(device)
    opt    = torch.optim.AdamW(list(fusion.parameters())+list(head.parameters()), lr=lr, weight_decay=wd)
    lossfn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    best_val, wait, best_state = -1.0, early_stop, None
    for ep in range(max_epochs):
        fusion.train(); head.train()
        for xu, xm, y in ld_tr:
            xu, xm, y = xu.to(device), xm.to(device), y.to(device)
            loss = lossfn(head(fusion(xu, xm)), y)
            opt.zero_grad(); loss.backward(); opt.step()

        val_macro, _ = macro_auc(head, fusion, ld_va, nclass, device)
        if val_macro > best_val:
            best_val, wait = val_macro, early_stop
            best_state = (head.state_dict(), fusion.state_dict())
        else:
            wait -= 1
            if wait == 0:
                break

    if best_state is not None:
        head.load_state_dict(best_state[0]); fusion.load_state_dict(best_state[1])

    int_macro, int_aucs = macro_auc(head, fusion, ld_in, nclass, device)
    ext_macro, ext_aucs = macro_auc(head, fusion, ld_te, nclass, device)

    return {
        "val_macro_best": float(best_val),
        "internal_macro": float(int_macro),
        "external_macro": float(ext_macro),
        "internal_aucs": [float(a) for a in int_aucs],
        "external_aucs": [float(a) for a in ext_aucs],
    }


# --------------------- Main ---------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 1) Load & merge
    df, feat_uni_cols, feat_mae_cols = load_and_merge(UNI_XLS, MAE_XLS)

    # 2) Pre-defined splits (train/internal/test)
    df_train = df[df["Group"] == "train"].reset_index(drop=True)
    df_val   = df[df["Group"] == "internal"].reset_index(drop=True)  # evaluation only
    df_test  = df[df["Group"] == "test"].reset_index(drop=True)

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("Please ensure Group contains 'train', 'internal', and 'test' rows.")

    le = LabelEncoder().fit(df_train["Label1"])
    for D in (df_train, df_val, df_test):
        D["y_enc"] = le.transform(D["Label1"])
    nclass = len(le.classes_)
    print(f"[Info] Classes: {list(le.classes_)} | K={nclass}")

    # 3) Grid search CV on train (no leakage)
    best_cfg, best_cv = None, -1.0
    for P in PCA_GRID:
        for H in HEADS_GRID:
            if P % H != 0:
                print(f"[Skip] PCA={P} not divisible by HEADS={H}.")
                continue
            for BS in BATCH_GRID:
                for DP in DROP_GRID:
                    for LR in LR_GRID:
                        for WD in WD_GRID:
                            for EP in EPCV_GRID:
                                cv_score = cv_one_setting(
                                    df_train, feat_uni_cols, feat_mae_cols, nclass, device,
                                    pca_dim=P, batch_size=BS, heads=H, dropout=DP,
                                    lr=LR, wd=WD, epochs=EP, seed=SEED, folds=5
                                )
                                print(f"[CV] PCA={P}, BS={BS}, H={H}, drop={DP}, "
                                      f"lr={LR}, wd={WD}, ep={EP} → MacroAUC={cv_score:.4f}")
                                if cv_score > best_cv:
                                    best_cv = cv_score
                                    best_cfg = dict(pca=P, heads=H, batch=BS, drop=DP, lr=LR, wd=WD, ep_cv=EP)

    with open(os.path.join(OUT_DIR, "best_unimodal_config.json"), "w") as f:
        json.dump({"cv_macro_auc": best_cv, "best_config": best_cfg}, f, indent=2)
    print(f"\n>>> Best CV Macro AUC={best_cv:.4f} | cfg={best_cfg}")

    # 4) Final train on full train with early stop inside train (9:1), evaluate on internal & test
    final_metrics = final_train(
        train_df=df_train,
        internal_df=df_val,
        test_df=df_test,
        feat_uni_cols=feat_uni_cols,
        feat_mae_cols=feat_mae_cols,
        nclass=nclass,
        device=device,
        pca_dim=best_cfg["pca"],
        batch_size=best_cfg["batch"],
        heads=best_cfg["heads"],
        dropout=best_cfg["drop"],
        lr=best_cfg["lr"],
        wd=best_cfg["wd"],
        max_epochs=MAX_EPOCHS_FINAL,
        early_stop=EARLY_STOP_PATIENCE,
        seed=SEED
    )

    print("\n=== Final Results (Unimodal: UNI+MAE via Cross-Attn) ===")
    print(json.dumps(final_metrics, indent=2))

    with open(os.path.join(OUT_DIR, "final_unimodal_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    main()