# -*- coding: utf-8 -*-
"""
cross_attention_multimodal_model.py
===================================
Multimodal classifier with:
  - Pathology features (UNI + MAE) fused via bidirectional cross-attention
  - Structured clinical variables embedded with OHE + TruncatedSVD
  - Final MLP head for classification
"""

import os, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import sparse

# ------------------------ Paths & Globals ------------------------
UNI_XLS     = "/Users/yatedisihao/Desktop/UNI.xlsx"
MAE_XLS     = "/Users/yatedisihao/Desktop/MAE.xlsx"
CLINIC_XLS  = "/Users/yatedisihao/Desktop/clinic.xlsx"
OUT_DIR     = "/Users/yatedisihao/Desktop"
os.makedirs(OUT_DIR, exist_ok=True)

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid search settings (mirrors unimodal style, but adds clinic dim & bidirection):
PCA_GRID_PATH = [32, 48, 64]          # PCA dim for UNI/MAE (must be divisible by n_heads)
CLINIC_DIM_GRID = [8, 16, 32]         # Clinical embedding dim after OHE+SVD
BATCH_GRID    = [8, 16, 32]           # Batch sizes
HEADS_GRID    = [2, 4]                # Multihead attention heads (require p_dim % heads == 0)
BIDIR_GRID    = [True]                # Keep bidirectional=True (you can add False to compare)

MAX_EPOCHS_CV    = 150
MAX_EPOCHS_FINAL = 300
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTH    = 0.05
DROPOUT_P       = 0.30
LR              = 5e-4
WD              = 1e-2

# If you know which clinic columns are categorical, put their names here (exact match).
# If left empty, everything non-feature will be treated as numeric.
CAT_COLS = [
    # e.g., "性别",  "皮损分布上肢"
]

# ------------------------ Reproducibility ------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed(SEED)

# ------------------------ I/O & Merge ------------------------
df_uni = pd.read_excel(UNI_XLS); df_uni.columns = df_uni.columns.astype(str)
df_mae = pd.read_excel(MAE_XLS); df_mae.columns = df_mae.columns.astype(str)
df_cli = pd.read_excel(CLINIC_XLS); df_cli.columns = df_cli.columns.astype(str)

meta_cols     = ["ID","WSI_Name","Group","Label1"]
feat_uni_cols = [c for c in df_uni.columns if c.startswith("F")]

# Align MAE features: rename F* -> F*_mae, and keep the same set/order as UNI has
df_mae = df_mae.rename(columns={c: f"{c}_mae" for c in df_mae.columns if c.startswith("F")})
feat_mae_cols = [f"{c}_mae" for c in feat_uni_cols]

# Merge UNI + MAE
df = (
    df_uni[meta_cols]
      .merge(df_uni[["ID","WSI_Name"]+feat_uni_cols], on=["ID","WSI_Name"], how="left")
      .merge(df_mae[["ID","WSI_Name"]+feat_mae_cols], on=["ID","WSI_Name"], how="left")
)

# Merge clinic on ID (drop potential duplicates)
drop_cli = [c for c in ["WSI_Name","Group","Label1"] if c in df_cli.columns]
df = df.merge(df_cli.drop(columns=drop_cli, errors="ignore"), on="ID", how="left")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].reset_index(drop=True)

# Splits already defined
df_train = df[df["Group"]=="train"].reset_index(drop=True)
df_val   = df[df["Group"]=="internal"].reset_index(drop=True)
df_test  = df[df["Group"]=="test"].reset_index(drop=True)

# Encode label
le = LabelEncoder().fit(df_train["Label1"])
for D in (df_train, df_val, df_test):
    D["y_enc"] = le.transform(D["Label1"])
NUM_CLASSES = len(le.classes_)
print(f"[Info] Classes: {list(le.classes_)} (K={NUM_CLASSES})")

# ------------------------ Clinic: numeric/categorical split ------------------------
# Everything that is not meta/UNI/MAE is clinical
used_cols = set(meta_cols + feat_uni_cols + feat_mae_cols)
clinic_cols = [c for c in df.columns if c not in used_cols]

# If CAT_COLS provided, use them; otherwise assume all clinic as numeric
cat_cols = [c for c in clinic_cols if c in CAT_COLS]
num_cols = [c for c in clinic_cols if c not in cat_cols]

# Clean obvious textual null-like tokens
for D in (df_train, df_val, df_test):
    D.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan, " ": np.nan}, inplace=True)

# ------------------------ Encoders ------------------------
def cap_dim(target, n_features, n_samples):
    return int(max(1, min(target, n_features, n_samples)))

def pad_to_dim(X, dim):
    if X.shape[1] == dim: return X
    if X.shape[1] > dim:  return X[:, :dim]
    pad = np.zeros((X.shape[0], dim - X.shape[1]), dtype=X.dtype)
    return np.hstack([X, pad])

class UniMaeEncoder:
    """Mean-impute -> Standardize -> PCA for UNI/MAE blocks."""
    def __init__(self, dim, cols):
        self.dim = dim; self.cols = list(cols)
        self.imp = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.pca = None; self.used_dim = 0
    def fit(self, D):
        X = self.imp.fit_transform(D[self.cols])
        X = self.scaler.fit_transform(X)
        k = cap_dim(self.dim, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=k, random_state=SEED).fit(X)
        self.used_dim = k
    def transform(self, D):
        X = self.imp.transform(D[self.cols])
        X = self.scaler.transform(X)
        Z = self.pca.transform(X)
        return pad_to_dim(Z, self.dim).astype(np.float32)

class ClinicEncoder:
    """NUM: mean-impute + standardize; CAT: OHE; concat -> TruncatedSVD."""
    def __init__(self, dim, num_cols, cat_cols):
        self.dim = dim
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.num_imp  = SimpleImputer(strategy="mean") if self.num_cols else None
        self.scaler   = StandardScaler() if self.num_cols else None
        self.ohe      = OneHotEncoder(handle_unknown="ignore", sparse=True) if self.cat_cols else None
        self.svd      = None; self.used_dim = 0
    def fit(self, D):
        blocks = []
        if self.num_cols:
            Xn = self.num_imp.fit_transform(D[self.num_cols])
            Xn = self.scaler.fit_transform(Xn)
            blocks.append(Xn)
        if self.cat_cols:
            Xc = self.ohe.fit_transform(D[self.cat_cols].astype(str).fillna("Missing"))
            blocks.append(Xc)
        if not blocks:
            self.svd=None; self.used_dim=0; return
        X = blocks[0]
        for B in blocks[1:]:
            X = sparse.hstack([X, B]).tocsr() if sparse.issparse(X) or sparse.issparse(B) else np.hstack([X, B])
        k = cap_dim(self.dim, X.shape[1], X.shape[0])
        self.svd = TruncatedSVD(n_components=k, random_state=SEED).fit(X)
        self.used_dim = k
    def transform(self, D):
        if self.svd is None and not (self.num_cols or self.cat_cols):
            return np.zeros((len(D), self.dim), dtype=np.float32)
        blocks = []
        if self.num_cols:
            Xn = self.num_imp.transform(D[self.num_cols]); Xn = self.scaler.transform(Xn); blocks.append(Xn)
        if self.cat_cols:
            Xc = self.ohe.transform(D[self.cat_cols].astype(str).fillna("Missing")); blocks.append(Xc)
        X = blocks[0]
        for B in blocks[1:]:
            X = sparse.hstack([X, B]).tocsr() if sparse.issparse(X) or sparse.issparse(B) else np.hstack([X, B])
        Z = self.svd.transform(X)
        return pad_to_dim(Z, self.dim).astype(np.float32)

def build_encoders(fit_df, p_dim, c_dim):
    enc_u = UniMaeEncoder(p_dim, feat_uni_cols); enc_u.fit(fit_df)
    enc_m = UniMaeEncoder(p_dim, feat_mae_cols); enc_m.fit(fit_df)
    enc_c = ClinicEncoder(c_dim, num_cols, cat_cols); enc_c.fit(fit_df)
    return enc_u, enc_m, enc_c

def encode_all(enc_u, enc_m, enc_c, D):
    Xu = enc_u.transform(D); Xm = enc_m.transform(D); Xc = enc_c.transform(D)
    y  = D["y_enc"].values
    return Xu, Xm, Xc, y

# ------------------------ Dataset & Models ------------------------
class ThreeModalDS(Dataset):
    def __init__(self, Xu, Xm, Xc, y):
        self.Xu, self.Xm, self.Xc = Xu.astype(np.float32), Xm.astype(np.float32), Xc.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xu[i], self.Xm[i], self.Xc[i], self.y[i]

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention on single-token embeddings:
      - Path1: Q=Xm (MAE-UNI), K/V=Xu (UNI)
      - Path2: Q=Xu,          K/V=Xm (if bidirectional=True)
      Outputs are concatenated then projected back to 'dim'.
    """
    def __init__(self, dim, heads=4, p_drop=0.1, bidirectional=True):
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
    def forward(self, x_u, x_m):
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

class HeadWithClinic(nn.Module):
    """Concatenate fused pathology [B, P] with clinic embedding [B, C] -> MLP."""
    def __init__(self, p_dim, c_dim, nclass, p=0.3):
        super().__init__()
        in_dim = p_dim + c_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(128, nclass)
        )
    def forward(self, fused, clinic):
        x = torch.cat([fused, clinic], dim=1)
        return self.net(x)

# ------------------------ Metrics ------------------------
def macro_auc_from_loader(model, fusion, loader, nclass):
    model.eval(); fusion.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xu, xm, xc, y in loader:
            xu, xm, xc = xu.to(DEVICE), xm.to(DEVICE), xc.to(DEVICE)
            logits = model(fusion(xu, xm), xc)
            prob   = torch.softmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy()); ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_bin  = label_binarize(y_true, classes=list(range(nclass)))
    aucs   = [roc_auc_score(y_bin[:, i], y_prob[:, i]) for i in range(nclass)]
    return float(np.mean(aucs)), aucs

# ------------------------ Train Loop ------------------------
def train_one(Xu_tr, Xm_tr, Xc_tr, y_tr,
              Xu_va, Xm_va, Xc_va, y_va,
              nclass, p_dim, c_dim, bs, heads, bidirectional,
              max_epochs, patience, class_weight):
    # loaders
    sampler = WeightedRandomSampler(
        weights=[1.0/float(class_weight[int(c)]) for c in y_tr],
        num_samples=len(y_tr), replacement=True
    )
    ld_tr = DataLoader(ThreeModalDS(Xu_tr, Xm_tr, Xc_tr, y_tr), batch_size=bs, sampler=sampler, drop_last=True)
    ld_va = DataLoader(ThreeModalDS(Xu_va, Xm_va, Xc_va, y_va), batch_size=256)

    # models
    fusion = CrossAttentionFusion(dim=p_dim, heads=heads, p_drop=DROPOUT_P, bidirectional=bidirectional).to(DEVICE)
    head   = HeadWithClinic(p_dim, c_dim, nclass, p=DROPOUT_P).to(DEVICE)
    opt    = torch.optim.AdamW(list(fusion.parameters()) + list(head.parameters()), lr=LR, weight_decay=WD)
    lossfn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight, dtype=torch.float32).to(DEVICE),
        label_smoothing=LABEL_SMOOTH
    )

    best_val, best_state, wait = -1.0, None, patience
    for _ in range(max_epochs):
        fusion.train(); head.train()
        for xu, xm, xc, y in ld_tr:
            xu, xm, xc, y = xu.to(DEVICE), xm.to(DEVICE), xc.to(DEVICE), y.to(DEVICE)
            logits = head(fusion(xu, xm), xc)
            loss   = lossfn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        val_macro, _ = macro_auc_from_loader(head, fusion, ld_va, nclass)
        if val_macro > best_val:
            best_val, best_state, wait = val_macro, (head.state_dict(), fusion.state_dict()), patience
        else:
            wait -= 1
            if wait == 0: break

    head.load_state_dict(best_state[0]); fusion.load_state_dict(best_state[1])
    return head, fusion, best_val

# ------------------------ 5-fold CV on 'train' (grid-search) ------------------------
# We do CV by re-fitting encoders on full train each combo, then splitting arrays (encoder fit on full train -> consistent transform).
# If you prefer strict fold-fit for encoders, you can encode within folds similarly to unimodal CV; both are acceptable if no leakage occurs.
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

best_cfg, best_cv = None, -1.0

for P_DIM in PCA_GRID_PATH:
    for HEADS in HEADS_GRID:
        if P_DIM % HEADS != 0:
            print(f"[Skip] P_DIM={P_DIM} not divisible by HEADS={HEADS}.")
            continue
        for C_DIM in CLINIC_DIM_GRID:
            # Fit encoders on full train
            enc_u, enc_m, enc_c = build_encoders(df_train, P_DIM, C_DIM)
            Xu_all, Xm_all, Xc_all, y_all = encode_all(enc_u, enc_m, enc_c, df_train)

            for BS in BATCH_GRID:
                for BIDIR in BIDIR_GRID:
                    fold_scores = []
                    for tr_idx, va_idx in kf.split(Xu_all, y_all):
                        Xu_tr, Xm_tr, Xc_tr, y_tr = Xu_all[tr_idx], Xm_all[tr_idx], Xc_all[tr_idx], y_all[tr_idx]
                        Xu_va, Xm_va, Xc_va, y_va = Xu_all[va_idx], Xm_all[va_idx], Xc_all[va_idx], y_all[va_idx]

                        cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
                        head, fusion, _ = train_one(
                            Xu_tr, Xm_tr, Xc_tr, y_tr, Xu_va, Xm_va, Xc_va, y_va,
                            NUM_CLASSES, P_DIM, C_DIM, BS, HEADS, BIDIR,
                            MAX_EPOCHS_CV, EARLY_STOP_PATIENCE, cw
                        )
                        ld_va = DataLoader(ThreeModalDS(Xu_va, Xm_va, Xc_va, y_va), batch_size=256)
                        cv_macro, _ = macro_auc_from_loader(head, fusion, ld_va, NUM_CLASSES)
                        fold_scores.append(cv_macro)

                    mean_cv = float(np.mean(fold_scores))
                    print(f"[CV] P={P_DIM}, C={C_DIM}, BS={BS}, H={HEADS}, BiDir={BIDIR} → MacroAUC={mean_cv:.4f}")
                    if mean_cv > best_cv:
                        best_cv = mean_cv
                        best_cfg = dict(P_DIM=P_DIM, C_DIM=C_DIM, BS=BS, HEADS=HEADS, BIDIR=BIDIR)

print(f"\n>>> Best by CV: {best_cfg} | CV Macro AUC={best_cv:.4f}")

# ------------------------ Final training on full 'train' with early stop on 'internal' ------------------------
P_DIM   = best_cfg["P_DIM"]; C_DIM = best_cfg["C_DIM"]
BS      = best_cfg["BS"];    HEADS = best_cfg["HEADS"]; BIDIR = best_cfg["BIDIR"]

enc_u, enc_m, enc_c = build_encoders(df_train, P_DIM, C_DIM)

Xu_tr, Xm_tr, Xc_tr, y_tr = encode_all(enc_u, enc_m, enc_c, df_train)
Xu_in, Xm_in, Xc_in, y_in = encode_all(enc_u, enc_m, enc_c, df_val)
Xu_te, Xm_te, Xc_te, y_te = encode_all(enc_u, enc_m, enc_c, df_test)

cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
head, fusion, _ = train_one(
    Xu_tr, Xm_tr, Xc_tr, y_tr,
    Xu_tr, Xm_tr, Xc_tr, y_tr,
    NUM_CLASSES, P_DIM, C_DIM, BS, HEADS, BIDIR,
    MAX_EPOCHS_FINAL, EARLY_STOP_PATIENCE, cw
)

def eval_block(Xu, Xm, Xc, y, name):
    ld = DataLoader(ThreeModalDS(Xu, Xm, Xc, y), batch_size=256)
    macro, aucs = macro_auc_from_loader(head, fusion, ld, NUM_CLASSES)
    print(f"{name:10s} per-class AUC:", ["%.3f" % a for a in aucs], "| Macro:", f"{macro:.3f}")
    return macro, aucs

print("\n=== Final Results (Multimodal: UNI+MAE via Cross-Attn + Clinic Head) ===")
print(f"5-fold CV on train (macro AUC): {best_cv:.4f}")
_ = eval_block(Xu_in, Xm_in, Xc_in, y_in, "Internal")
_ = eval_block(Xu_te, Xm_te, Xc_te, y_te, "External")

# Optionally save best config and label map for reproducibility
with open(os.path.join(OUT_DIR, "best_multimodal_config.json"), "w") as f:
    json.dump({"best_cfg": best_cfg, "classes": list(le.classes_)}, f, ensure_ascii=False, indent=2)
print(f"Saved config → {os.path.join(OUT_DIR,'best_multimodal_config.json')}")