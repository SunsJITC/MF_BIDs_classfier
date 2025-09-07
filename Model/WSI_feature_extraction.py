# -*- coding: utf-8 -*-
"""
extract_features.py
===================
General script for extracting WSI-level features using UNI / MAE encoders.

IMPORTANT: To reproduce your paper’s setting, you need to run this script TWICE:
   1) UNI canonical branch: use the official UNI checkpoint (e.g., pytorch_model.bin).
   2) UNI MAE-adapted branch: use the UNI encoder initialized with my MAE-pretrained weights.
      (This is the "MAE branch" in the paper – still a UNI backbone, but loaded with your own MAE ckpt.)

Optionally:
   - If you want to test a raw MAE encoder (not required in the paper), you can also select backbone=MAE.
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import timm
import openslide
from sklearn.decomposition import PCA
from torchvision import transforms


# ------------------------
# Build UNI encoder
# ------------------------
def build_uni(ckpt_path, device="cuda"):
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model(**timm_kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ------------------------
# Build raw MAE encoder
# (⚠️ not used in your paper, only for exploration)
# ------------------------
def build_mae(ckpt_path, device="cuda"):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# ------------------------
# Patch transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


# ------------------------
# Background filter
# ------------------------
def is_background(patch, threshold=210, ratio=0.8):
    gray = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray > threshold)
    return (white_pixels / gray.size) > ratio


# ------------------------
# Process single WSI
# ------------------------
def process_wsi(model, wsi_path, label, device="cuda",
                patch_size=224, step_size=224):
    slide = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.basename(wsi_path)
    width, height = slide.dimensions
    features_list = []

    print(f"[INFO] Processing {wsi_name} | size: {width}x{height} | label: {label}")

    for y in tqdm(range(0, height, step_size), desc=f"Scanning {wsi_name}"):
        for x in range(0, width, step_size):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")

            if is_background(patch):
                continue

            patch_tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(patch_tensor).cpu().numpy().flatten()

            features_list.append([wsi_name, x, y, label] + feature.tolist())

    slide.close()
    return features_list


# ------------------------
# Aggregate patch-level features -> WSI-level
# ------------------------
def aggregate_features(patch_features, method="mean_pooling"):
    if patch_features.size == 0:
        return np.zeros(1536)

    if method == "mean_pooling":
        return np.mean(patch_features, axis=0)
    elif method == "weighted_mean":
        weights = np.linspace(1, 2, patch_features.shape[0])
        return np.average(patch_features, axis=0, weights=weights)
    elif method == "max_pooling":
        return np.max(patch_features, axis=0)
    elif method == "pca":
        pca = PCA(n_components=100)
        reduced = pca.fit_transform(patch_features)
        return np.mean(reduced, axis=0)
    else:
        raise ValueError("Unsupported aggregation method!")


# ------------------------
# Main
# ------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Select backbone
    if args.backbone == "UNI":
        model = build_uni(args.ckpt, device)
    elif args.backbone == "MAE":
        model = build_mae(args.ckpt, device)
    else:
        raise ValueError("backbone must be UNI or MAE")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Process all WSIs
    for label in os.listdir(args.wsi_root):
        wsi_folder = os.path.join(args.wsi_root, label)
        if not os.path.isdir(wsi_folder):
            continue
        wsi_files = [f for f in os.listdir(wsi_folder)
                     if f.lower().endswith((".tiff"))]

        for wsi_file in wsi_files:
            wsi_path = os.path.join(wsi_folder, wsi_file)
            patch_features = process_wsi(model, wsi_path, label, device)

            patch_features = np.array(patch_features)
            if patch_features.ndim == 1:
                patch_features = patch_features.reshape(1, -1)
            numeric_features = patch_features[:, 4:].astype(np.float32)

            wsi_feature = aggregate_features(numeric_features, method=args.agg)

            columns = ["WSI_Name", "Label"] + [f"F{i}" for i in range(len(wsi_feature))]
            df = pd.DataFrame([[wsi_file, label] + list(wsi_feature)], columns=columns)
            df.to_csv(args.out_csv, mode="a", header=not os.path.exists(args.out_csv), index=False)

            print(f"[DONE] {wsi_file} -> appended to {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_root", type=str, required=True,
                    help="Root directory of WSIs, organized by label subfolders.")
    ap.add_argument("--backbone", type=str, choices=["UNI", "MAE"], required=True,
                    help="Encoder backbone: UNI (canonical or MAE-adapted) or raw MAE.")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint file (.bin or .pth).")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Output CSV file for WSI-level features.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device to run on: cuda or cpu.")
    ap.add_argument("--agg", type=str, default="mean_pooling",
                    choices=["mean_pooling", "weighted_mean", "max_pooling", "pca"],
                    help="Aggregation method for patch-level features.")
    args = ap.parse_args()
    main(args)