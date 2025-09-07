# -*- coding: utf-8 -*-
"""
mf_bids_classifier
==================
Public API for the MF/BIDs multimodal project.

This package-level __init__ exposes:
- Core models:
    * CrossAttentionFusion (unimodal & multimodal)
    * LinearHead (unimodal head)
    * HeadWithClinic (multimodal head)
- Utilities:
    * set_seed
    * load_and_merge (unimodal)
    * build_encoders / encode_all (multimodal)
- (Optional) WSI feature-extraction helpers if present:
    * process_wsi, aggregate_features  (from WSI_feature_UNI.py)

All imports are optional: if a submodule is missing, we silently skip it.
"""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # unimodal
    "CrossAttentionFusion",
    "LinearHead",
    "set_seed",
    "load_and_merge",
    # multimodal
    "CrossAttentionFusionMM",
    "HeadWithClinic",
    "build_encoders",
    "encode_all",
    # optional feature extraction
    "process_wsi",
    "aggregate_features",
    # helpers
    "get_device",
    "get_available_components",
]

# -------------------------- helpers --------------------------
def _safe_attr(mod, name, alias=None, _globals=None):
    """Safely bind attribute 'name' from module object 'mod' into globals()."""
    if mod is None:
        return
    if hasattr(mod, name):
        (_globals or globals())[alias or name] = getattr(mod, name)

def _try_import(module_name):
    try:
        return import_module(module_name)
    except Exception:
        return None

def get_device(prefer: str = "cuda"):
    """
    Return a torch.device given preference ("cuda" | "cpu").
    Falls back to CPU if CUDA not available.
    """
    try:
        import torch
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        # Torch not installed yet; return a simple string as a hint
        return "cpu"

# --------------------- unimodal (cross_attention_unimodal_model.py) ---------------------
_mod_uni = _try_import("mf_bids_classifier.cross_attention_unimodal_model")

_safe_attr(_mod_uni, "CrossAttentionFusion")   # unimodal fusion
_safe_attr(_mod_uni, "LinearHead")
_safe_attr(_mod_uni, "set_seed")
_safe_attr(_mod_uni, "load_and_merge")

# --------------------- multimodal (cross_attention_multimodal_model.py) -----------------
_mod_multi = _try_import("mf_bids_classifier.cross_attention_multimodal_model")

# Avoid name clash: expose multimodal fusion as CrossAttentionFusionMM
if _mod_multi and hasattr(_mod_multi, "CrossAttentionFusion"):
    CrossAttentionFusionMM = getattr(_mod_multi, "CrossAttentionFusion")
    globals()["CrossAttentionFusionMM"] = CrossAttentionFusionMM

_safe_attr(_mod_multi, "HeadWithClinic")
_safe_attr(_mod_multi, "build_encoders")
_safe_attr(_mod_multi, "encode_all")

# --------------------- optional: WSI UNI feature extractor ------------------------------
# If your file is named differently, adjust the module path below accordingly.
# e.g., "mf_bids_classifier.WSI_feature_UNI"
_mod_wsi_uni = _try_import("mf_bids_classifier.WSI_feature_UNI")
_safe_attr(_mod_wsi_uni, "process_wsi")
_safe_attr(_mod_wsi_uni, "aggregate_features")

# --------------------- discovery utility ---------------------
def get_available_components():
    """
    Return a dict indicating which components were successfully imported.
    This is helpful when developing or running only part of the pipeline.
    """
    return {
        "unimodal": {
            "CrossAttentionFusion": "CrossAttentionFusion" in globals(),
            "LinearHead": "LinearHead" in globals(),
            "set_seed": "set_seed" in globals(),
            "load_and_merge": "load_and_merge" in globals(),
        },
        "multimodal": {
            "CrossAttentionFusionMM": "CrossAttentionFusionMM" in globals(),
            "HeadWithClinic": "HeadWithClinic" in globals(),
            "build_encoders": "build_encoders" in globals(),
            "encode_all": "encode_all" in globals(),
        },
        "feature_extraction": {
            "process_wsi": "process_wsi" in globals(),
            "aggregate_features": "aggregate_features" in globals(),
        },
    }