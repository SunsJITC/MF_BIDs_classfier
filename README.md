# MF_BIDs_classfier
A multimodal AI classifier for differentiating Mycosis Fungoides (MF) from benign inflammatory dermatoses (BIDs, including Atopic Dermatitis AD, Psoriasis PSO, and Cutaneous Adverse Drug Reaction 
CADR), based on histopathology whole-slide images (WSIs) and clinical features.
## Weights & Models

Our framework builds upon two foundation encoders:

- **UNI** (Universal Pathology Foundation Model)  
  GitHub: [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI)  
  Reference: Chen RJ, et al. *Nature Medicine* (2024).

- **MAE** (Masked Autoencoder for Visual Representation Learning)  
  GitHub: [https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae)  
  Reference: He K, et al. *CVPR* (2022).

We provide wrappers in `src/mfai/models/backbones.py` to load these encoders.  
Due to size and license constraints, **pretrained weights are not stored in this repository**.  
Users should download the official weights from the links above, or use the instructions in `configs/paths.example.yaml` to set local paths.

