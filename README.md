# acmmm26--DEHFsubmission-code
# DeHF-ECGFlow

Official PyTorch implementation of **DeHF-ECGFlow**, a conditional Flow Matching framework for **single-lead (Lead II) to 12-lead ECG reconstruction**, with a frequency-shaped source prior and high-frequency correction modules.

## Overview

Reconstructing a clinically informative 12-lead ECG from a more accessible single-lead input is an important problem for wearable and resource-limited cardiac monitoring.

This repository provides a conditional Flow Matching implementation that improves both:

- **time-domain morphology fidelity**
- **frequency-domain consistency**, especially in the high-frequency range

The framework includes:

- a **cross-attention 1D U-Net** backbone
- a **GFF-inspired colored Gaussian source prior**
- a **high-frequency correction adapter (HFAdapter)**
- an **adaptive band-splitting module (SplitNet)**

The current codebase uses **Lead II as the condition input**, while reconstructing the **full 12-lead ECG jointly**. The training setup follows a condition/target design where Lead II is min-max normalized to `[-1, 1]` for conditioning, while the full 12-lead target is normalized in a max-abs domain. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Features

- Single-lead to 12-lead ECG reconstruction
- Conditional Flow Matching training
- Frequency-shaped `x0` sampling with GFF-style colored Gaussian prior
- Pre-training PSD inspection for source/target spectral comparison
- Frozen-backbone high-frequency refinement with:
  - `HFAdapter`
  - `SplitNet`
- Flexible checkpoint loading across different save formats :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Repository Structure

```text
.
├── data.py                 # dataset loading and preprocessing
├── dehf_train.py           # baseline Flow Matching training
├── model.py                # model definitions
├── splits_hfadapter.py     # HFAdapter + SplitNet training
└── README.md


DATASET_DIR/
├── II_train.npy
├── II_test.npy
├── eleven_train.npy
└── eleven_test.npy
Expected array shapes:

II_train.npy: (N, T)
II_test.npy: (N, T)
eleven_train.npy: (N, T, 11) or (N, 11, T)
eleven_test.npy: (N, T, 11) or (N, 11, T)

Installation

Tested with Python and PyTorch.

pip install torch torchvision torchaudio numpy matplotlib tqdm
Training
Baseline training
python dehf_train.py
