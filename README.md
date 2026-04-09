<div align="center">

# Triplet-MHA ViT for Plant Disease Detection

**A novel Vision Transformer framework replacing standard Multi-Head Attention (MHA) with 
Triplet Multi-Head Attention (t-MHA) in each encoder block for fine-grained plant disease 
classification.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Triplet--MHA--ViT-purple)](https://arxiv.org/)

</div>

---

<div align="justify">

## Overview

This repository provides the complete codes of our [work](https://doi.org/10.1016/j.eswa.2025.127743) 
published in Expert systems with applications. In this study, we proposed an inproved ViT network 
which introduces triplet multi-head attention (t-MHA) function within the transformer encoder. 
The t-MHA function includes cascading arrangement of attention units with residual connections, 
enabling the proposed network to progressively refine attention scores across multiple dimensions, 
learning more fine-grain feature representation of the images. Here, we have considered the two 
most important crops, viz., Rice and Apple (aka RicApp dataset), which contribute a lot to the 
overall GDP of this country. The proposed ViT network obtained a classification 
accuracy of 97.99% on unseen test data of the RicApp dataset, outperforming the standard Vision 
Transformer (ViT) by 2.2%. Therefore, the proposed network offers a precise and efficient solution for automated crop 
disease detection, with promising applications in real-time crop monitoring and precision agriculture.
</div>

---

## Novel Contribution

The core architectural innovation is the replacement of the standard MHA encoder sublayer with a **Triplet Multi-Head Attention (t-MHA)** block:

### Standard ViT Encoder Block
```
Input ──► LayerNorm → MHA → Add (residual) → LayerNorm → MLP → Add (residual) ──► Output
```

### Triplet-MHA ViT Encoder Block (Ours)
```
Input ──►  tripet-MHA ──► Output

'triplet-MHA function'

    ──► LayerNorm → MHA-1 → Add (residual) ──────────────────────► x2
          └─► LayerNorm → MHA-2 → Add (residual) ─────────────────► x4
                └─► LayerNorm → MHA-3 → Add (residual) ────────────► x6
                      └─► LayerNorm → MLP (GELU) → Add (residual) ─────► 
```

Each MHA sublayer attends over the full patch sequence, with residual connections preserving gradient flow across all three stages. This allows:

1. **MHA-1** — global patch relationship learning
2. **MHA-2** — refinement of coarse attention maps
3. **MHA-3** — fine-grained disease feature localisation
4. **MLP** — non-linear feature transformation and projection

---

## Architecture

```
Input Image (224 × 224 × 3)
         │
         ▼
┌─────────────────────┐
│   Patch Extract     │  12×12 patches → 324 patches, each 432-dim
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Patch Embedding   │  Linear Projection + Learnable Positional Embeddings
│   (embed_dim = 64)  │  → (batch, 324, 64)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Triplet-MHA Block 1│  (batch, 324, 64)
│  Triplet-MHA Block 2│  
├─────────────────────┤
│  Triplet-MHA Block 3│
│  Triplet-MHA Block 4│ 
├─────────────────────┤
│  Triplet-MHA Block 5│ 
├─────────────────────┤ 
         │
         ▼
 LayerNorm → GlobalAveragePooling1D → Dense(21, softmax)
         │
         ▼
  Class Prediction (21 classes)
```

**Each Triplet-MHA Block internally:**
- heads = 8, key_dim = 64
- MLP hidden units: [256, 64] (i.e. [4×dims, dims])
- Dropout = 0.3 throughout
- GELU activation in MLP

**Graphical Architecture of the Network**

![image](https://ars.els-cdn.com/content/image/1-s2.0-S095741742501365X-gr2_lrg.jpg)

---
## Dataset

In this study, we collected a large number of disease-infected images of Rice & Apple crops 
(aka RicApp dataset) from diverse agro-climatic regions of India. These images were collected 
in the natural field conditions under the supervision of domain experts from agricultural 
universities (Sher-e-Kashmir University of Agricultural Sciences and Technology, Srinagar; 
Bidhan Chandra Krishi Viswavidyalaya, Nadia, and University of Agricultural Sciences, GKVK, Bangalore).
The dataset consists of approximately 7,331 raw images categorized into 10 classes of Apple crop and
around 12,322 images categorized into 11 classes of Rice crop.

![image](https://ars.els-cdn.com/content/image/1-s2.0-S095741742501365X-gr3.jpg)

---

## Experimental Results

Our proposed network, obtained approx. 97.99% accuracy on the test set of the RicApp dataset 
and outperforming the standard Vision Transformer (ViT) by 2.2%.

|     Networks            |     Accuracy    |     Precision    |     Recall    |     F1-Score    |
|-------------------------|-----------------|------------------|---------------|-----------------|
|     InceptionV3         |     96.34%      |     96.22%       |     96.1%     |     96.16%      |
|     Resnet101           |     95.86%      |     96.19%       |     95.75%    |     96.01%      |
|     ResNet151           |     95.73%      |     96.02%       |     95.56%    |     96.12%      |
|     Densenet121         |     93.58%      |     94.47%       |     93.17%    |     93.82%      |
|     Xception            |     96.99%      |     97.2%        |     96.81%    |     97.01%      |
|     MobileNet           |     95.13%      |     95.46%       |     94.88%    |     95.13%      |
|     EfficientNet        |     93.43%      |     93.86%       |     93.18%    |     94.3%       |
|     EfficientNetV2      |     95.62%      |     95.91%       |     95.39%    |     95.65%      |
|     Base ViT            |     95.71%      |     95.97%       |     95.55%    |     95.76%      |
|     DeiT                |     96.05%      |     96.26%       |     95.99%    |     96.12%      |
|     Swin-T              |     94.31%      |     95.53%       |     93.67%    |     94.59%      |
|     MobileViT           |     97.05%      |     97.3%        |     96.77%    |     97.03%      |
|     Proposed Network    |     97.99%      |     97.9%        |     97.94%    |     97.92%      |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mah-iasri/triplet-mha-vit.git
cd triplet-mha-vit
pip install -e .
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\activate           # Windows
```
--- 

### Requirements Summary

| Package | Version |
|---|---|
| tensorflow | ≥ 2.10.0 |
| numpy | ≥ 1.23.0 |
| Pillow | ≥ 9.0.0 |
| matplotlib | ≥ 3.5.0 |
| scikit-learn | ≥ 1.1.0 |
| pandas | ≥ 1.4.0 |
| xlsxwriter | ≥ 3.0.0 |
| pyyaml | ≥ 6.0 |
| tqdm | ≥ 4.64.0 |

---

## Dataset Preparation

The model expects a dataset organised into class sub-folders under each split directory:

```
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image_001.jpg
│   │   └── ...
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   ├── Rice___Blast/
│   ├── Rice___Brown_spot/
│   └── ...   (21 classes total)
├── val/
│   ├── Apple___Apple_scab/
│   └── ...   (21 classes total)
└── test/
    ├── Apple___Apple_scab/
    └── ...   (21 classes total)
```

**Supported image formats:** `.jpg`, `.jpeg`, `.png`

All images are automatically:
- Resized to `224 × 224` pixels
- Converted to RGB (3 channels)
- Normalised to `[0, 1]` float32

Class labels are assigned as integers in **alphabetical order** of sub-folder names.

---

## Configuration

All hyperparameters and paths are controlled from a single file: `configs/config.yaml`.

> **Before training, update `paths.dataset_root`** to your actual dataset location.

```yaml
paths:
  dataset_root: "/your/path/to/dataset"
  train_dir:    "train"
  val_dir:      "val"
  test_dir:     "test"
  results_dir:  "results"

data:
  img_size:   224
  n_classes:  21

model:
  patch_size:              [12, 12]
  embed_dim:               64
  num_heads:               8
  dense_units_multipliers: [4, 1]   # → [256, 64]
  mlp_head_units:          [128]
  dropout:                 0.3

training:
  batch_size:    128
  epochs:        100
  learning_rate: 0.001
  weight_decay:  0.0001

callbacks:
  monitor:        "val_accuracy"
  mode:           "max"
  save_best_only: true
  tensorboard:    true
```

---

## Training

```bash
python scripts/train.py --config configs/config.yaml
```

On each run, a new experiment directory is auto-created (`results/exp0/`, `results/exp1/`, ...) so that no previous experiment is ever overwritten.

**What happens during training:**
1. Dataset loads from `train/`, `val/`, and `test/` directories
2. Model is built and compiled with `Adam` optimizer  and `categorical cross-entropy` loss function
3. `ModelCheckpoint` saves the best model weights to `weights/best_model.hdf5` on the basis of best `val_accuracy`
4. `CSVLogger` streams per-epoch metrics to `train_logs/train_logs.csv` during the training time
5. `TensorBoard` logs are written to `tb_logs/` (disable in config if not needed)
6. After training, the final model is saved as `weights/last_model.hdf5`
7. Training curves (accuracy, loss, precision, recall) are saved as PNG format

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir results/exp0/tb_logs
```

---

## Evaluation

Load any saved `.hdf5` checkpoint and evaluate on the test sets:

```bash
python scripts/evaluate.py \
    --config  configs/config.yaml \
    --weights results/exp0/weights/best_model.hdf5
```

**Output:**
- Prints accuracy, loss, precision, and recall to the console
- Generates a per-class **classification report** (sklearn format)
- Saves `results_best.xlsx` with:
  - Sheet 1 — Full confusion matrix
  - Sheet 2 — Per-class classification report
---

## Results & Outputs

After a completed training run, the experiment folder contains:

```
results/exp0/
├── weights/
│   ├── best_model.hdf5        ← Best val_accuracy checkpoint
│   └── last_model.hdf5        ← Final epoch weights
├── train_logs/
│   ├── train_logs.csv         ← Epoch-wise metrics (loss, acc, precision, recall)
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── precision_curve.png
│   ├── recall_curve.png
│   └── results_best.xlsx      ← Confusion matrix + classification report
└── tb_logs/                   ← TensorBoard event files
```

---

## Hyperparameters

| Parameter | Value          | Description                               |
|---|----------------|-------------------------------------------|
| `img_size` | (224, 224)     | Input image resolution (H = W)            |
| `patch_size` | (8, 8)         | Non-overlapping patch dimensions          |
| `num_patches` | 784            | (224 ÷ 8)² = 24² = 784                    |
| `embed_dim` | 64             | Patch embedding dimension                 |
| `num_heads` | 8              | Attention heads per MHA sublayer          |
| `key_dim` | 64             | Qeury, Key & Value dimensions per head    |
| `dense_units` | [256, 64]      | MLP hidden units (4×`key_dim`, 1×`key_dim`)           |
| `dropout` | 0.3            | Dropout rate in MHA + MLP                 |
| `transformer_blocks` | 5              | Total Triplet-MHA encoder blocks          |
| `batch_size` | 128            | Training batch size                       |
| `epochs` | 100            | Maximum training epochs                   |
| `optimizer` | Adam           | Adaptive learning rate optimiser          |
| `learning_rate` | 0.0001          | Initial Adam learning rate                |
| `weight_decay` | 0.0001         | L2 regularisation (positional embeddings) |
| `loss` | Categorical CE | Multi-class cross-entropy loss            |
| `n_classes` | 21             | Apple + Rice disease categories           |

---



## Citation

If you use this work in your research, please cite:

```bibtex
@article{haque2025enhanced,
  title={An enhanced vision transformer network for efficient and accurate crop disease detection},
  author={Haque, Md Ashraful and Deb, Chandan Kumar and Gole, Pushkar and Karmakar, Sayantani and Dheeraj, Akshay and Shah, Mehraj Ul Din and Dutta, Subrata and Kumar, MK Prasanna and Marwaha, Sudeep},
  journal={Expert Systems with Applications},
  volume={283},
  pages={127743},
  year={2025},
  publisher={Elsevier}
```

Or in plain text:
>Haque, M. A., Deb, C. K., Gole, P., Karmakar, S., Dheeraj, A., Shah, M. U. D., ... 
> & Marwaha, S. (2025). An enhanced vision transformer network for 
> efficient and accurate crop disease detection. 
> *Expert Systems with Applications*, 283, 127743.https://doi.org/10.1016/j.eswa.2025.127743
---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.

---

<div align="center">
  <sub>Developed by <a href="https://github.com/mah-iasri>">Md Ashraful Haque</a> · Agricultural Deep Learning Research Lab</sub>
</div>
