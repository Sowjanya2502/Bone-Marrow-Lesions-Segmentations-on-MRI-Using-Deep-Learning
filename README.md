# Hybrid BML Segmentation: InceptionResNetV2-UNet with Patch Feature Fusion

A deep learning segmentation model for automated detection of **Bone Marrow Lesions (BML)** in MRI images. The model combines a global U-Net encoder–decoder with a parallel patch-level feature extraction path to capture both large-scale context and fine-grained local detail.

---

## Architecture

```
Input (448×448×3)
   │
   ├──► Global Encoder (InceptionResNetV2)
   │       Skip connections: s1(448), s2(224), s3(112), s4(56), b1(28)
   │
   └──► Patch Encoder (FoldPatches7to28)
           Split image into 4×4 grid of 112×112 patches
           Each patch → InceptionResNetV2 → 7×7 feature map
           Fold 16 tiles → 28×28 patch feature map
   │
   ▼
Concatenate [b1_proj, patch_map_28]
   │
   ▼
SE Recalibration Block (channel-wise attention)
   │
   ▼
Decoder: d1(56) → d2(112) → d3(224) → d4(448)
   │
   ▼
Conv2D(1, sigmoid) → Binary BML Mask (448×448×1)
```

**Key design choices:**
- **Dual-path feature fusion**: The global path captures organ-level context while the patch path preserves fine-grained lesion boundaries.
- **SE block**: Recalibrates the fused feature channels so the network can emphasise the most discriminative features.
- **DiceBCE loss**: Binary Cross-Entropy + soft Dice loss combined, which balances pixel-level accuracy with region-level overlap.

---

## Requirements

```
tensorflow >= 2.8
keras
classification-models          # pip install image-classifiers
opencv-python
scikit-image
scikit-learn
tqdm
matplotlib
pandas
numpy
```

Install with:
```bash
pip install tensorflow keras opencv-python scikit-image scikit-learn tqdm matplotlib pandas image-classifiers
```

---

## Dataset Structure

The code expects BMP-format images with paired masks:

```
<dataset_root>/
├── Train/
│   ├── images/        # e.g., patient001_v001.bmp
│   └── masks/         # e.g., patient001_v001_mask.bmp
├── Validate/
│   ├── images/
│   └── masks/
└── Test/
    ├── images/
    └── masks/
```

Update the path variables at the top of the **Imports & Configuration** cell:

```python
traingdata = "<PATH_TO_YOUR_DATASET>/Train/images"
traingmask  = "<PATH_TO_YOUR_DATASET>/Train/masks"
# ... (see notebook for all six paths)
```

---

## Usage

### 1. GPU Setup
Run **Cell 1** to configure CUDA visibility and enable memory growth for all GPUs. Adjust `CUDA_VISIBLE_DEVICES` to match your hardware.

### 2. Imports & Configuration
Run **Cell 2** to import libraries, set dataset paths, define hyperparameters, and load all model/loss definitions.

Key hyperparameters (edit in Cell 2):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `width / height` | `448` | Input resolution |
| `EPOCHS` | `100` | Maximum training epochs |
| `BATCH_SIZE` | `32` | Mini-batch size |
| `L2W` | `1e-3` | L2 weight decay |

### 3. Training & Evaluation
Run **Cell 4** (Training & Testing) to:
1. Load train/validation data.
2. Build and compile the model with AdamW + DiceBCE loss.
3. Train with early stopping and learning-rate reduction.
4. Generate predictions on the test set.
5. Save per-image and patient-level Dice scores as CSV files.
6. Print final evaluation metrics.

---

## Output Files

| File | Description |
|------|-------------|
| `<output>/predictions/*.jpg` | Predicted binary masks for each test image |
| `<output>/results/per_image_dice.csv` | Dice score for every test image |
| `<output>/results/patient_wise_metrics.csv` | Per-patient avg 2D Dice, 3D volumetric Dice, volume in voxels and mm³ |

---

## Evaluation Metrics

The notebook reports:
- **IoU / Jaccard** — intersection over union
- **Dice coefficient** — both hard (thresholded) and soft
- **Precision** — TP / (TP + FP)
- **Recall / Sensitivity** — TP / (TP + FN)
- **Avg 2D Dice (per patient)** — mean slice-level Dice aggregated by patient
- **Volumetric 3D Dice (per patient)** — 3D overlap computed from the full slice stack
- **Pearson correlation** — predicted vs. ground-truth BML volume (voxels and mm³)

---

## Notebook Structure

| Cell | Content |
|------|---------|
| 1 | Project overview (this README's summary as Markdown) |
| 2 | GPU configuration |
| 3 | Imports, dataset paths, hyperparameters, model/loss definitions |
| 4 | Training & Testing header |
| 5 | Data loading, training loop, prediction, evaluation & reporting |

---

## Citation

If you use this code in your research, please cite the associated paper:

> [Add full citation here once published]

---

## Notes

- The voxel spacing used for volume calculation (0.5 × 0.5 × 3.0 mm) is set in the training cell. Update it to match your MRI acquisition protocol.
- Layer names `activation_203`, `activation_206`, and `activation_276` are hard-coded to the standard TF2/Keras `InceptionResNetV2` implementation. If you update TensorFlow, verify these names still exist before running.
- The notebook is designed for multi-GPU training via `tf.distribute.MirroredStrategy`. It will also run on a single GPU or CPU — just set `CUDA_VISIBLE_DEVICES` accordingly.
