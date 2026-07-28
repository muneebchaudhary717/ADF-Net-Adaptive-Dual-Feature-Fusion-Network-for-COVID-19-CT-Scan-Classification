# ADF-Net: Adaptive Dual-Feature Fusion Network for COVID-19 CT Scan Classification

ADF-Net is a state-of-the-art deep learning architecture designed for automated identification and classification of COVID-19, Healthy Normal, and Other Lung Diseases from 2D chest CT scans. By using parallel backbone feature extraction, custom trainable weighted adaptive feature fusion, and CBAM (Convolutional Block Attention Module) spatial-channel refinement, ADF-Net delivers robust classification performance suitable for academic publications.

---

## Model Architecture

The architecture consists of the following phases:
1. **Parallel Feature Extraction**: EfficientNet-B0 and DenseNet-121 extract multi-scale features in parallel.
2. **Channel Projection**: Projections scale incoming maps to 512 dimensions using 1x1 convolutions.
3. **Adaptive Weighted Fusion**: A custom trainable layer optimizes the fusion weights ($W_1, W_2$) via backpropagation where $W_1 + W_2 = 1$.
4. **CBAM Attention block**: Channel Attention followed by Spatial Attention refines features before pooling.
5. **Global Pooling & Head**: Global Average Pooling feeds a dual-dense classification layer with Batch Normalization and Dropout to prevent overfitting.

```
Input (224x224x3)
  ├── Branch A: EfficientNetB0 ── Conv2D (1x1, 512) ──┐
  │                                                    ├──> [Adaptive Weight Fusion] ──> [CBAM Attention] ──> [GAP] ──> [Dense Head] ──> Softmax Output
  └── Branch B: DenseNet121    ── Conv2D (1x1, 512) ──┘
```

---

## Folder Structure

```
ADF-Net/
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── __init__.py
│   ├── backbone.py          # Pre-trained backbones & projection layers
│   ├── cbam.py              # CBAM attention blocks (Channel & Spatial)
│   ├── adaptive_fusion.py   # Trainable weight fusion layer
│   └── model.py             # Complete ADF-Net architecture
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py     # CLAHE, resizing, generators
│   ├── metrics.py           # Metrics calculation library
│   └── visualization.py     # Confusion Matrix, ROC curves, Loss/Acc curves
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── predict.py               # Single image prediction
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

---

## Installation & Setup

1. **Clone or copy the directory**:
   Place the project files inside your workspace folder.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Organize your dataset**:
   Organize your dataset inside `dataset/` following this structure:
   ```
   dataset/
   ├── train/
   │   ├── COVID/
   │   ├── Normal/
   │   └── Other/
   ├── validation/
   │   ├── COVID/
   │   ├── Normal/
   │   └── Other/
   └── test/
       ├── COVID/
       ├── Normal/
       └── Other/
   ```

---

## Usage

### 1. Training the Model
Run the following script to train ADF-Net on your dataset. The script automatically applies CLAHE preprocessing, saves the best weights (`best_model.h5`), and plots training history curves:
```bash
python train.py --dataset_dir "dataset" --epochs 30 --batch_size 32 --lr 0.0001
```

### 2. Evaluating the Model
Evaluate the model's performance on the test split. This computes Accuracy, Precision, Recall, F1, Specificity, AUC, and generates publication-quality plots (Confusion Matrix, ROC Curve, and PR Curve):
```bash
python evaluate.py --dataset_dir "dataset" --model_path "best_model.h5" --plots_dir "results"
```

### 3. Predicting on a Single Image
Run inference on a single test CT scan slice:
```bash
python predict.py --image_path "path/to/ct_scan.png" --model_path "best_model.h5" --class_names "COVID-19,Normal,Other"
```
This prints the predicted label and confidence percentages, and saves a dual visualization panel as `prediction_output.png`.

---

## Key Design Decisions & Publication Highlights
* **Adaptive Fusion**: The network decides the importance weights of the two backbones per batch dynamically.
* **Overfitting Prevention**: Global Average Pooling replaces parameter-dense flattening, while Batch Normalization stabilizes optimization.
* **CLAHE Preprocessing**: Converts image space to LAB and equalizes L-channel, mitigating scanning variations between devices.
