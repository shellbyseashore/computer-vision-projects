# Butterfly Image Classification — Simple Neural Network (LA2_Tercero)

A Jupyter notebook that classifies butterfly species from images using a **fully-connected (dense) neural network** — deliberately **without any convolutional layers** (`Conv2D` / `MaxPooling`). It also studies how input image resolution affects the performance of such a network.

- **Notebook:** `LA2_Tercero.ipynb`
- **Dataset:** [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) (Kaggle, `phucthaiv02`)
- **Task:** Multi-class image classification (75 butterfly species)

## Dataset

| Property | Value |
|---|---|
| Labeled images | 5,384 (as listed in `Training_set.csv`) |
| Classes | 75 species |
| Average images per class | ~86.65 |
| Largest class | `MOURNING CLOAK` (131 images) |
| Smallest class | `WOOD SATYR` (71 images) |

Expected folder layout:

```
butterfly/
├── Training_set.csv      # columns: filename, label
└── train/
    ├── Image_1.jpg
    ├── Image_2.jpg
    └── ...
```

Update `DATASET_DIR` in the **Configuration** cell to point to your local copy (the notebook currently uses `/home/farroyo/butterfly`).

> **Note:** Some filenames in the CSV were not present in the local `train/` folder when the notebook was run. Those are skipped automatically (with a message). The reported experiments use the images that were available.

## Notebook Walkthrough

1. **Imports** – NumPy, pandas, Matplotlib, Pillow, scikit-learn, TensorFlow/Keras (GPU disabled for the main model).
2. **Configuration** – paths, image size (64×64), batch size (8), max epochs (30), random seed (42).
3. **Load labels** – reads `Training_set.csv`.
4. **Preview images** – shows 5 random samples with their labels.
5. **Load & preprocess** – resize, convert to RGB, normalize pixels to `[0, 1]`.
6. **Encode labels** – `LabelEncoder` + one-hot encoding; class distribution summary.
7. **Split** – stratified 70% train / 15% validation / 15% test.
8. **Model** – simple dense network (see below).
9. **Compile** – SGD optimizer, categorical cross-entropy.
10. **Train** – with early stopping on validation loss.
11. **Evaluate** – test loss/accuracy and per-class classification report.
12. **Training history** – accuracy/loss curves (saved as `training_history.png`).
13. **Confusion matrix** – saved as `confusion_matrix.png`.
14. **Resolution experiments** – retrains the same architecture at 32×32, 64×64 and 128×128.

## Model Architecture

```
Input (64, 64, 3)
  → Flatten                (12,288)
  → Dense(512, ReLU)
  → Dropout(0.3)
  → Dense(256, ReLU)
  → Dropout(0.3)
  → Dense(75, Softmax)
```

| Setting | Value |
|---|---|
| Optimizer | SGD, learning rate 0.01 |
| Loss | Categorical cross-entropy |
| Batch size | 8 (main model), 4 (resolution experiments) |
| Max epochs | 30 |
| Early stopping | Monitor `val_loss`, patience 5, restore best weights |
| Split | 70 / 15 / 15 (3,768 / 808 / 808 samples), stratified |

## Results

### Main model (64 × 64)

| Metric | Value |
|---|---|
| Test loss | 2.7190 |
| Test accuracy | **29.46%** |

Per-class performance varies widely: some species are recognized reasonably well (e.g., `ADONIS` F1 ≈ 0.61), while others are not recognized at all (e.g., `AMERICAN SNOOT` F1 = 0.00).

### Resolution experiments

The same architecture is retrained from scratch at three resolutions (same seed, same split strategy, same early-stopping rule).

| Resolution | Input Features | Parameters | Train Acc. | Val Acc. | Test Acc. | Training Time |
|---|---|---|---|---|---|---|
| 32 × 32 | 3,072 | 1,723,979 | 0.4379 | 0.3205 | **0.3428** | 6.83 min |
| 64 × 64 | 12,288 | 6,442,571 | 0.2872 | 0.2710 | 0.2871 | 12.21 min |
| 128 × 128 | 49,152 | 25,316,939 | 0.0900 | 0.1126 | 0.1238 | 13.70 min |

### Key takeaways

- **Higher resolution did not help** a dense network. The 32×32 model achieved the best test accuracy (~34%) and trained fastest.
- Flattening images makes the first layer's parameter count grow quickly with resolution (1.7M → 25.3M), which makes the model harder to train with plain SGD and limited data (~70 images per class).
- Dense networks ignore the spatial structure of images, so accuracy is limited. A **CNN or transfer learning** (e.g., a pretrained ResNet/MobileNet) would be the natural next step and would be expected to perform far better.

## Requirements

- Python 3.11
- TensorFlow 2.x (notebook run with 2.21.0)
- numpy, pandas, matplotlib, Pillow, scikit-learn
- Jupyter Notebook / JupyterLab

Install:

```bash
pip install tensorflow numpy pandas matplotlib pillow scikit-learn jupyter
```

## How to Run

1. Download the dataset from Kaggle and extract it so it matches the folder layout above.
2. Edit `DATASET_DIR` in the Configuration cell.
3. Launch Jupyter and run all cells in order:

   ```bash
   jupyter notebook LA2_Tercero.ipynb
   ```

Outputs generated: `training_history.png`, `confusion_matrix.png`.

> Training runs on CPU by default (`CUDA_VISIBLE_DEVICES=-1`). The 128×128 experiment loads images in batches on the fly (via a custom `Sequence`) to keep memory use low.

## Possible Improvements

- Replace the dense network with a CNN or use transfer learning.
- Add data augmentation (flips, rotations, brightness).
- Try Adam instead of SGD, and tune learning rate / layer sizes.
- Use class weights or more data for under-performing species.
