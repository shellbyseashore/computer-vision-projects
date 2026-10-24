# computer-vision-projects

Hands-on computer vision projects and lab activities. The repository contains two projects that cover both sides of the field: **classical image processing** (filters and edge detection) and **deep learning** (neural-network image classification).

## Repository Structure

```
computer-vision-projects/
├── coin-detection/            # Lab 1: convolution, filtering & edge detection on Philippine coins
│   └── Coin_Detection_Lab_Activity.ipynb
├── butterfly-classificaton/   # Lab 2: butterfly species classification with a dense neural network
│   └── LA2_Tercero.ipynb
└── README.md
```

| Project | Type | Tools | Headline result |
|---|---|---|---|
| [Coin Detection](#1-coin-detection--convolution-filtering-and-edge-detection) | Classical image processing | OpenCV, NumPy, Matplotlib | Clean, continuous edges and readable denominations for coins totaling ₱21.10 |
| [Butterfly Classification](#2-butterfly-image-classification--simple-neural-network) | Deep learning (dense ANN, no CNN) | TensorFlow/Keras, scikit-learn | 29.5% test accuracy on 75 classes (best: 34.3% at 32×32) |

---

## 1. Coin Detection — Convolution, Filtering and Edge Detection

**Notebook:** `coin-detection/Coin_Detection_Lab_Activity.ipynb`

### Objective
Use convolution, spatial filtering, and edge detection to process images of Philippine coins so that the coins forming **₱21.10** can be clearly identified, with clean edges and recognizable denominations.

### Selected Coins

| Coin | Value |
|---|---|
| 10 Sentimo (2009 series) | ₱0.10 |
| 1 Piso (Jose Rizal, 2010 series) | ₱1.00 |
| 20 Piso (bi-metallic series) | ₱20.00 |
| **Total** | **₱21.10** |

### Methodology

The pipeline runs in seven stages. Plain Canny edge detection on raw coin photos produces broken, noisy outlines because of metal grain, glare, and shadows, so most of the work goes into preparing the image before edge detection and cleaning up afterward.

1. **Load and combine** – the three coin images are resized to a common height (650 px) and placed side by side on a labeled canvas.
2. **Pre-processing**
   - Grayscale conversion (color isn't needed for edges).
   - *Illumination flattening*: subtract a heavily blurred copy (σ = 40) to remove glare and left-to-right brightness changes. This brings out faint engravings such as the *20* and *PISO* on the bi-metallic coin.
   - Median filter (5×5) to remove speckle noise while preserving edges.
   - Bilateral filter + Gaussian blur to smooth metal grain while keeping strong edges.
3. **Convolution (sharpening)** – a 3×3 kernel re-emphasizes the engravings softened by blurring. Its weights sum to 1, so flat regions stay unchanged.

   ```
   [ 0 -1  0 ]
   [-1  5 -1 ]
   [ 0 -1  0 ]
   ```
4. **Edge detection** – two methods are compared on the sharpened image:
   - **Sobel**: horizontal and vertical gradients combined into an edge-strength image.
   - **Canny**: gradient + non-maximum suppression + hysteresis thresholding. Thresholds are *adaptive*: the high threshold is the 87th percentile of Sobel magnitude inside the coins, and the low threshold is 35% of it. Coins are located with a Hough circle transform and masked so background edges are ignored.
5. **Clean the result**
   - Connected-component filtering removes stray edge fragments (< 130 px).
   - Morphological closing (5×5) joins small gaps.
   - Gap linking connects loose line ends (within ~35 px) so numerals become continuous outlines.
   - A dedicated pass for the low-contrast *20 Piso* numerals using CLAHE and a black-hat filter.
   - Each coin's outer rim is drawn as one continuous circle, and the edges are dilated for visibility.
6. **Final output** – cleaned edges are overlaid in green on a faded grayscale copy of the coins, labeled with each denomination and the ₱21.10 total.

### Results

Canny gave thinner, better-localized edges than Sobel, so it was used for the final output. After cleanup, the final image shows continuous outer rims and readable numerals, portraits, and lettering on all three coins. The hardest case was the 20 Piso coin, where the gray digits sit on a gray panel. It needed its own local-contrast pass before the numerals came out clearly.

Intermediate images saved by the notebook:

| File | Stage |
|---|---|
| `01_original_combined.png` | Original labeled coins |
| `02_grayscale_filtered.png` | Flattened lighting + median/bilateral/Gaussian filtering |
| `03_convolution_sharpened.png` | After sharpening convolution |
| `04a_sobel_edges.png` | Sobel gradient magnitude |
| `04b_canny_edges.png` | Canny binary edge map |
| `05_final_edge_detected.png` | Final cleaned output |

---

## 2. Butterfly Image Classification — Simple Neural Network

**Notebook:** `butterfly-classificaton/LA2_Tercero.ipynb`
**Dataset:** [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) (Kaggle, `phucthaiv02`)

### Objective
Classify butterfly images into 75 species using a **fully-connected neural network with no convolutional layers**, and study how input resolution affects such a network.

### Dataset
- 5,384 labeled images across 75 species (about 86.65 images per class).
- Largest class: `MOURNING CLOAK` (131 images). Smallest: `WOOD SATYR` (71 images).

### Methodology
1. **Preprocessing** – images are resized to 64×64, converted to RGB, and normalized to `[0, 1]`. Labels are integer-encoded and then one-hot encoded.
2. **Split** – stratified 70% train / 15% validation / 15% test (3,768 / 808 / 808 samples).
3. **Model** – flattened pixels fed through dense layers:

   ```
   Input (64, 64, 3) → Flatten → Dense(512, ReLU) → Dropout(0.3)
                     → Dense(256, ReLU) → Dropout(0.3) → Dense(75, Softmax)
   ```
4. **Training** – SGD (learning rate 0.01), categorical cross-entropy, up to 30 epochs with early stopping on validation loss (patience 5, best weights restored).
5. **Evaluation** – test loss/accuracy, per-class classification report, training curves, and confusion matrix.
6. **Resolution experiment** – the same architecture retrained from scratch at 32×32, 64×64, and 128×128.

### Results

**Main model (64×64):** test loss 2.7190, test accuracy **29.46%**. Per-class performance varied widely, from reasonably good (e.g. `ADONIS`, F1 ≈ 0.61) to none at all (e.g. `AMERICAN SNOOT`, F1 = 0.00).

**Resolution experiment:**

| Resolution | Input Features | Parameters | Train Acc. | Val Acc. | Test Acc. | Training Time |
|---|---|---|---|---|---|---|
| 32 × 32 | 3,072 | 1,723,979 | 0.4379 | 0.3205 | **0.3428** | 6.83 min |
| 64 × 64 | 12,288 | 6,442,571 | 0.2872 | 0.2710 | 0.2871 | 12.21 min |
| 128 × 128 | 49,152 | 25,316,939 | 0.0900 | 0.1126 | 0.1238 | 13.70 min |

**Takeaway:** higher resolution did *not* help. Flattening makes the first layer's parameter count explode (1.7M → 25.3M) while the data stays small (~70 images per class), so the larger models were harder to train with plain SGD. Dense networks also ignore spatial structure. A CNN or transfer learning would be the natural next step.

---

## Key Takeaways Across Projects

- **Classical methods need careful preparation.** The coin project shows that edge quality depends mostly on pre-processing (lighting correction, denoising) and post-processing (cleanup, gap linking), not just on the edge detector itself.
- **Architecture matters more than input size.** The butterfly project shows that giving a dense network more pixels makes it worse, which motivates convolutional architectures that exploit spatial structure.

## Requirements

- Python 3.11
- `opencv-python`, `numpy`, `matplotlib` (coin detection)
- `tensorflow`, `pandas`, `pillow`, `scikit-learn` (butterfly classification)
- Jupyter Notebook / JupyterLab

```bash
pip install opencv-python numpy matplotlib tensorflow pandas pillow scikit-learn jupyter
```

## How to Run

1. Clone the repository and open a project folder.
2. **Coin detection:** place the coin images (file names containing e.g. `coin_10sentimo`, `coin_1piso`, `coin_20piso`) next to the notebook.
3. **Butterfly classification:** download the Kaggle dataset and set `DATASET_DIR` in the Configuration cell.
4. Run `jupyter notebook` and execute the cells in order.
