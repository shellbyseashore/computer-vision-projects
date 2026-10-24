# Coin Detection Lab Activity

Convolution, spatial filtering, and edge detection on Philippine coins.
**Target amount: ₱21.10** = 10 Sentimo (₱0.10) + 1 Piso (₱1.00) + 20 Piso (₱20.00)

Notebook: `Coin_Detection_Lab_Activity_fixed.ipynb`

## What it does

The notebook loads three coin images, combines them into one labeled image, and processes them so the
coin outlines and the numerals **10**, **1** and **20** show as clear, continuous green lines on a faded
gray copy of the coins.

## Requirements

- Python 3
- `opencv-python`, `numpy`, `matplotlib`
- `scikit-image` (optional, used to thin the **20** outline and trace the **2**; the notebook falls back
  to a simpler method if it is missing)

```
pip install opencv-python numpy matplotlib scikit-image
```

## Input files

Put three coin images in the same folder as the notebook (sub-folders, `/content` and `/mnt/data` are also
searched). The file name must contain one of these:

| Coin | File name contains |
|---|---|
| 10 Sentimo | `coin_10sentimo`, `10sentimo`, `10_sentimo` |
| 1 Piso | `coin_1peso`, `coin_1piso`, `1piso`, `1peso` |
| 20 Piso | `coin_20peso`, `coin_20piso`, `20piso`, `20peso` |

Accepted formats: png, jpg, jpeg, webp, bmp.

## How to run

Run all cells from top to bottom. Cells depend on earlier ones (for example, the edge detection cell needs
the sharpening cell), so do not skip any.

## Pipeline

| Section | Step |
|---|---|
| 1–2 | Setup, load and resize the coins (height 650 px), build the labeled combined image |
| 3 | Pre-processing: grayscale, lighting flattening, median, bilateral and Gaussian filtering |
| 4 | Convolution with the 3×3 sharpening kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` |
| 5 | Edge detection: Sobel (for comparison) and Canny with thresholds taken from the image's own gradient strength; coins found with a circle detector so edges are only searched inside each coin |
| 6 | Clean-up: close small gaps, remove short fragments, join broken line ends on the **10** and **20**, trace the **2**, thicken lines |
| 7 | Final green-on-gray output with labels and the ₱21.10 total |
| 8 | Summary |

## Output files

Written to the notebook's folder:

- `01_original_combined.png` – original coins with labels
- `02_grayscale_filtered.png` – after pre-processing
- `03_convolution_sharpened.png` – after sharpening
- `04a_sobel_edges.png` – Sobel edge strength
- `04b_canny_edges.png` – Canny edge map
- `05_final_edge_detected.png` – final result
