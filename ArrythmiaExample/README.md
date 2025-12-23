# The Heartbeat of Math: Arrhythmia Detection
### Comparative Analysis: Domain-Specific Feature Engineering vs. Deep Learning

![Status](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Tech](https://img.shields.io/badge/TensorFlow-Scikit--Learn-orange)
![Domain](https://img.shields.io/badge/Domain-Biomedical_Engineering-red)

## Summary
This project challenges the industry trend of relying solely on Deep Learning for biological signal processing. It benchmarks a 1D-Convolutional Neural Network (CNN) against a lightweight Logistic Regression classifier that utilizes domain-specific feature engineering (Non-Linear Dynamics and Chaos Theory).

**Core Hypothesis:** A model grounded in physiological principles (Entropy, Poincaré geometry, Signal Morphology) can achieve comparable diagnostic performance to a "Black Box" Neural Network while offering superior interpretability and 1,000x faster training times.

## Data Source and Processing
* **Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (PhysioNet).
* **Specifications:** 48 half-hour excerpts of 2-channel ambulatory ECG recordings at 360 samples per second.
* **Preprocessing Pipeline:**
    * **Noise Removal:** Custom 5-15Hz Bandpass filter implementation.
    * **Peak Detection:** Robust Pan-Tompkins Algorithm implemented from scratch to handle baseline wander.
    * **Windowing:** Signals sliced into 10-second non-overlapping windows.
    * **Labeling Strategy:** "Weakly Supervised" hierarchy. Windows are classified based on the most severe beat present (Hierarchy: Ventricular > Atrial > Conduction Block > Normal).

## Major Libraries
* **Signal Processing:** `wfdb`, `scipy` (Signal and Stats modules)
* **Machine Learning:** `scikit-learn` (Logistic Regression, PCA, Metrics)
* **Deep Learning:** `tensorflow` / `keras` (1D-CNN)
* **Data Manipulation:** `numpy`, `pandas`
* **Visualization:** `matplotlib`, `seaborn`

## Code Structure
The project is contained within a single reproducible script (`arrythmiaml.py`) designed to run end-to-end:

1.  **Config Class:** Centralized configuration for sample rates, window sizes, and paths.
2.  **download_full_dataset():** Automates data ingestion directly from PhysioNet.
3.  **HeartEngineer Class (The Core):**
    * `pan_tompkins_detector`: Robust R-peak detection.
    * `extract_features`: Generates 11 domain features including SD1/SD2 (Chaos), Kurtosis/Skewness (Morphology), and Sample Entropy.
4.  **Model Definitions:**
    * **Engineering Pipeline:** StandardScaler -> Logistic Regression (Multinomial).
    * **Deep Learning Pipeline:** 2-layer 1D-CNN with Batch Normalization and Dropout.
5.  **Visualization Functions:**
    * `plot_chaos_gallery`: Visualizes Poincaré plots for different arrhythmia classes.
    * `visualize_interpretability`: Compares CNN Saliency Maps vs. Engineering Feature Space.

## Results and Evaluations
The study resulted in a comparison between the two approaches across 5 classes (Normal, LBBB, RBBB, PVC, APC).

| Metric | Engineering Model (Logistic Reg) | Deep Learning (1D-CNN) | Winner |
| :--- | :--- | :--- | :--- |
| **Training Time** | **~0.02 Seconds** | ~30.0 Seconds | **Engineering (1500x Faster)** |
| **Interpretability** | **Transparent (Physics-based)** | Opaque (Saliency Maps) | **Engineering** |
| **RBBB Recall** | **0.90** | 0.82 | **Engineering** |
| **PVC Recall** | 0.49 | **0.54** | Deep Learning |
| **Overall Accuracy** | ~83% | ~86% | Deep Learning (Marginal) |

**Key Findings:**
* **Morphology Matters:** Initially, the Engineering model failed on Bundle Branch Blocks (LBBB). Adding statistical moments (Kurtosis) fixed this, as LBBB beats are statistically "flatter" than normal beats.
* **Efficiency:** The Engineering model is lightweight enough to run on ultra-low-power edge devices (e.g., smartwatches) without GPU acceleration.

## Future Work
* **Edge Deployment:** Port the HeartEngineer logic to C++ for embedded microcontroller testing.
* **Cross-Dataset Validation:** Test robustness on the AHA or PTB Diagnostic ECG Database.
* **Hybrid Architecture:** Implement a stack where the Engineering model handles initial screening (for speed) and the CNN handles ambiguous cases (for precision).
