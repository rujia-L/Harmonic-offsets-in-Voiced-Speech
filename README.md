# Harmonic Offsets in Voiced Speech (Spectral Analysis)

## Overview
Voiced speech is quasi-harmonic: spectral peaks occur near integer multiples of a fundamental frequency **f0**.
In real recordings, harmonic peaks are slightly shifted. This project measures and analyzes these **harmonic offsets** at the frame level, and studies:

1) **Speaker-independent** relationships between offsets and standard acoustic features  
2) **Between-speaker** differences in the *distribution* of offsets (not only mean shifts)

## Harmonic Offsets
For each voiced frame, estimate f0 and the peak frequency of the nth harmonic, fn, then define:

\[
\Delta_n = f_n - n f_0,\quad n=1,\dots,6
\]

Each frame is represented by the offset vector:
\[
\Delta = [\Delta_1,\Delta_2,\dots,\Delta_6]^\top
\]

In modeling, Δ1 is often used as a stable representative target (fewer missing/extreme values).

## Data & Preprocessing (high level)
- Start from continuous speech recordings with multiple speakers (unbalanced number of frames per speaker).
- Extract **voiced, vowel-like** regions and cut into **fixed-length frames** (typically **20 ms**).
- Save each frame as an individual `.wav` file so frames can be audited and re-processed independently.

## Feature Pipeline (MATLAB)
The MATLAB pipeline computes one-row-per-frame features, including:
- Basic quality measures: RMS level, zero-crossing rate, periodicity proxy, HNR-like proxy
- Robust **f0** estimation with octave-error control
- Harmonic peak refinement around each expected location (n f0) and offsets Δ1–Δ6
- Additional spectral descriptors (centroid/spread/tilt, harmonic amplitude ratios, etc.)

Output: a CSV table with metadata, f0 diagnostics, offsets, features, and quality flags.

## Analyses (Notebooks)

### 1) Speaker-independent modeling (Δ1)
Notebook: `delta_n1_Exploratory modeling.ipynb`

- Predict Δ1 using common acoustic measures (e.g., f0, HNR proxy, intensity proxy, spectral descriptors)
- **Speaker-grouped cross-validation** (GroupKFold): no speaker overlap between train/test
- Models explored: OLS baseline, GAM (descriptive nonlinearity), tree ensembles, mixed-effects diagnostics

### 2) Offset distribution differences across speakers (Δ in R^6)
Notebook: `Harmonic offset distribution difference analysis.ipynb`

- Compute per-speaker mean and covariance of Δ, and total variance tr(Σ)
- Use a **cross-fit** protocol to avoid “double dipping” when defining groups
- Tests target multiple distributional aspects:
  - Mean shift (Hotelling’s T²)
  - Covariance/shape (Box’s M)
  - Dispersion (PERMDISP-like permutation test)
  - Speaker-mean geometry (Mahalanobis-distance comparisons)

## Key Findings (summary)
- Standard acoustic features show **limited speaker-independent explanatory power** for Δ1 under strict out-of-speaker evaluation.
- Offset distributions are **strongly speaker-dependent**, with differences more consistently expressed in **dispersion/covariance structure** than in simple mean shifts.

## Repository Contents
- `example_extract vowel frame as single files.ipynb`  
  Extract vowel-like voiced frames and save each frame as a separate `.wav`.
- `run_vowel_frame_feature_pipeline.m`  
  MATLAB pipeline: f0 estimation + Δ1–Δ6 extraction + acoustic feature table export.
- `delta_n1_Exploratory modeling.ipynb`  
  Speaker-independent modeling of Δ1 using acoustic features.
- `Harmonic offset distribution difference analysis.ipynb`  
  Between-speaker distribution tests on Δ (6D).
- `report/` (recommended)  
  Put the final report PDF here (details, figures, diagnostics).

## How to Run (suggested workflow)
1) **Frame extraction**  
   Run `example_extract vowel frame as single files.ipynb` to create per-frame `.wav` files.

2) **Feature extraction (MATLAB)**  
   Run `run_vowel_frame_feature_pipeline.m` to generate a frame-level CSV feature table.

3) **Analysis (Python notebooks)**  
   Open and run:
   - `delta_n1_Exploratory modeling.ipynb`
   - `Harmonic offset distribution difference analysis.ipynb`

> Note: If the original audio data cannot be redistributed, include a `data/README.md` explaining how to obtain it,
> and/or provide a small synthetic/sample dataset that matches the expected schema.

## Notes
- Frames/harmonics with missing or unreliable peak estimates should be flagged/excluded before analyses requiring complete Δ1–Δ6.
- For reproducibility, consider adding:
  - `requirements.txt` / `environment.yml` (Python)
  - a MATLAB `startup.m` or a short “dependencies” section

## References
- Elvander & Jakobsson (2020), defining fundamental frequency for speech signals
- Elvander (2023), optimal transport priors for harmonic models
