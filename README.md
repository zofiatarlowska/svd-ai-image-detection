# Distinguishing Real from Generative AI Images via Linear Algebra (SVD)

An explainable, zero-parameter mathematical approach to detecting generative AI artifacts using **Singular Value Decomposition (SVD)** and spectral energy distributions — without relying on deep neural networks.

---

## Research Hypothesis
Generative models (diffusion models, GANs) optimize pixel distributions across high-dimensional latent spaces, introducing subtle, uniform regularities into micro-textures. 

**Hypothesis:** AI-generated images exhibit a flatter singular value decay spectrum and higher mid-to-tail singular values compared to real photographs, where energy is naturally concentrated in low-rank dominant components due to physical camera sensors and asymmetric lighting.

---

## Mathematical Formulation

Each grayscale image is represented as a matrix $A \in \mathbb{R}^{m \times n}$. By the Spectral Theorem on symmetric matrices ($A^T A$), the SVD is formulated as:

$$A = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

Where:
- $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$.
- Cumulative Frobenius energy retained by rank-$k$ truncation:
  $$E_k = \frac{\sum_{i=1}^k \sigma_i^2}{\|A\|_F^2} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$

---

## Key Findings

1. **Energy Dispersion:** AI images require significantly more singular values to preserve 95% and 99% of Frobenius energy ($k_{95\%} \approx 18.1$ for AI vs $10.7$ for Real; $k_{99\%} \approx 115.4$ vs $72.4$).
2. **Mid-Spectrum Elevation:** On a logarithmic scale, AI singular values form a distinct elevated shelf across indices 100–800.
3. **Tail Regularity:** Real photographs feature noisy, irregular sensor shot noise in the spectral tail (last 30 SVs), whereas generative models produce smoothed, regularized micro-textures.

---

## The 3-Feature Voting Classifier
Without training parameters or gradient descent, a voting heuristic was formulated using three spectral properties:
1. **Mid-frequency sum:** $\sum_{i=100}^{800} \sigma_i$
2. **Energy threshold rank:** $k$ such that $E_k \ge 0.95$
3. **Spectral tail sum:** $\sum_{i=r-30}^{r} \sigma_i$

### Empirical Performance:
- **Reference dataset:** **90.0% accuracy** (9/10 pairs correct)
- **Unseen test dataset:** **80.0% out-of-sample accuracy** (4/5 pairs correct)

---

## Repository Contents
- `svd_ai_detection.ipynb` — Full research notebook containing mathematical derivations, SVD decomposition, and spectral visualizations.
- `presentation.pdf` — Project slide deck summarizing methodology, spectral theory, and empirical findings.

---

## Tech Stack
- **Python:** NumPy, Matplotlib, SciPy, Pillow
