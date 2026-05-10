# SOLUTION.md

## Reproducibility Instructions

### Environment

Python 3.8+, dependencies:

```bash
pip install numpy scipy gdown
```

### Running

```bash
python applicant_solution.py
```

The script automatically downloads `challenge.mat` from Google Drive, runs the baseline and my solution, and writes `results.json`.

### Result

```json
{
  "baseline": { "average_db": 4.02 },
  "yours":    { "average_db": 9.71 }
}
```

Per channel: ch0 — 9.24 dB, ch1 — 10.99 dB, ch2 — 10.73 dB, ch3 — 7.88 dB.

---

## Final Solution

### Interference Model

The received signal is described as:

```
rx[n, c] = s[n, c] + F_c(TX[n]) + E[n, c] + η[n, c]
```

where `F_c(TX)` is the nonlinear TX-driven component, and `E[n, c] = a[n] · b[c]` is the external rank-1 interference (the same temporal waveform across all four receive channels, with different amplitudes). The baseline estimates only `F_c` and achieves ~4 dB. My solution additionally estimates and subtracts `E`.

### fit_tx_prediction

The function models the TX nonlinearity using third-order intermodulation products. Physically, such products arise when two signals on different carriers pass through a nonlinear amplifier — they generate spurious tones at frequencies `2f_A − f_B` and `2f_B − f_A` that fall within the receive band.

Specifically, 10 cross-terms between pairs of TX channels are used:

```
tx[:,0]² · conj(tx[:,1])   and   tx[:,1]² · conj(tx[:,0])   # High-A × High-B
tx[:,0]² · conj(tx[:,3])   and   tx[:,3]² · conj(tx[:,0])   # High-A × Medium-B
tx[:,1]² · conj(tx[:,2])   and   tx[:,2]² · conj(tx[:,1])   # High-B × Medium-A
tx[:,2]² · conj(tx[:,3])   and   tx[:,3]² · conj(tx[:,2])   # Medium-A × Medium-B
tx[:,0]² · conj(tx[:,5])   and   tx[:,5]² · conj(tx[:,0])   # High-A × Low-B
```

Each of the 10 features is taken with 13 time lags (−6 … +6 samples), giving 130 columns in total. Coefficients are found by solving the normal equations with ridge regularisation `λ = 1e-6`. The prediction is built in the scoring band at 1.9 ± 0.3 MHz and then reconstructed into the full-length time-domain signal accounting for the lags.

It is important to understand that `fit_tx_prediction` is a projector onto the space of TX features. It can only explain what is linearly expressible through these 130 basis functions. Feeding it anything unrelated to TX produces spurious correlations at the output. This became the key lesson of attempt 1.

### Final Solution Architecture: Four Stages

**Stage 1 — TX nonlinearity.**  
`fit_tx_prediction(rx)` estimates `F̂_c` for each of the 4 receive channels and subtracts it from `rx`. The residual `R1 = rx − F̂_c` contains the external interference `E` and noise.

**Stage 2 — First rank-1 pass.**  
`R1` is passed through `score_filter` (a Blackman-windowed bandpass filter, 2047 taps, band 1.9 ± 0.3 MHz). A 4×4 spatial covariance matrix is built from the filtered matrix and its principal eigenvector `v1` is extracted. It defines the spatial signature of the external interference — the way the same source `a[n]` is distributed across channels with different amplitudes `b[c]`.

Interference temporal waveform: `s1 = R1_band @ v1`.  
Projection onto each channel: `Ê1[:, c] = (⟨s1, R1_band[:, c]⟩ / ‖s1‖²) · s1`

**Stage 3 — Second rank-1 pass with grid search.**  
The procedure is repeated on the residual `R2 = R1 − Ê1`. The TX estimation in stage 1 slightly masks part of the external interference (both procedures operate in the same band and partially overlap), so a single pass does not capture `E` completely.

A second projection `Ê2` is extracted. The subtraction coefficient `α` is chosen by sweeping the grid `[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]` — minimising the residual power in the scoring band. `α` is clipped to the range `[0.15, 0.7]`.

It is worth noting that on a weakened residual, PCA starts capturing noise and the desired signal, so partial subtraction is safer.

**Stage 4 — Conditional third pass.**  
The fraction of variance explained by the principal eigenvalue of the residual covariance matrix is checked:

```
dominant_ratio = λ_max / Σλ
```

Under uniform noise with no structure, each of the 4 channels contributes ~25% of the total variance. A threshold of 0.5 means that a single component explains twice as much as expected — a clear sign of remaining spatially coherent structure. If the condition holds, `Ê3` is extracted and subtracted with a fixed coefficient `α3 = 0.3`. A fixed value rather than grid search — because the third pass operates on a very weak residual, and optimisation at that noise level is unstable.

Final formula:

```
rx_hat = rx − F̂_c − Ê1 − α2·Ê2 − [α3·Ê3]
```

### Code

```python
def your_canceller(tx_n, rx):
    fit_tx = helpers["fit_tx_prediction"]
    score_filter = helpers["score_filter"]

    # Stage 1: TX nonlinearity
    tx_pred = fit_tx(rx)
    rx1 = rx - tx_pred

    # Stage 2: first rank-1 pass
    band1 = np.column_stack([score_filter(rx1[:, ch]) for ch in range(4)])
    cov1 = band1.conj().T @ band1 / band1.shape[0]
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    v1 = eigvecs1[:, np.argmax(eigvals1)]
    s1 = band1 @ v1
    norm1 = np.vdot(s1, s1) + 1e-30
    rank1_1 = np.column_stack([
        (np.vdot(s1, band1[:, ch]) / norm1) * s1 for ch in range(4)
    ])
    rx2 = rx1 - rank1_1

    # Stage 3: second rank-1 pass + grid search over alpha
    band2 = np.column_stack([score_filter(rx2[:, ch]) for ch in range(4)])
    cov2 = band2.conj().T @ band2 / band2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    v2 = eigvecs2[:, np.argmax(eigvals2)]
    s2 = band2 @ v2
    norm2 = np.vdot(s2, s2) + 1e-30
    rank1_2 = np.column_stack([
        (np.vdot(s2, band2[:, ch]) / norm2) * s2 for ch in range(4)
    ])

    best_alpha, best_power = 0.3, np.mean(np.abs(band2) ** 2)
    for alpha_test in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        test_power = np.mean(np.abs(band2 - alpha_test * rank1_2) ** 2)
        if test_power < best_power:
            best_power = test_power
            best_alpha = alpha_test

    best_alpha = np.clip(best_alpha, 0.15, 0.7)
    rx_hat = rx2 - best_alpha * rank1_2

    # Stage 4: conditional third pass
    band3 = np.column_stack([score_filter(rx_hat[:, ch]) for ch in range(4)])
    cov3 = band3.conj().T @ band3 / band3.shape[0]
    eigvals3, eigvecs3 = np.linalg.eigh(cov3)
    dominant_ratio = eigvals3[-1] / (np.sum(eigvals3) + 1e-30)

    if dominant_ratio > 0.5:
        v3 = eigvecs3[:, np.argmax(eigvals3)]
        s3 = band3 @ v3
        norm3 = np.vdot(s3, s3) + 1e-30
        rank1_3 = np.column_stack([
            (np.vdot(s3, band3[:, ch]) / norm3) * s3 for ch in range(4)
        ])
        rx_hat = rx_hat - 0.3 * rank1_3

    return rx_hat
```

---

## What Contributed Most to Improving the Metric

Adding rank-1 subtraction to the baseline produced the biggest jump — from 4.02 to 7.01 dB. The second PCA pass with grid search over the coefficient and the conditional third pass contributed another 2.7 dB.

The key architectural decision was to keep the TX estimation path and the external interference estimation path strictly separate. `fit_tx_prediction` operates only on the TX signal; rank-1 estimation operates only on the residual after it. As soon as these paths are mixed, the result drops below the baseline (see attempt 1).

---

## Experiments and Failed Attempts

### Attempt 1: rank-1 through fit_tx (3.63 dB — worse than baseline)

After subtracting the TX component, the residual `R1` should contain the external interference `E`. I filtered `R1` through `score_filter`, extracted the rank-1 component, and then fed it back into `fit_tx_prediction`, trying to "expand" the narrowband estimate to the full time-domain signal.

This does not work for a fundamental reason. `fit_tx_prediction` solves `XW ≈ Y`, where `X` is the matrix of 130 TX features. Feeding external interference `E` into it asks the regression to find `W` such that the TX features explain `E`. But `E` is by definition independent of TX — the regression finds spurious correlations in the noise and adds artefacts to the scoring band. The result dropped to 3.63 dB.

### Attempt 2: direct rank-1 subtraction (7.01 dB)

Removed `fit_tx` from the external interference estimation chain entirely. The residual after the baseline is passed through `score_filter`, PCA on the channel covariance matrix yields the principal eigenvector, and the rank-1 projection is subtracted directly. Gain: +2.99 dB over the baseline.

### Attempt 3: two rank-1 passes with fixed coefficient 0.5 (8.48 dB)

Added a second PCA pass on the already cleaned signal. Hypothetically, `fit_tx` and rank-1 both operate in the same band and their estimates are slightly interdependent, so a single pass does not capture `E` completely. The second pass was subtracted with coefficient 0.5 — conservatively, to avoid overfitting to noise. Gain: +1.47 dB over attempt 2.

### Attempt 4: adaptive coefficient on the second pass (8.66 dB)

Replaced the fixed 0.5 with the formula `0.3 × (residual_power / rank1_power)`, clipped to `[0.25, 0.65]`. The idea was that if the rank-1 component explains a large share of the residual power, subtraction should be more aggressive. The gain was small (+0.18 dB): the formula did not give enough freedom.

### Attempt 5 → final solution (9.71 dB)

Two changes simultaneously:
- grid search over `α ∈ [0.1, 0.7]` instead of the analytical formula — directly minimises residual power in the scoring band, which is equivalent to maximising the metric;
- conditional third pass based on `dominant_ratio`.

Result: 9.71 dB. Two channels crossed the 10 dB mark.

---

## Limitations

Channel 3 consistently shows the worst result (7.88 dB vs. 9–11 dB on the other channels). This is most likely because the share of external interference in that channel is objectively smaller relative to the desired signal and noise — the method hits the SNR floor rather than a modelling limitation.

`task_and_baseline.py` and the dataset structure were not modified. All improvement comes exclusively from processing the residual after `fit_tx_prediction`.
