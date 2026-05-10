import json
import gdown
import numpy as np
from scipy.io import loadmat
from task_and_baseline import baseline, build_task_helpers

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    fit_tx = helpers["fit_tx_prediction"]
    score_filter = helpers["score_filter"]

    # stage 1: TX nonlinear
    tx_pred = fit_tx(rx)
    rx1 = rx - tx_pred

    # stage 2: rank-1 external
    band1 = np.column_stack([
        score_filter(rx1[:, ch]) for ch in range(4)
    ])
    cov1 = band1.conj().T @ band1 / band1.shape[0]
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    v1 = eigvecs1[:, np.argmax(eigvals1)]
    s1 = band1 @ v1
    norm1 = np.vdot(s1, s1) + 1e-30
    rank1_1 = np.column_stack([
        (np.vdot(s1, band1[:, ch]) / norm1) * s1
        for ch in range(4)
    ])
    rx2 = rx1 - rank1_1

    # stage 3: second pass rank-1
    band2 = np.column_stack([
        score_filter(rx2[:, ch]) for ch in range(4)
    ])
    cov2 = band2.conj().T @ band2 / band2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    v2 = eigvecs2[:, np.argmax(eigvals2)]
    s2 = band2 @ v2
    norm2 = np.vdot(s2, s2) + 1e-30
    rank1_2 = np.column_stack([
        (np.vdot(s2, band2[:, ch]) / norm2) * s2
        for ch in range(4)
    ])

    # optimal shrinkage via grid search
    best_alpha = 0.3
    best_power = np.mean(np.abs(band2) ** 2)
    for alpha_test in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        rx_test_band = band2 - alpha_test * rank1_2
        test_power = np.mean(np.abs(rx_test_band) ** 2)
        if test_power < best_power:
            best_power = test_power
            best_alpha = alpha_test

    best_alpha = np.clip(best_alpha, 0.15, 0.7)
    rx_hat = rx2 - best_alpha * rank1_2

    # stage 4: light third pass (conditional)
    band3 = np.column_stack([
        score_filter(rx_hat[:, ch]) for ch in range(4)
    ])
    cov3 = band3.conj().T @ band3 / band3.shape[0]
    eigvals3, eigvecs3 = np.linalg.eigh(cov3)
    dominant_ratio = eigvals3[-1] / (np.sum(eigvals3) + 1e-30)

    if dominant_ratio > 0.5:
        v3 = eigvecs3[:, np.argmax(eigvals3)]
        s3 = band3 @ v3
        norm3 = np.vdot(s3, s3) + 1e-30
        rank1_3 = np.column_stack([
            (np.vdot(s3, band3[:, ch]) / norm3) * s3
            for ch in range(4)
        ])
        alpha3 = 0.3
        rx_hat = rx_hat - alpha3 * rank1_3

    return rx_hat


print("=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== My Solution ===")
my_reds, my_avg = helpers["score"](
    rx, your_canceller(tx_n, rx), label="yours"
)

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": my_reds,
        "average_db": my_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)