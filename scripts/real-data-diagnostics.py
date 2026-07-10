import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zuko
from pitcp import PITCP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.contra import CONTRA
from utils.cqr import CQRHyperRectangle
from utils.hpd import HPD
from utils.scp import SCP
from utils.volume import vol_base, vol_contra, vol_cqr, vol_hpd, vol_pitcp

ROOT = Path(__file__).resolve().parents[1]
N_RUNS = 10
QUANTILES = (0.6, 0.7, 0.8, 0.9)


def get_covgap(covered, clusters):
    return np.max(
        [covered[clusters == k].mean() for k in np.unique(clusters)]
    ) - np.min([covered[clusters == k].mean() for k in np.unique(clusters)])


def run(
    X_train,
    y_train,
    X_valtest,
    y_valtest,
    y_pred,
    n_features,
    n_targets,
    batch_size,
):
    half = len(X_valtest) // 2
    three_quarters = half + (len(X_valtest) - half) // 2

    # Compute residuals
    residuals = y_valtest - y_pred

    # Combine train and val for others
    X_train_val = np.concatenate([X_train, X_valtest[:half]], axis=0)
    y_train_val = np.concatenate([y_train, y_valtest[:half]], axis=0)

    # Standardize X
    X_scaler = StandardScaler()
    X_train_val_scaled = X_scaler.fit_transform(X_train_val)
    X_val_scaled = X_scaler.transform(X_valtest[:half])
    X_cal_scaled = X_scaler.transform(X_valtest[half:three_quarters])
    X_test_scaled = X_scaler.transform(X_valtest[three_quarters:])

    # Standardize y
    y_scaler = StandardScaler()
    y_train_val_scaled = y_scaler.fit_transform(y_train_val)
    y_cal_scaled = y_scaler.transform(y_valtest[half:three_quarters])
    y_test_scaled = y_scaler.transform(y_valtest[three_quarters:])

    # Standardize residuals (for PIT-CP and SCP)
    r_scaler = StandardScaler()
    residuals_val_scaled = r_scaler.fit_transform(residuals[:half])
    residuals_cal_scaled = r_scaler.transform(residuals[half:three_quarters])
    residuals_test_scaled = r_scaler.transform(residuals[three_quarters:])

    # Scores (L-infinity norm of scaled residuals)
    scores_val = np.max(np.abs(residuals_val_scaled), axis=1)[:, None]
    scores_cal = np.max(np.abs(residuals_cal_scaled), axis=1)[:, None]
    scores_test = np.max(np.abs(residuals_test_scaled), axis=1)[:, None]

    s_scaler = StandardScaler()
    scores_val_scaled = s_scaler.fit_transform(scores_val).flatten()
    scores_cal_scaled = s_scaler.transform(scores_cal).flatten()
    scores_test_scaled = s_scaler.transform(scores_test).flatten()

    # HPD
    model_hpd = zuko.flows.MAF(
        features=n_targets, context=n_features, hidden_features=(32, 32)
    )
    optimizer_hpd = torch.optim.Adam(model_hpd.parameters(), lr=1e-3)
    hpd = HPD(model_hpd, optimizer_hpd, n_epochs=500, batch_size=batch_size)
    hpd.fit(X_train_val_scaled, y_train_val_scaled)
    hpd.conformalize(X_cal_scaled, y_cal_scaled)

    # CONTRA (uses flow learned in HPD)
    contra = CONTRA(hpd.estimator, batch_size=batch_size)
    contra.conformalize(X_cal_scaled, y_cal_scaled)

    # PIT-CP
    model_pitcp = zuko.flows.SOSPF(
        features=1, context=n_features, hidden_features=(16, 16)
    )
    optimizer_pitcp = torch.optim.Adam(model_pitcp.parameters(), lr=1e-3)
    pitcp = PITCP(model_pitcp, optimizer_pitcp, n_epochs=1000, batch_size=batch_size)
    pitcp.fit(X_val_scaled, scores_val_scaled)
    pitcp.conformalize(X_cal_scaled, scores_cal_scaled)

    # K-Means diagnostics
    clusters = KMeans(n_clusters=10, random_state=42).fit_predict(X_test_scaled)

    results = {}
    for q in QUANTILES:
        results_q = {}
        # SCP
        scp = SCP(alpha=1 - q).conformalize(X_test_scaled, scores_cal_scaled)
        covered_base = scp.predict_coverage(X_test_scaled, scores_test_scaled)
        vol_base_q1, vol_base_q2, vol_base_q3 = vol_base(scp, s_scaler, r_scaler)

        results_q["SCP"] = {
            "Marginal Coverage": covered_base.mean(),
            "CovGap": get_covgap(covered_base, clusters),
            "Vol Q1": vol_base_q1,
            "Vol Median": vol_base_q2,
            "Vol Q3": vol_base_q3,
        }

        # CQR
        cqr = CQRHyperRectangle(alpha=1 - q)
        cqr.fit(X_train_val_scaled, y_train_val_scaled)
        cqr.conformalize(X_cal_scaled, y_cal_scaled)

        covered_cqr = cqr.predict_coverage(X_test_scaled, y_test_scaled)
        vol_cqr_q1, vol_cqr_q2, vol_cqr_q3 = vol_cqr(cqr, X_test_scaled, y_scaler)

        results_q["CQR"] = {
            "Marginal Coverage": covered_cqr.mean(),
            "CovGap": get_covgap(covered_cqr, clusters),
            "Vol Q1": vol_cqr_q1,
            "Vol Median": vol_cqr_q2,
            "Vol Q3": vol_cqr_q3,
        }

        # HPD
        covered_hpd = hpd.predict_coverage(X_test_scaled, y_test_scaled, quantile=q)
        vol_hpd_q1, vol_hpd_q2, vol_hpd_q3 = vol_hpd(hpd, X_test_scaled, y_scaler, q)
        results_q["HPD"] = {
            "Marginal Coverage": covered_hpd.mean(),
            "CovGap": get_covgap(covered_hpd, clusters),
            "Vol Q1": vol_hpd_q1,
            "Vol Median": vol_hpd_q2,
            "Vol Q3": vol_hpd_q3,
        }

        # CONTRA
        covered_contra = contra.predict_coverage(
            X_test_scaled, y_test_scaled, quantile=q
        )
        vol_contra_q1, vol_contra_q2, vol_contra_q3 = vol_contra(
            contra, X_test_scaled, y_scaler, q
        )
        results_q["CONTRA"] = {
            "Marginal Coverage": covered_contra.mean(),
            "CovGap": get_covgap(covered_contra, clusters),
            "Vol Q1": vol_contra_q1,
            "Vol Median": vol_contra_q2,
            "Vol Q3": vol_contra_q3,
        }

        # PIT-CP
        covered_pit = pitcp.predict_coverage(
            X_test_scaled, scores_test_scaled, quantile=q
        )
        vol_pit_q1, vol_pit_q2, vol_pit_q3 = vol_pitcp(
            pitcp, X_test_scaled, s_scaler, r_scaler, q
        )

        results_q["PIT-CP"] = {
            "Marginal Coverage": covered_pit.mean(),
            "CovGap": get_covgap(covered_pit, clusters),
            "Vol Q1": vol_pit_q1,
            "Vol Median": vol_pit_q2,
            "Vol Q3": vol_pit_q3,
        }
        results[q] = pd.DataFrame(results_q).T

    return results


def main():
    parser = argparse.ArgumentParser()
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--sarcos", action="store_true")
    dataset_group.add_argument("--naval", action="store_true")
    args = parser.parse_args()
    dataset_name = "sarcos" if args.sarcos else "naval"
    n_features, n_targets = (21, 7) if args.sarcos else (16, 2)
    batch_size = 1024 if args.sarcos else 512

    np.random.seed(42)
    torch.manual_seed(42)

    train_data = np.loadtxt(ROOT / "data" / f"{dataset_name}-train.csv", delimiter=",")
    X_train, y_train = train_data[:, :n_features], train_data[:, n_features:]
    valtest_data = np.loadtxt(
        ROOT / "data" / f"{dataset_name}-valtest.csv", delimiter=","
    )
    X_valtest = valtest_data[:, :n_features]
    y_valtest = valtest_data[:, n_features:]
    y_pred = np.loadtxt(ROOT / "data" / f"{dataset_name}-pred.csv", delimiter=",")

    results_by_q = defaultdict(list)
    for _ in range(N_RUNS):
        idx = np.random.permutation(len(X_valtest))
        result = run(
            X_train,
            y_train,
            X_valtest[idx],
            y_valtest[idx],
            y_pred[idx],
            n_features,
            n_targets,
            batch_size,
        )
        for quantile, frame in result.items():
            results_by_q[quantile].append(frame)

    for quantile, frames in results_by_q.items():
        frame = pd.concat(frames)
        mean_frame = frame.groupby(level=0).mean()
        std_frame = frame.groupby(level=0).std()
        final_frame = pd.DataFrame()
        for column in mean_frame.columns:
            final_frame[f"{column} Mean"] = mean_frame[column]
            final_frame[f"{column} Std"] = std_frame[column]

        print(f"\nQuantile: {quantile}")
        print(final_frame)
        final_frame.to_string(
            ROOT / "figures" / f"{dataset_name}-quantile-{quantile}.txt"
        )


if __name__ == "__main__":
    main()
