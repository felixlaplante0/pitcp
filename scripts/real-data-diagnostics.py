import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zuko
from catboost import CatBoostRegressor
from pitcp import CONTRA, CQR, HPD, PITCP, SCP
from pitcp.utils import contra_volume, coverage_gap, cqr_volume, hpd_volume, lp_volume
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
N_RUNS = 10
QUANTILES = (0.6, 0.7, 0.8, 0.9)


def summarize(
    covered: np.ndarray,
    volumes: np.ndarray,
    clusters: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, float]:
    volumes = volumes * np.prod(scaler.scale_)
    if np.any(volumes <= 0):
        raise ValueError("Volumes must be positive for geometric statistics.")
    return {
        "Marginal Coverage": covered.mean(),
        "CovGap": coverage_gap(clusters, covered),
        "Vol": np.exp(np.log(volumes).mean()),
    }


def run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valtest: np.ndarray,
    y_valtest: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
    n_targets: int,
    batch_size: int,
) -> dict[float, pd.DataFrame]:
    # Split data
    half = len(X_valtest) // 2
    three_quarters = half + (len(X_valtest) - half) // 2
    residuals = y_valtest - y_pred

    X_train_val = np.concatenate([X_train, X_valtest[:half]])
    y_train_val = np.concatenate([y_train, y_valtest[:half]])

    # Scale features
    X_scaler = StandardScaler()
    X_train_val = X_scaler.fit_transform(X_train_val)
    X_val = X_scaler.transform(X_valtest[:half])
    X_cal = X_scaler.transform(X_valtest[half:three_quarters])
    X_test = X_scaler.transform(X_valtest[three_quarters:])

    # Scale targets
    y_scaler = StandardScaler()
    y_train_val = y_scaler.fit_transform(y_train_val)
    y_cal = y_scaler.transform(y_valtest[half:three_quarters])
    y_test = y_scaler.transform(y_valtest[three_quarters:])

    # Scale residuals and compute scores
    residual_scaler = StandardScaler()
    residuals_val = residual_scaler.fit_transform(residuals[:half])
    residuals_cal = residual_scaler.transform(residuals[half:three_quarters])
    residuals_test = residual_scaler.transform(residuals[three_quarters:])
    scores_val = np.max(np.abs(residuals_val), axis=1)
    scores_cal = np.max(np.abs(residuals_cal), axis=1)
    scores_test = np.max(np.abs(residuals_test), axis=1)

    # HPD and CONTRA
    density = zuko.flows.MAF(
        features=n_targets, context=n_features, hidden_features=(32, 32)
    )
    optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
    hpd = HPD(density, optimizer, n_epochs=500, batch_size=batch_size)
    hpd.fit(X_train_val, y_train_val).conformalize(X_cal, y_cal)
    contra = CONTRA(density, optimizer, batch_size=batch_size)
    contra.conformalize(X_cal, y_cal)

    # PIT-CP
    density = zuko.flows.SOSPF(features=1, context=n_features, hidden_features=(16, 16))
    optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
    pitcp = PITCP(density, optimizer, n_epochs=1000, batch_size=batch_size)
    pitcp.fit(X_val, scores_val).conformalize(X_cal, scores_cal)

    # SCP
    scp = SCP().conformalize(scores_cal)

    # Compute coverage and volume
    clusters = KMeans(n_clusters=10, random_state=42).fit_predict(X_test)
    coverages = {
        "SCP": scp.contains(scores_test, confidence_level=QUANTILES),
        "HPD": hpd.contains(X_test, y_test, confidence_level=QUANTILES),
        "CONTRA": contra.contains(X_test, y_test, confidence_level=QUANTILES),
        "PIT-CP": pitcp.contains(X_test, scores_test, confidence_level=QUANTILES),
    }
    volumes = {
        "SCP": lp_volume(
            scp, X_test, n_targets, confidence_level=QUANTILES, ord=np.inf
        ),
        "HPD": hpd_volume(hpd, X_test, QUANTILES),
        "CONTRA": contra_volume(contra, X_test, QUANTILES),
        "PIT-CP": lp_volume(
            pitcp, X_test, n_targets, confidence_level=QUANTILES, ord=np.inf
        ),
    }
    scalers = {
        "SCP": residual_scaler,
        "HPD": y_scaler,
        "CONTRA": y_scaler,
        "PIT-CP": residual_scaler,
    }
    results = {}
    for i, quantile in enumerate(QUANTILES):
        result = {
            name: summarize(
                coverage[:, i], volumes[name][:, i], clusters, scalers[name]
            )
            for name, coverage in coverages.items()
        }

        # CQR
        cqr = CQR(CatBoostRegressor(verbose=False), confidence_level=quantile).fit(
            X_train_val, y_train_val
        )
        cqr.conformalize(X_cal, y_cal)
        result["CQR"] = summarize(
            cqr.contains(X_test, y_test),
            cqr_volume(cqr, X_test),
            clusters,
            y_scaler,
        )
        results[quantile] = pd.DataFrame(result).T

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

    # Load training data
    train_data = np.loadtxt(ROOT / "data" / f"{dataset_name}-train.csv", delimiter=",")
    X_train, y_train = train_data[:, :n_features], train_data[:, n_features:]

    # Load validation and test data
    valtest_data = np.loadtxt(
        ROOT / "data" / f"{dataset_name}-valtest.csv", delimiter=","
    )
    X_valtest = valtest_data[:, :n_features]
    y_valtest = valtest_data[:, n_features:]

    # Load predictions
    y_pred = np.loadtxt(ROOT / "data" / f"{dataset_name}-pred.csv", delimiter=",")

    # Run repeated evaluations
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

    # Summarize results
    for quantile, frames in results_by_q.items():
        frame = pd.concat(frames)
        grouped = frame.groupby(level=0)
        mean_frame = grouped.mean()
        std_frame = grouped.std()
        log_volumes = np.log(frame["Vol"]).groupby(level=0)
        mean_frame["Vol"] = np.exp(log_volumes.mean())
        std_frame["Vol"] = np.exp(log_volumes.std())
        final_frame = pd.DataFrame()
        for column in mean_frame.columns:
            final_frame[f"{column} Mean"] = mean_frame[column]
            final_frame[f"{column} Std"] = std_frame[column]

        print(f"\nQuantile: {quantile}")  # noqa: T201
        print(final_frame)  # noqa: T201
        final_frame.to_string(
            ROOT / "figures" / f"{dataset_name}-quantile-{quantile}.txt"
        )


if __name__ == "__main__":
    main()
