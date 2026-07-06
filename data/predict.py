import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.multioutput import MultiOutputRegressor
from tabpfn_client import TabPFNRegressor, set_access_token

ROOT = Path(__file__).resolve().parents[1]

def main():
    """Generates deterministic train, validation-test, and prediction files."""
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--sarcos", action="store_true")
    dataset_group.add_argument("--naval", action="store_true")
    args = parser.parse_args()
    dataset_name = "sarcos" if args.sarcos else "naval"

    if args.sarcos:
        mat = loadmat(ROOT / "data" / "sarcos_inv.mat")
        data = mat["sarcos_inv"]
        X, y = data[:, :21], data[:, 21:]
    else:
        data = np.loadtxt(ROOT / "data" / "naval.txt")
        X, y = data[:, :-2], data[:, -2:]

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split_idx = len(X) // 3
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_valtest, y_valtest = X[split_idx:], y[split_idx:]

    set_access_token(os.environ["TABPFN_ACCESS_TOKEN"])
    model = MultiOutputRegressor(
        TabPFNRegressor(
            thinking_mode=True,
            thinking_effort="medium",
            thinking_timeout_s=60,
            thinking_metric="rmse",
            random_state=42,
        )
    )
    model.fit(X_train, y_train)

    train_output = np.hstack([X_train, y_train])
    valtest_output = np.hstack([X_valtest, y_valtest])
    y_valtest_pred = model.predict(X_valtest)

    np.savetxt(
        ROOT / "data" / f"{dataset_name}-train.csv", train_output, delimiter=","
    )
    np.savetxt(
        ROOT / "data" / f"{dataset_name}-valtest.csv", valtest_output, delimiter=","
    )
    np.savetxt(
        ROOT / "data" / f"{dataset_name}-pred.csv", y_valtest_pred, delimiter=","
    )


if __name__ == "__main__":
    main()
