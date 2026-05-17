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
from utils.volume import vol_base, vol_cqr, vol_pit

# Set seed for reproducibility
torch.manual_seed(42)

# Load data
train_data = np.loadtxt("../data/train.csv", delimiter=",")
X_train, y_train = train_data[:, :21], train_data[:, 21:]

valtest_data = np.loadtxt("../data/valtest.csv", delimiter=",")
X_valtest, y_valtest = valtest_data[:, :21], valtest_data[:, 21:]

# Load predictions
y_pred = np.loadtxt("../data/pred.csv", delimiter=",")

# Split indices
half = len(valtest_data) // 2
three_quarters = half + (len(valtest_data) - half) // 2

# Compute residuals
residuals = y_valtest - y_pred

# Combine train and val for others
X_train_val = np.concatenate([X_train, X_valtest[:half]], axis=0)
y_train_val = np.concatenate([y_train, y_valtest[:half]], axis=0)

# Standardize X
X_scaler = StandardScaler()
X_train_val_scaled = X_scaler.fit_transform(X_train_val)
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
scores_val_scaled = s_scaler.fit_transform(scores_val)
scores_cal_scaled = s_scaler.transform(scores_cal)
scores_test_scaled = s_scaler.transform(scores_test)

# HPD
model_hpd = zuko.flows.MAF(features=7, context=21, hidden_features=[32, 32])
optimizer_hpd = torch.optim.Adam(model_hpd.parameters(), lr=1e-3)
hpd = HPD(model_hpd, optimizer_hpd, n_epochs=500, batch_size=1024)
hpd.fit(X_train_val_scaled, y_train_val_scaled)
hpd.conformalize(X_cal_scaled, y_cal_scaled)

# CONTRA (uses flow learned in HPD)
contra = CONTRA(hpd.estimator)
contra.conformalize(X_cal_scaled, y_cal_scaled)

# PIT-CP
model_pit = zuko.flows.SOSPF(features=1, context=21, hidden_features=[32, 32])
optimizer_pit = torch.optim.Adam(model_pit.parameters(), lr=1e-3)
X_val_scaled = X_scaler.transform(X_valtest[:half])
pit = PITCP(model_pit, optimizer_pit, n_epochs=1000, batch_size=1024)
pit.fit(X_val_scaled, scores_val_scaled)
pit.conformalize(X_cal_scaled, scores_cal_scaled)

# K-Means diagnostics
clusters = KMeans(n_clusters=5, random_state=42).fit_predict(X_test_scaled)


def get_gap(covered, clusters):
    return np.max(
        [covered[clusters == k].mean() for k in np.unique(clusters)]
    ) - np.min([covered[clusters == k].mean() for k in np.unique(clusters)])


# Evaluation loop for different quantiles
for q in [0.6, 0.7, 0.8, 0.9]:
    print(f"\nQuantile: {q}")
    results = {}

    # SCP (SCP)
    scp = SCP(alpha=1 - q).conformalize(X_test_scaled, scores_cal_scaled)
    covered_base = scp.predict_coverage(X_test_scaled, scores_test_scaled)
    vol_base_q1, vol_base_q2, vol_base_q3 = vol_base(scp, s_scaler, r_scaler)

    results["SCP"] = {
        "Gap": get_gap(covered_base, clusters),
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

    results["CQR"] = {
        "Gap": get_gap(covered_cqr, clusters),
        "Vol Q1": vol_cqr_q1,
        "Vol Median": vol_cqr_q2,
        "Vol Q3": vol_cqr_q3,
    }

    # HPD
    covered_hpd = hpd.predict_coverage(X_test_scaled, y_test_scaled, quantile=q)
    results["HPD"] = {
        "Gap": get_gap(covered_hpd, clusters),
        "Vol Q1": np.nan,
        "Vol Median": np.nan,
        "Vol Q3": np.nan,
    }

    # CONTRA
    covered_contra = contra.predict_coverage(X_test_scaled, y_test_scaled, quantile=q)
    results["CONTRA"] = {
        "Gap": get_gap(covered_contra, clusters),
        "Vol Q1": np.nan,
        "Vol Median": np.nan,
        "Vol Q3": np.nan,
    }

    # PIT-CP
    covered_pit = pit.predict_coverage(X_test_scaled, scores_test_scaled, quantile=q)
    vol_pit_q1, vol_pit_q2, vol_cqr_q3 = vol_pit(
        pit, X_test_scaled, s_scaler, r_scaler, q
    )

    results["PIT-CP"] = {
        "Gap": get_gap(covered_pit, clusters),
        "Vol Q1": vol_pit_q1,
        "Vol Median": vol_pit_q2,
        "Vol Q3": vol_cqr_q3,
    }

    # Display and save results
    df = pd.DataFrame(results).T
    print(df)
    df.to_string(f"../figures/real-data-quantile-{q}.txt")
