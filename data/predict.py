from io import BytesIO

import numpy as np
import requests
from scipy.io import loadmat
from sklearn.multioutput import MultiOutputRegressor
from tabpfn_client import TabPFNRegressor, set_access_token

# Set seed for reproducibility
np.random.seed(42)

# Load data from the web
url = "https://gaussianprocess.org/gpml/data/sarcos_inv.mat"
response = requests.get(url, timeout=30)
response.raise_for_status()

mat = loadmat(BytesIO(response.content))
data = mat["sarcos_inv"]
X, y = data[:, :21], data[:, 21:]

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Split
split_idx = len(X) // 3
X_train, y_train = X[:split_idx], y[:split_idx]
X_valtest, y_valtest = X[split_idx:], y[split_idx:]

# Auth
API_TOKEN = "PUT_YOUR_TOKEN_HERE"
set_access_token(API_TOKEN)

# Model
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

# Save data
train_output = np.hstack([X_train, y_train])
valtest_output = np.hstack([X_valtest, y_valtest])

y_valtest_pred = model.predict(X_valtest)

np.savetxt("train.csv", train_output, delimiter=",")
np.savetxt("valtest.csv", valtest_output, delimiter=",")
np.savetxt("pred.csv", y_valtest_pred, delimiter=",")
