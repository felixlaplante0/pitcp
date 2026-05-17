import numpy as np
from scipy.io import loadmat
from sklearn.multioutput import MultiOutputRegressor
from tabpfn_client import TabPFNRegressor, set_access_token

# Set seed
np.random.seed(42)

# Load data
data = loadmat("sarcos_inv")["sarcos_inv"]
X, y = data[:, :21], data[:, 21:]
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Split
split_idx = len(X) // 3
X_train, y_train = X[:split_idx], y[:split_idx]
X_valtest, y_valtest = X[split_idx:], y[split_idx:]

# Authenticate
API_TOKEN = "PUT_YOUR_TOKEN_HERE"
set_access_token(API_TOKEN)

# Train
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

# Save: X, y in each file
train_output = np.hstack([X_train, y_train])
valtest_output = np.hstack([X_valtest, y_valtest])

# Predict
y_valtest_pred = model.predict(X_valtest)

# Save predictions
np.savetxt("train.csv", train_output, delimiter=",")
np.savetxt("valtest.csv", valtest_output, delimiter=",")
np.savetxt("pred.csv", y_valtest_pred, delimiter=",")
