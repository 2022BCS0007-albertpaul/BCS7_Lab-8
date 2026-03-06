import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/housing.csv")

# Example: assume last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)

metrics = {
    "rmse": float(rmse),
    "r2": float(r2),
    "training_samples": len(X_train)
}

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)