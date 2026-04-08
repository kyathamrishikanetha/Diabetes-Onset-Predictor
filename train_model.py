import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("/Users/rishika/Desktop/Diabetes onset predictor/diabetes.csv")

# Check dataset (optional but recommended)
print(data.head())

# Split data
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

# Fit model
model.fit(X_train, y_train)

# Save model (give full path to avoid confusion)
model_path = "/Users/rishika/Desktop/Diabetes onset predictor/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at:", model_path)