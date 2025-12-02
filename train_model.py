import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# ---------- 1. Load Data ----------
data_path = "../data/iris.csv"
df = pd.read_csv(data_path)

# Features & target
X = df.drop("species", axis=1)
y = df["species"]

# ---------- 2. Split Data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 3. Scale Features ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. Train Model ----------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------- 5. Evaluate ----------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- 6. Save Model & Scaler ----------
os.makedirs("../model", exist_ok=True)

with open("../model/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully!")
