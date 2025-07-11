import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Create necessary folders
os.makedirs("output", exist_ok=True)
os.makedirs("output/symptom_graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 2: Load dataset
df = pd.read_csv("dataset.csv")

# Step 3: Clean dataset
df = df.dropna()
df.to_csv("output/cleaned_dataset.csv", index=False)

# Step 4: Prepare features and label
X = df.drop("target", axis=1)
y = df["target"]

# Step 5: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Step 8: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 10: Save model and scaler
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Step 11: Generate and save symptom-wise graphs
for col in X.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=y, y=df[col])
    plt.title(f"{col} vs Heart Disease")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"output/symptom_graphs/{col}_vs_target.png")
    plt.close()