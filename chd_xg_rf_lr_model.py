import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("framingham.csv")
df.dropna(subset=["TenYearCHD"], inplace=True)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
df[df.columns] = imputer.fit_transform(df)

# Feature engineering
df["bmi_category"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=[0, 1, 2, 3]).astype(int)
df.drop(columns=["education"], inplace=True, errors="ignore")

# Prepare features and target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Define base models
xgb = XGBClassifier(eval_metric="logloss", n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
lr = LogisticRegression(max_iter=1000)

# Create stacking classifier
stack_model = StackingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# Train model
stack_model.fit(X_train, y_train)

# Predict
y_pred = stack_model.predict(X_test)
y_prob = stack_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n--- Evaluation Metrics ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(stack_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
print("\n5-Fold Stratified Cross-Validation Accuracy: {:.2f}%".format(cv_scores.mean() * 100))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Manual Prediction ----------
manual_input = {
    'male': 0,
    'age': 61,
    'currentSmoker': 1,
    'cigsPerDay': 30,
    'BPMeds': 0,
    'prevalentStroke': 0,
    'prevalentHyp': 1,
    'diabetes': 0,
    'totChol': 225,
    'sysBP': 150,
    'diaBP': 95,
    'BMI': 28.58,
    'heartRate': 65,
    'glucose': 103,
    'bmi_category': 2
}

input_df = pd.DataFrame([manual_input])
input_scaled = scaler.transform(input_df)
prediction = stack_model.predict(input_scaled)[0]

print("\n--- Manual Prediction ---")
print(f"Predicted CHD Risk: {'Yes (1)' if prediction == 1 else 'No (0)'}")
