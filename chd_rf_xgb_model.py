import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("framingham.csv")
df.dropna(subset=["TenYearCHD"], inplace=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Feature engineering: create BMI category
df['bmi_category'] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=[0, 1, 2, 3]
).astype(int)

# Drop less useful columns (optional, e.g., "education")
df.drop(columns=["education"], inplace=True, errors="ignore")

# Define features and target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Define base learners
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
log_model = LogisticRegression(max_iter=1000)

# Stacking ensemble
stack = StackingClassifier(
    estimators=[('xgb', xgb_model), ('lr', log_model)],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# Train ensemble
stack.fit(X_train, y_train)

# Predictions
y_pred = stack.predict(X_test)
y_prob = stack.predict_proba(X_test)[:, 1]

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n--- EVALUATION ---")
print(" Accuracy      :", round(acc * 100, 2), "%")
print(" F1 Score      :", round(f1, 3))
print(" ROC AUC Score :", round(auc, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(stack, X_resampled, y_resampled, cv=5)
print("\n📊 5-Fold CV Accuracy: {:.2f}%".format(cv_scores.mean() * 100))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Manual prediction (new sample)
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
manual_df = pd.DataFrame([manual_input])
manual_scaled = scaler.transform(manual_df)
manual_pred = stack.predict(manual_scaled)[0]

print("\n--- Manual Prediction ---")
print("Predicted 10-Year CHD Risk:", "Yes" if manual_pred == 1 else "No")
