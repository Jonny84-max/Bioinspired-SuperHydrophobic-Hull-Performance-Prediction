import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATASET
# ============================================================
DATA_FILE = "biomimetic_opsimml_dataset.csv"
df = pd.read_csv(DATA_FILE)

print("="*70)
print("SHS PREDICTIVE V2.1 - Unified Multi-Output Training")
print("="*70)
print(f"Samples: {df.shape[0]}, Columns: {df.shape[1]}")

# ============================================================
# 2. DEFINE TARGETS & FEATURES
# ============================================================
target_columns = ["drag_reduction", "biofouling"]
excluded_columns = target_columns + ["durability"]

feature_columns = [col for col in df.columns if col not in excluded_columns]
assert len(feature_columns) == 33, f"Expected 33 features, got {len(feature_columns)}"

X = df[feature_columns].copy()
y = df[target_columns].copy()

categorical_columns = ["material", "coating", "wetting_state"]
numeric_columns = [col for col in feature_columns if col not in categorical_columns]

print("\nFeature configuration")
print("-"*50)
print(f"Raw input features: {len(feature_columns)}")
print("Excluded: durability, drag_reduction, biofouling")

# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", encoder, categorical_columns),
        ("numeric", "passthrough", numeric_columns)
    ]
)

rf = RandomForestRegressor(
    n_estimators=126,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(rf))
])

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("\nTrain/Test Split")
print("-"*50)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ============================================================
# 5. INITIAL TRAINING
# ============================================================
print("\nTraining unified multi-output Random Forest...")
model.fit(X_train, y_train)
print("Training complete.")

# ============================================================
# 6. EVALUATION ON TEST SET
# ============================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

metrics = []
for i, target in enumerate(target_columns):
    train_r2 = r2_score(y_train[target], y_pred_train[:, i])
    test_r2 = r2_score(y_test[target], y_pred_test[:, i])
    test_rmse = np.sqrt(mean_squared_error(y_test[target], y_pred_test[:, i]))
    test_mae = mean_absolute_error(y_test[target], y_pred_test[:, i])

    metrics.append({
        "Target": target,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("shs_model_metrics.csv", index=False)
print("\nSaved metrics: shs_model_metrics.csv")
print(metrics_df)

# ============================================================
# 7. CROSS-VALIDATION (on training set only)
# ============================================================
print("\n5-Fold Cross-Validation (development set only)")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {target: [] for target in target_columns}

for train_idx, val_idx in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    cv_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", MultiOutputRegressor(rf))
    ])
    cv_model.fit(X_fold_train, y_fold_train)
    y_val_pred = cv_model.predict(X_fold_val)

    for i, target in enumerate(target_columns):
        fold_r2 = r2_score(y_fold_val[target], y_val_pred[:, i])
        cv_results[target].append(fold_r2)

for target in target_columns:
    scores = cv_results[target]
    print(f"{target}: Mean R² = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# ============================================================
# 8. FINAL REFIT & FEATURE IMPORTANCE
# ============================================================
model.fit(X_train, y_train)  # refit on full 80% training set

estimators = model.named_steps["regressor"].estimators_
encoded_features = model.named_steps["preprocessor"].get_feature_names_out()

for i, target in enumerate(target_columns):
    importances = estimators[i].feature_importances_
    feat_imp = pd.Series(importances, index=encoded_features).sort_values(ascending=False)

    feat_imp_df = pd.DataFrame({
        "Feature": feat_imp.index,
        "Importance": feat_imp.values
    })
    feat_imp_df.to_csv(f"{target}_feature_importance.csv", index=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=feat_imp.head(10).values, y=feat_imp.head(10).index)
    plt.title(f"Top 10 Features: {target}")
    plt.tight_layout()
    plt.savefig(f"{target}_feature_importance.png", dpi=300)
    plt.close()

print("\nSaved feature importance CSVs and PNGs.")

# ============================================================
# 9. SAVE ARTIFACTS
# ============================================================
joblib.dump(model, "shs_model.pkl")
joblib.dump(feature_columns, "shs_features.pkl")  # raw schema

test_results = pd.DataFrame({
    "actual_drag_reduction": y_test["drag_reduction"].values,
    "predicted_drag_reduction": y_pred_test[:, 0],
    "actual_biofouling": y_test["biofouling"].values,
    "predicted_biofouling": y_pred_test[:, 1]
})
test_results.to_csv("shs_test_predictions.csv", index=False)

print("\nArtifacts saved:")
print("  shs_model.pkl")
print("  shs_features.pkl (raw 33 features)")
print("  shs_model_metrics.csv")
print("  shs_test_predictions.csv")
print("  drag_feature_importance.csv / .png")
print("  biofouling_feature_importance.csv / .png")
