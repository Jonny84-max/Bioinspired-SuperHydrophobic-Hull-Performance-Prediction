import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from scipy.stats import ks_2samp


def run_v2_reliability_study(
    csv_path="biomimetic_opsimml_dataset.csv",
    model_path="shs_model.pkl",
    features_path="shs_features.pkl",
    test_size=0.20,
    random_state=42,
    save_figure=False,
    figure_path="v2_reliability_study.png"
):
    """
    V2.1 Reliability Study for the unified SHS predictive pipeline.

    Evaluates:
        1. Drag reduction prediction
        2. Biofouling prediction

    Metrics:
        - R²
        - MAE
        - RMSE
        - Mean Bias Error (MBE)
        - Residual STD
        - Maximum Absolute Error
        - KS statistic
        - KS p-value

    Visual diagnostics:
        - Actual vs Predicted
        - Residual scatter
        - Residual histogram
        - Empirical CDF / KS distribution comparison

    Important:
        The 20% hold-out set is created using the same random_state
        and splitting procedure used during model development.
    """

    # 1. Load data, model, and schema
    df = pd.read_csv(csv_path)
    pipeline = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    target_columns = ["drag_reduction", "biofouling"]

    required_columns = feature_columns + target_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[feature_columns]
    y = df[target_columns]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 3. Predictions
    y_pred_array = np.asarray(pipeline.predict(X_test))
    if y_pred_array.ndim != 2 or y_pred_array.shape[1] != 2:
        raise ValueError("Pipeline did not return two outputs.")

    # 4. Figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))  # scatter + residual scatter
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(14, 5.5))  # residual histograms
    fig_ks, axes_ks = plt.subplots(1, 2, figsize=(14, 5.5))  # KS ECDFs

    targets_info = [
        ("drag_reduction", 0, "Drag Reduction (%)"),
        ("biofouling", 1, "Biofouling Index")
    ]

    metrics = {}

    # 5. Evaluate each target
    for target, idx, label_name in targets_info:
        y_true = y_test[target].to_numpy(dtype=float)
        y_pred = y_pred_array[:, idx].astype(float)

        # Metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        residuals = y_true - y_pred
        mean_bias_error = np.mean(residuals)
        residual_std = np.std(residuals, ddof=1)
        max_abs_error = np.max(np.abs(residuals))
        ks_result = ks_2samp(y_true, y_pred)

        metrics[target] = {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Mean_Bias_Error": mean_bias_error,
            "Residual_STD": residual_std,
            "Maximum_Absolute_Error": max_abs_error,
            "KS_Statistic": ks_result.statistic,
            "KS_p_value": ks_result.pvalue
        }

        # Scatter: Actual vs Predicted
        ax_scatter = axes[0, idx]
        ax_scatter.scatter(y_true, y_pred, alpha=0.55, s=18)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Prediction")
        ax_scatter.set_title(f"{label_name}\nActual vs Predicted ($R^2$ = {r2:.4f})")
        ax_scatter.set_xlabel(f"Measured {label_name}")
        ax_scatter.set_ylabel("V2 Pipeline Prediction")
        ax_scatter.legend()
        ax_scatter.grid(alpha=0.25)

        # Residual scatter
        ax_resid = axes[1, idx]
        ax_resid.scatter(y_pred, residuals, alpha=0.55, s=18)
        ax_resid.axhline(0, color="red", linestyle="--", lw=2)
        ax_resid.set_title(f"{label_name}\nResiduals (Bias = {mean_bias_error:.4f})")
        ax_resid.set_xlabel("Predicted Value")
        ax_resid.set_ylabel("Residual (True − Predicted)")
        ax_resid.grid(alpha=0.25)

        # Residual histogram
        ax_hist = axes_hist[idx]
        ax_hist.hist(residuals, bins=30, color="gray", alpha=0.7)
        ax_hist.axvline(0, color="red", linestyle="--", lw=2)
        ax_hist.set_title(f"{label_name}\nResidual Distribution")
        ax_hist.set_xlabel("Residual")
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(alpha=0.25)

        # KS ECDF
        y_true_sorted, y_pred_sorted = np.sort(y_true), np.sort(y_pred)
        ecdf_true = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
        ecdf_pred = np.arange(1, len(y_pred_sorted) + 1) / len(y_pred_sorted)
        ax_ks = axes_ks[idx]
        ax_ks.plot(y_true_sorted, ecdf_true, lw=2, label="Test Dataset")
        ax_ks.plot(y_pred_sorted, ecdf_pred, lw=2, linestyle="--", label="Model Predictions")
        ax_ks.set_title(f"{label_name}\nKS Stat = {ks_result.statistic:.4f}, p = {ks_result.pvalue:.4f}")
        ax_ks.set_xlabel(label_name)
        ax_ks.set_ylabel("Cumulative Probability")
        ax_ks.legend()
        ax_ks.grid(alpha=0.25)

    fig.tight_layout()
    fig_hist.tight_layout()
    fig_ks.suptitle("V2 SHS Model — Distributional Similarity Analysis", fontsize=14)
    fig_ks.tight_layout()

    # Optional save
    if save_figure:
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        fig_hist.savefig(figure_path.replace(".png", "_ResidualHist.png"), dpi=300, bbox_inches="tight")
        fig_ks.savefig(figure_path.replace(".png", "_KS.png"), dpi=300, bbox_inches="tight")

    # Professional summary
    print("\n" + "="*70)
    print("V2 SHS RELIABILITY STUDY")
    print("="*70)
    print(f"Dataset:       {csv_path}")
    print(f"Model:         {model_path}")
    print(f"Test fraction: {test_size:.0%}")
    print(f"Random state:  {random_state}")
    print(f"Test samples:  {len(X_test)}")
    print("-"*70)

    for target in target_columns:
        result = metrics[target]
        print(f"\n{target.upper()}")
        for k, v in result.items():
            print(f"{k:25}: {v:.6f}")

    print("\n" + "="*70)
    print("\nInterpretation:")
    print("- R², MAE and RMSE evaluate predictive agreement.")
    print("- Mean Bias Error indicates systematic over/under-prediction.")
    print("- Residual STD and histogram show dispersion and bias visually.")
    print("- Maximum Absolute Error highlights worst-case deviation.")
    print("- KS statistic and p-value assess distributional similarity.")
    print("- KS similarity does not prove sample-level accuracy; it only compares distributions.")
    print("="*70)

    return metrics, fig, fig_hist, fig_ks
