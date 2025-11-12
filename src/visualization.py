"""
Visualization functions for the MiniLab.
Creates plots for evaluating models and learning.

  - Visualization of model complexity (Features) and performance (Task 2.4)
  - Heatmap & curve plots for polynomial models (Task 3.2, 3.3)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Visualization of model complexity (Features) and performance (Task 2.4)
# ---------------------------------------------------------------------
def plot_feature_performance(results, output_dir="results", file_name="Task_2"):
    """
    Create and save a performance plot showing how model performance
    (R² and RMSE) changes with the number of features.

    Parameters
    ----------
    results : list of dict
        Example format:
        [
            {"n_features": 1, "features": ["livingSpace"], "r2": 0.70, "rmse": 250.0},
            {"n_features": 2, "features": ["livingSpace", "numberOfRooms"], "r2": 0.78, "rmse": 210.0},
            ...
        ]
    output_dir : str, default="results"
        Directory where the plot is saved.
    file_name : str, default="Task_2"
        Identifier used for the filename.

    Returns
    -------
    str
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_features = [r["n_features"] for r in results]
    r2_values = [r["r2"] for r in results]
    rmse_values = [r["rmse"] for r in results]

    # --- Create figure ---
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_r2 = "tab:blue"
    color_rmse = "tab:red"

    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("R²", color=color_r2)
    ax1.tick_params(axis="y", labelcolor=color_r2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE (€)", color=color_rmse)
    ax2.tick_params(axis="y", labelcolor=color_rmse)

    plt.title("Model performance (validation) vs. number of features")
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved performance plot to: {output_path}")
    return output_path

# ---------------------------------------------------------------------
# Heatmap for polynomial performance (Matplotlib version, Task 3.2)
# ---------------------------------------------------------------------
def plot_heatmap_performance(results_list, output_dir="results", metric="r2_val", file_name="Task_3_heatmap"):
    """
    Visualize validation performance as a heatmap over polynomial degree × number of features.
    Works directly from flat list of dicts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Infer matrix structure automatically
    feature_groups = sorted(list(set(["+".join(r["features"]) for r in results_list])))
    max_degree = max(r["degree"] for r in results_list)

    results_matrix = np.zeros((len(feature_groups), max_degree))
    for i, f_label in enumerate(feature_groups):
        for r in results_list:
            if "+".join(r["features"]) == f_label:
                results_matrix[i, r["degree"] - 1] = r[metric]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(results_matrix, cmap="viridis", origin="lower", aspect="auto")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Feature combination")
    ax.set_title(f"Validation {metric.upper()} across model complexity")

    ax.set_xticks(np.arange(max_degree))
    ax.set_xticklabels(np.arange(1, max_degree + 1))
    ax.set_yticks(np.arange(len(feature_groups)))
    ax.set_yticklabels(feature_groups)

    for i in range(results_matrix.shape[0]):
        for j in range(results_matrix.shape[1]):
            ax.text(j, i, f"{results_matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

    plt.colorbar(im, ax=ax, label=metric.upper())
    plt.tight_layout()

    path = os.path.join(output_dir, f"{file_name}_{metric}.pdf")
    plt.savefig(path)
    plt.close(fig)
    print(f"✅ Saved heatmap to: {path}")
    return path


# ---------------------------------------------------------------------
# 2D plot: train vs. validation curves over degree (Task 3.3)
# ---------------------------------------------------------------------
def plot_polynomial_results(results_list, output_dir="results", file_name="Task_3_curves"):
    """
    Plot R² (train vs validation) over polynomial degree for a flat list of results.

    Parameters
    ----------
    results_list : list of dict
        Flat list from evaluate_polynomial_models().
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Sort by polynomial degree
    sorted_results = sorted(results_list, key=lambda x: x["degree"])
    degrees = [r["degree"] for r in sorted_results]
    r2_train = [r["r2_train"] for r in sorted_results]
    r2_val = [r["r2_val"] for r in sorted_results]

    # Plot
    ax.plot(degrees, r2_train, marker="o", label="Train R²")
    ax.plot(degrees, r2_val, marker="x", linestyle="--", label="Validation R²")

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("R²")
    ax.set_title("Polynomial model performance")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(path)
    plt.close(fig)
    print(f"✅ Saved polynomial performance plot to: {path}")
    return path