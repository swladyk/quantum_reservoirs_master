# src/visualization.py

import matplotlib.pyplot as plt
import numpy as np

from .data_generation import create_io_pairs
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)


def _retrain_best_qrc(best_qrc_row, train_series, constants):
    """Re-train the best QRC model on train_series; return artifacts + window size."""
    qrc_win_size = int(best_qrc_row['window_size'])
    qrc_train_inputs, qrc_train_outputs = create_io_pairs(train_series, qrc_win_size)

    W_out_q, weights_q, biases_q, _ = train_esn_reservoir(
        qrc_train_inputs, qrc_train_outputs,
        n_layers=int(best_qrc_row['n_layers']),
        n_qubits=qrc_win_size,
        leakage_rate=best_qrc_row['leakage_rate'],
        lambda_reg=best_qrc_row['lambda_reg'],
        seed=constants['SEED']
    )
    return W_out_q, weights_q, biases_q, qrc_win_size


def _retrain_best_classical(best_classical_row, train_series, constants):
    """Re-train the best Classical ESN on train_series; return artifacts + window size."""
    classical_win_size = int(best_classical_row.get('window_size', 10))
    classical_train_inputs, classical_train_outputs = create_io_pairs(train_series, classical_win_size)

    W_in_c, W_res_c = initialize_classical_reservoir(
        reservoir_size=int(best_classical_row['reservoir_size']),
        input_dim=classical_win_size,
        spectral_radius=best_classical_row['spectral_radius'],
        sparsity=best_classical_row['sparsity'],
        seed=constants['SEED']
    )
    W_out_c, last_state_c = train_classical_reservoir(
        classical_train_inputs, classical_train_outputs, W_in_c, W_res_c,
        reservoir_size=int(best_classical_row['reservoir_size']),
        leakage_rate=best_classical_row['leakage_rate'],
        lambda_reg=best_classical_row['lambda_reg']
    )
    return W_in_c, W_res_c, W_out_c, last_state_c, classical_win_size


def plot_best_model_comparison(best_qrc_row, best_classical_row, data_profile_config, constants):
    """
    Plot best QRC prediction on the held-out test segment (first 200 steps).
    Training is performed on the full CV pool (TRAIN_FRACTION of the series).
    """
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating plot for profile: {profile_name} ---\n")

    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    cv_end = int(len(time_series) * constants['TRAIN_FRACTION'])
    train_series = time_series[:cv_end]
    test_series = time_series[cv_end:]

    washout = constants.get('WASHOUT', 100)

    print("Retraining best QRC model on full CV pool...")
    W_out_q, weights_q, biases_q, qrc_win_size = _retrain_best_qrc(
        best_qrc_row, train_series, constants
    )
    qrc_test_inputs, qrc_test_outputs = create_io_pairs(test_series, qrc_win_size)
    qrc_preds = predict_esn(
        qrc_test_inputs, weights_q, biases_q, W_out_q,
        int(best_qrc_row['n_layers']), qrc_win_size,
        best_qrc_row['leakage_rate'], washout=washout
    )
    print(f"Best QRC Median Test MSE: {best_qrc_row['median_test_mse']:.6f}")

    print("Retraining best Classical ESN on full CV pool...")
    W_in_c, W_res_c, W_out_c, last_state_c, classical_win_size = _retrain_best_classical(
        best_classical_row, train_series, constants
    )
    classical_test_inputs, classical_test_outputs = create_io_pairs(test_series, classical_win_size)
    classical_preds = predict_esn_classical(
        classical_test_inputs, W_in_c, W_res_c, W_out_c,
        reservoir_size=int(best_classical_row['reservoir_size']),
        leakage_rate=best_classical_row['leakage_rate'],
        initial_state=last_state_c, washout=washout
    )
    print(f"Best Classical ESN Median Test MSE: {best_classical_row['median_test_mse']:.6f}")

    qrc_test_outputs_aligned = qrc_test_outputs[washout:]
    plot_limit = 200
    min_len = min(len(qrc_test_outputs_aligned), plot_limit)

    test_outputs = qrc_test_outputs_aligned[:min_len]
    qrc_preds_plot = qrc_preds[:min_len]

    plt.figure(figsize=(15, 7))
    plt.plot(test_outputs, label="True Data (Test Set)", color="black", linewidth=2.5, alpha=0.8)
    plt.plot(qrc_preds_plot,
             label=f"Best QRC Prediction (Median Test MSE: {best_qrc_row['median_test_mse']:.6f})",
             color="black", linestyle="--", alpha=0.9, linewidth=2.5)
    plt.xlabel("Time Step (in test set)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.title(f"One-Step-Ahead Prediction Comparison for: {profile_name}", fontsize=14, weight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"../reports/figures/{profile_name}_comparison.png", dpi=300)
    plt.show()


def plot_cv_split_overview(best_qrc_row, data_profile_config, constants):
    """
    Plot the full time series with the CV pool (TRAIN_FRACTION) and held-out test
    regions shaded, sliding-window CV fold val-boundaries marked, and the best
    QRC prediction overlaid on the test segment.
    """
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating CV split overview for profile: {profile_name} ---\n")

    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    cv_end = int(len(time_series) * constants['TRAIN_FRACTION'])
    train_series = time_series[:cv_end]
    test_series = time_series[cv_end:]

    washout = constants.get('WASHOUT', 100)
    n_splits = constants.get('N_SPLITS', 5)

    W_out_q, weights_q, biases_q, qrc_win_size = _retrain_best_qrc(
        best_qrc_row, train_series, constants
    )
    qrc_test_inputs, _ = create_io_pairs(test_series, qrc_win_size)
    qrc_preds = predict_esn(
        qrc_test_inputs, weights_q, biases_q, W_out_q,
        int(best_qrc_row['n_layers']), qrc_win_size,
        best_qrc_row['leakage_rate'], washout=washout
    )

    # Predictions are aligned with the test segment after windowing + washout.
    test_start_abs = cv_end + qrc_win_size + washout
    pred_x = np.arange(test_start_abs, test_start_abs + len(qrc_preds))

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(np.arange(len(time_series)), time_series,
            color='black', linewidth=1.2, label='True Data')

    train_pct = int(round(constants['TRAIN_FRACTION'] * 100))
    test_pct = 100 - train_pct
    ax.axvspan(0, cv_end, color='tab:blue', alpha=0.10,
               label=f"CV pool ({train_pct}%) — sliding {n_splits}-fold")
    ax.axvspan(cv_end, len(time_series), color='tab:green', alpha=0.15,
               label=f"Held-out Test ({test_pct}%)")

    # Mark sliding-window CV val-segment starts inside the CV pool.
    n = len(train_series)
    val_size = n // (n_splits + 1)
    train_size = (n_splits - 1) * val_size
    max_shift = n - train_size - val_size
    for i in range(n_splits):
        shift = int(round(i * max_shift / (n_splits - 1))) if n_splits > 1 else 0
        val_start = shift + train_size
        ax.axvline(val_start, color='tab:orange', linewidth=0.8, linestyle=':', alpha=0.7)
    # Dummy line for legend entry
    ax.plot([], [], color='tab:orange', linewidth=0.8, linestyle=':', alpha=0.7,
            label='CV fold val-start boundaries')

    # CV pool / test boundary
    ax.axvline(cv_end, color='black', linewidth=0.8, linestyle='--')

    ax.plot(pred_x, qrc_preds,
            color='crimson', linewidth=1.4, linestyle='--',
            label=f"Best QRC prediction (Median Test MSE: {best_qrc_row['median_test_mse']:.6f})")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title(f"CV pool / Held-out Test split with QRC prediction — {profile_name}",
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"../reports/figures/{profile_name}_cv_overview.png", dpi=300)
    plt.show()


def plot_cv_vs_test_scatter(results_df, save_path='../reports/figures/cv_vs_test_scatter.png'):
    """
    Honest-CV scatter: median_cv_mse (x) vs median_test_mse (y) for every
    hyperparameter combination in results_df. Points clustered around the y=x
    line indicate the CV procedure ranks hyperparameters consistently with the
    held-out test.

    One subplot per data profile; markers distinguish QRC vs Classical_ESN.
    """
    profiles = sorted(results_df['data_profile'].unique())
    n_profiles = len(profiles)

    n_cols = 3
    n_rows = int(np.ceil(n_profiles / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.5 * n_rows), squeeze=False)

    for idx, profile in enumerate(profiles):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        profile_df = results_df[results_df['data_profile'] == profile].dropna(
            subset=['median_cv_mse', 'median_test_mse']
        )

        for model_type, marker, color in [('QRC', 'o', 'tab:blue'),
                                          ('Classical_ESN', '^', 'tab:orange')]:
            m = profile_df[profile_df['model_type'] == model_type]
            if not m.empty:
                ax.scatter(m['median_cv_mse'], m['median_test_mse'],
                           marker=marker, color=color, alpha=0.55,
                           edgecolors='black', linewidths=0.4,
                           label=model_type, s=40)

        if not profile_df.empty:
            mn = min(profile_df['median_cv_mse'].min(), profile_df['median_test_mse'].min())
            mx = max(profile_df['median_cv_mse'].max(), profile_df['median_test_mse'].max())
            mn = max(mn, 1e-14)
            ax.plot([mn, mx], [mn, mx], color='black', linestyle='--', linewidth=0.9, label='y = x')
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlabel('Median CV MSE')
        ax.set_ylabel('Median Test MSE')
        ax.set_title(profile, fontsize=10)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize=8)

    for idx in range(n_profiles, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis('off')

    fig.suptitle("CV vs Test: median CV MSE vs median Test MSE per hyperparameter combination",
                 fontsize=13, weight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
