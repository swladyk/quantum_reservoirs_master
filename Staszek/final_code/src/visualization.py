# src/visualization.py

import matplotlib.pyplot as plt
import numpy as np

# Import necessary functions from other modules
from .data_generation import create_io_pairs
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)


def _split_indices(n, train_fraction, val_fraction):
    """Chronological split indices: (train_end, val_end)."""
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    return train_end, val_end


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
    Generates and displays a plot comparing the best QRC and Classical ESN models
    on the held-out test segment.
    """
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating plot for profile: {profile_name} ---\n")

    # --- 1. Regenerate Time Series and apply 60/20/20 split ---
    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    train_end, val_end = _split_indices(
        len(time_series), constants['TRAIN_FRACTION'], constants['VAL_FRACTION']
    )
    train_series = time_series[:train_end]
    test_series = time_series[val_end:]

    washout = constants.get('WASHOUT', 100)

    # --- 2. Re-train Best QRC Model ---
    print(f"Retraining best QRC model...")
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

    # --- 3. Re-train Best Classical ESN Model ---
    # (We keep the training logic to allow console comparison, but it won't be plotted)
    print(f"Retraining best Classical ESN model...")
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

    # --- 4. Plot Comparison ---
    # Align test outputs with washed-out predictions, then limit to first 200 steps
    qrc_test_outputs_aligned = qrc_test_outputs[washout:]
    classical_test_outputs_aligned = classical_test_outputs[washout:]
    plot_limit = 200
    min_len = min(len(qrc_test_outputs_aligned), len(classical_test_outputs_aligned), plot_limit)

    test_outputs = qrc_test_outputs_aligned[:min_len]
    qrc_preds_plot = qrc_preds[:min_len]
    # Classical predictions prepared but NOT plotted

    plt.figure(figsize=(15, 7))

    # 1. True Data (Solid line)
    plt.plot(test_outputs, label="True Data (Test Set)", color="black", linewidth=2.5, alpha=0.8)

    # 2. QRC Prediction (Dashed line, BOLDED via linewidth=2.5)
    plt.plot(qrc_preds_plot,
             label=f"Best QRC Prediction (Median Test MSE: {best_qrc_row['median_test_mse']:.6f})",
             color="black",
             linestyle="--",
             alpha=0.9,
             linewidth=2.5)

    # MODIFICATION: Classical ESN plot removed as requested.

    plt.xlabel("Time Step (in test set)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.title(f"One-Step-Ahead Prediction Comparison for: {profile_name}", fontsize=14, weight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"../reports/figures/{profile_name}_comparison.png", dpi=300)
    plt.show()


def plot_train_val_test_overview(best_qrc_row, data_profile_config, constants):
    """
    Plots the full time series with chronological train/val/test regions shaded,
    and overlays the best QRC one-step-ahead prediction on the test segment.
    """
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating split overview for profile: {profile_name} ---\n")

    # --- 1. Regenerate and split ---
    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    train_end, val_end = _split_indices(
        len(time_series), constants['TRAIN_FRACTION'], constants['VAL_FRACTION']
    )
    train_series = time_series[:train_end]
    test_series = time_series[val_end:]

    washout = constants.get('WASHOUT', 100)

    # --- 2. Re-train QRC on train and predict on test ---
    W_out_q, weights_q, biases_q, qrc_win_size = _retrain_best_qrc(
        best_qrc_row, train_series, constants
    )
    qrc_test_inputs, qrc_test_outputs = create_io_pairs(test_series, qrc_win_size)
    qrc_preds = predict_esn(
        qrc_test_inputs, weights_q, biases_q, W_out_q,
        int(best_qrc_row['n_layers']), qrc_win_size,
        best_qrc_row['leakage_rate'], washout=washout
    )

    # Map predictions back to absolute time-series indices.
    # First post-washout prediction targets test_series[qrc_win_size + washout].
    test_start_abs = val_end + qrc_win_size + washout  # lag is 0 throughout this study
    pred_x = np.arange(test_start_abs, test_start_abs + len(qrc_preds))

    # --- 3. Plot ---
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(np.arange(len(time_series)), time_series,
            color='black', linewidth=1.2, label='True Data')

    # Shade the three regions
    ax.axvspan(0, train_end, color='tab:blue', alpha=0.10, label='Train (60%)')
    ax.axvspan(train_end, val_end, color='tab:orange', alpha=0.15, label='Validation (20%)')
    ax.axvspan(val_end, len(time_series), color='tab:green', alpha=0.15, label='Test (20%)')

    # Boundary lines
    ax.axvline(train_end, color='black', linewidth=0.8, linestyle=':')
    ax.axvline(val_end, color='black', linewidth=0.8, linestyle=':')

    # QRC predictions on test
    ax.plot(pred_x, qrc_preds,
            color='crimson', linewidth=1.4, linestyle='--',
            label=f"Best QRC prediction (Median Test MSE: {best_qrc_row['median_test_mse']:.6f})")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title(f"Train/Validation/Test split with QRC prediction — {profile_name}",
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"../reports/figures/{profile_name}_split_overview.png", dpi=300)
    plt.show()
