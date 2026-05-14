# src/experiment.py

import numpy as np
from sklearn.metrics import mean_squared_error

from .data_generation import create_io_pairs
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)

# Reservoir washout (transient burn-in) length, applied uniformly to
# training-state regression and to val/test prediction.
WASHOUT = 100


def _split_train_val_test(time_series, train_fraction, val_fraction):
    """Chronological 60/20/20-style split. Returns (train, val, test)."""
    n = len(time_series)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    return time_series[:train_end], time_series[train_end:val_end], time_series[val_end:]


# --- Podstawowe funkcje uruchamiające pojedynczy eksperyment ---

def run_single_qrc_trial(params, profile, time_series, train_fraction, val_fraction, seed):
    """Runs a SINGLE QRC trial for one seed. Returns (mse_val, mse_test)."""
    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    train_data, val_data, test_data = _split_train_val_test(
        time_series, train_fraction, val_fraction
    )

    train_inputs, train_outputs = create_io_pairs(train_data, window_size, lag)
    val_inputs, val_outputs = create_io_pairs(val_data, window_size, lag)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size, lag)

    W_out, weights, biases, _ = train_esn_reservoir(
        train_inputs, train_outputs, n_layers, window_size,
        leakage_rate, lambda_reg, seed, washout=WASHOUT
    )
    val_predictions = predict_esn(
        val_inputs, weights, biases, W_out, n_layers,
        window_size, leakage_rate, washout=WASHOUT
    )
    test_predictions = predict_esn(
        test_inputs, weights, biases, W_out, n_layers,
        window_size, leakage_rate, washout=WASHOUT
    )
    mse_val = mean_squared_error(val_outputs[WASHOUT:], val_predictions)
    mse_test = mean_squared_error(test_outputs[WASHOUT:], test_predictions)
    return mse_val, mse_test


def run_single_classical_trial(params, profile, time_series, train_fraction, val_fraction, seed):
    """Runs a SINGLE Classical ESN trial for one seed. Returns (mse_val, mse_test)."""
    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg = params
    window_size = 10
    train_data, val_data, test_data = _split_train_val_test(
        time_series, train_fraction, val_fraction
    )

    train_inputs, train_outputs = create_io_pairs(train_data, window_size)
    val_inputs, val_outputs = create_io_pairs(val_data, window_size)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size)

    W_in, W_res = initialize_classical_reservoir(
        reservoir_size, window_size, spectral_radius, sparsity, seed
    )
    W_out, final_state = train_classical_reservoir(
        train_inputs, train_outputs, W_in, W_res,
        reservoir_size, leakage_rate, lambda_reg, washout=WASHOUT
    )
    val_predictions = predict_esn_classical(
        val_inputs, W_in, W_res, W_out,
        reservoir_size, leakage_rate, final_state, washout=WASHOUT
    )
    test_predictions = predict_esn_classical(
        test_inputs, W_in, W_res, W_out,
        reservoir_size, leakage_rate, final_state, washout=WASHOUT
    )
    mse_val = mean_squared_error(val_outputs[WASHOUT:], val_predictions)
    mse_test = mean_squared_error(test_outputs[WASHOUT:], test_predictions)
    return mse_val, mse_test


# --- Wrappery agregujące wyniki po pod-ziarnach ---

def _aggregate(scores):
    median = np.median(scores)
    std = np.std(scores)
    cv = std / median if median > 0 else 0
    return median, std, cv


def run_qrc_experiment_with_subseeds(params, profile, time_series,
                                     train_fraction, val_fraction,
                                     base_seed, num_trials=11):
    """Runs QRC trials across sub-seeds; returns aggregated val + test stats."""
    val_scores, test_scores = [], []
    sub_seeds = [base_seed + i for i in range(num_trials)]

    for seed in sub_seeds:
        mse_val, mse_test = run_single_qrc_trial(
            params, profile, time_series, train_fraction, val_fraction, seed
        )
        val_scores.append(mse_val)
        test_scores.append(mse_test)

    median_val, std_val, cv_val = _aggregate(val_scores)
    median_test, std_test, cv_test = _aggregate(test_scores)

    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    return {
        'model_type': 'QRC',
        'data_profile': profile['name'],
        'median_val_mse': median_val,
        'std_val_mse': std_val,
        'cv_val_mse': cv_val,
        'median_test_mse': median_test,
        'std_test_mse': std_test,
        'cv_test_mse': cv_test,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': window_size,
        'n_layers': n_layers,
        'lag': lag,
        'base_seed': base_seed
    }


def run_classical_experiment_with_subseeds(params, profile, time_series,
                                           train_fraction, val_fraction,
                                           base_seed, num_trials=11):
    """Runs Classical ESN trials across sub-seeds; returns aggregated val + test stats."""
    val_scores, test_scores = [], []
    sub_seeds = [base_seed + i for i in range(num_trials)]

    for seed in sub_seeds:
        mse_val, mse_test = run_single_classical_trial(
            params, profile, time_series, train_fraction, val_fraction, seed
        )
        val_scores.append(mse_val)
        test_scores.append(mse_test)

    median_val, std_val, cv_val = _aggregate(val_scores)
    median_test, std_test, cv_test = _aggregate(test_scores)

    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg = params
    return {
        'model_type': 'Classical_ESN',
        'data_profile': profile['name'],
        'median_val_mse': median_val,
        'std_val_mse': std_val,
        'cv_val_mse': cv_val,
        'median_test_mse': median_test,
        'std_test_mse': std_test,
        'cv_test_mse': cv_test,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'sparsity': sparsity,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': 10,
        'base_seed': base_seed
    }
