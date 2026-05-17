# src/experiment.py

import numpy as np
from sklearn.metrics import mean_squared_error

from .data_generation import (create_io_pairs, make_io_pairs,
                              series_length, slice_series)
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)

# Reservoir washout (transient burn-in) length, applied uniformly to
# training-state regression and to val/test prediction.
WASHOUT = 100


def _split_cv_pool_and_test(time_series, train_fraction):
    """
    Split a time series into (cv_pool, held_out_test).
    Works for both 1D arrays (autoregressive) and (s, y) tuples (input-driven).
    """
    n = series_length(time_series)
    cv_end = int(n * train_fraction)
    return (slice_series(time_series, 0, cv_end),
            slice_series(time_series, cv_end, n))


def sliding_cv_folds(cv_pool, n_splits):
    """
    Sliding-window time-series CV folds with fixed train_size.

    val_size   = len(cv_pool) // (n_splits + 1)
    train_size = (n_splits - 1) * val_size
    Folds slide evenly so the first fold starts at index 0 and the last fold's
    val ends at len(cv_pool).

    For n_splits=5 on a 1600-pt pool:
        val_size = 266, train_size = 1064, max_shift = 270
        shifts = [0, 67, 135, 202, 270]

    Works for both 1D arrays and (s, y) tuples.
    """
    n = series_length(cv_pool)
    val_size = n // (n_splits + 1)
    train_size = (n_splits - 1) * val_size
    max_shift = n - train_size - val_size

    if n_splits == 1:
        shifts = [0]
    else:
        shifts = [int(round(i * max_shift / (n_splits - 1))) for i in range(n_splits)]

    folds = []
    for shift in shifts:
        train_data = slice_series(cv_pool, shift, shift + train_size)
        val_data = slice_series(cv_pool, shift + train_size,
                                shift + train_size + val_size)
        folds.append((train_data, val_data))
    return folds


# --- Pojedynczy fold / pojedynczy test ---

def _qrc_fit_predict(params, train_data, eval_data, seed):
    """Train a QRC on train_data, evaluate on eval_data. Returns MSE."""
    leakage_rate, lambda_reg, window_size, n_layers, lag = params

    train_inputs, train_outputs = make_io_pairs(train_data, window_size, lag)
    eval_inputs, eval_outputs = make_io_pairs(eval_data, window_size, lag)

    W_out, weights, biases, _ = train_esn_reservoir(
        train_inputs, train_outputs, n_layers, window_size,
        leakage_rate, lambda_reg, seed, washout=WASHOUT
    )
    predictions = predict_esn(
        eval_inputs, weights, biases, W_out, n_layers,
        window_size, leakage_rate, washout=WASHOUT
    )
    return mean_squared_error(eval_outputs[WASHOUT:], predictions)


def _classical_fit_predict(params, train_data, eval_data, seed):
    """Train a Classical ESN on train_data, evaluate on eval_data. Returns MSE."""
    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg, window_size = params

    train_inputs, train_outputs = make_io_pairs(train_data, window_size)
    eval_inputs, eval_outputs = make_io_pairs(eval_data, window_size)

    W_in, W_res = initialize_classical_reservoir(
        reservoir_size, window_size, spectral_radius, sparsity, seed
    )
    W_out, final_state = train_classical_reservoir(
        train_inputs, train_outputs, W_in, W_res,
        reservoir_size, leakage_rate, lambda_reg, washout=WASHOUT
    )
    predictions = predict_esn_classical(
        eval_inputs, W_in, W_res, W_out,
        reservoir_size, leakage_rate, final_state, washout=WASHOUT
    )
    return mean_squared_error(eval_outputs[WASHOUT:], predictions)


# --- Wrappery agregujące wyniki ---

def _aggregate(scores):
    median = np.median(scores)
    std = np.std(scores)
    cv = std / median if median > 0 else 0
    return median, std, cv


def run_qrc_experiment_with_cv(params, profile, time_series,
                               train_fraction, n_splits,
                               base_seed, num_trials=11):
    """
    Sliding-window CV over n_splits folds and num_trials sub-seeds for a QRC.

    Returns a dict combining:
      - median/std/cv of n_splits * num_trials val MSEs from CV folds,
      - median/std/cv of num_trials test MSEs from Variant-A test eval
        (one re-training on the full CV pool per sub-seed).
    """
    cv_pool, test_data = _split_cv_pool_and_test(time_series, train_fraction)
    folds = sliding_cv_folds(cv_pool, n_splits)
    sub_seeds = [base_seed + i for i in range(num_trials)]

    cv_scores = []
    for seed in sub_seeds:
        for train_data, val_data in folds:
            cv_scores.append(_qrc_fit_predict(params, train_data, val_data, seed))

    test_scores = []
    for seed in sub_seeds:
        test_scores.append(_qrc_fit_predict(params, cv_pool, test_data, seed))

    median_cv, std_cv, cv_cv = _aggregate(cv_scores)
    median_test, std_test, cv_test = _aggregate(test_scores)

    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    return {
        'model_type': 'QRC',
        'data_profile': profile['name'],
        'median_cv_mse': median_cv,
        'std_cv_mse': std_cv,
        'cv_cv_mse': cv_cv,
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


def run_classical_experiment_with_cv(params, profile, time_series,
                                     train_fraction, n_splits,
                                     base_seed, num_trials=11):
    """
    Sliding-window CV over n_splits folds and num_trials sub-seeds for a Classical ESN.
    Same return contract as run_qrc_experiment_with_cv.
    """
    cv_pool, test_data = _split_cv_pool_and_test(time_series, train_fraction)
    folds = sliding_cv_folds(cv_pool, n_splits)
    sub_seeds = [base_seed + i for i in range(num_trials)]

    cv_scores = []
    for seed in sub_seeds:
        for train_data, val_data in folds:
            cv_scores.append(_classical_fit_predict(params, train_data, val_data, seed))

    test_scores = []
    for seed in sub_seeds:
        test_scores.append(_classical_fit_predict(params, cv_pool, test_data, seed))

    median_cv, std_cv, cv_cv = _aggregate(cv_scores)
    median_test, std_test, cv_test = _aggregate(test_scores)

    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg, window_size = params
    return {
        'model_type': 'Classical_ESN',
        'data_profile': profile['name'],
        'median_cv_mse': median_cv,
        'std_cv_mse': std_cv,
        'cv_cv_mse': cv_cv,
        'median_test_mse': median_test,
        'std_test_mse': std_test,
        'cv_test_mse': cv_test,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'sparsity': sparsity,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': window_size,
        'base_seed': base_seed
    }
