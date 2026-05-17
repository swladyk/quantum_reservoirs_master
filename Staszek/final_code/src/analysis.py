# src/analysis.py

import numpy as np
from sklearn.decomposition import PCA
from .data_generation import (
    create_io_pairs, make_io_pairs, series_length, slice_series,
    mackey_glass, generate_arma_data, generate_arma_input_driven_data,
    generate_narma_data, generate_narma_input_driven_data,
)
from .models import train_esn_reservoir, get_classical_reservoir_states


# --- Default profile generator registry ---
# Includes BOTH legacy autoregressive names AND new input-driven names so the
# NMSE helper works against either CSV vintage.
DEFAULT_PROFILE_GENERATORS = {
    'Mackey_Glass_(tau=17)':  lambda: mackey_glass(tau=17),
    'Mackey_Glass_(tau=30)':  lambda: mackey_glass(tau=30),
    'Mackey_Glass_(tau=100)': lambda: mackey_glass(tau=100),
    # Legacy autoregressive
    'ARMA_1_2_stochastic':    lambda: generate_arma_data(),
    'NARMA10_Chaotic':        lambda: generate_narma_data(order=10),
    'NARMA5_Chaotic':         lambda: generate_narma_data(order=5),
    # New input-driven
    'ARMA_InputDriven':       lambda: generate_arma_input_driven_data(),
    'NARMA10_InputDriven':    lambda: generate_narma_input_driven_data(order=10),
    'NARMA5_InputDriven':     lambda: generate_narma_input_driven_data(order=5),
}


def compute_nmse_columns(results_df, profile_generators=None,
                         train_fraction=0.8, n_splits=5, washout=None):
    """
    Adds NMSE columns to `results_df`. Returns (modified_df, var_cache).

    Adds: `test_var`, `cv_var`, `median_test_nmse`, `median_cv_nmse`.

    NMSE is computed as MSE / Var(targets), where the target variance is
    deterministic given (data_profile, window_size). Time series are regenerated
    from the profile_generators; both autoregressive and input-driven generators
    are supported via the make_io_pairs / slice_series dispatch.
    """
    # Local import to avoid circular dependency (experiment imports data_generation)
    from .experiment import sliding_cv_folds, WASHOUT as _DEFAULT_WASHOUT

    if profile_generators is None:
        profile_generators = DEFAULT_PROFILE_GENERATORS
    if washout is None:
        washout = _DEFAULT_WASHOUT

    profiles_in_csv = set(results_df['data_profile'].unique())
    unique_ws = sorted(results_df['window_size'].astype(int).unique())

    var_cache = {}
    for profile_name, gen_fn in profile_generators.items():
        if profile_name not in profiles_in_csv:
            continue
        ts = gen_fn()
        n = series_length(ts)
        cv_end = int(n * train_fraction)
        cv_pool = slice_series(ts, 0, cv_end)
        test_data = slice_series(ts, cv_end, n)
        folds = sliding_cv_folds(cv_pool, n_splits)

        for ws in unique_ws:
            _, y_test = make_io_pairs(test_data, ws)
            var_test = float(np.var(y_test[washout:]))
            y_cv_pool = np.concatenate([
                make_io_pairs(val, ws)[1][washout:] for _, val in folds
            ])
            var_cv = float(np.var(y_cv_pool))
            var_cache[(profile_name, ws)] = {'test': var_test, 'cv': var_cv}

    def _lookup(row, key):
        entry = var_cache.get((row['data_profile'], int(row['window_size'])))
        return entry[key] if entry else np.nan

    results_df = results_df.copy()
    results_df['test_var'] = results_df.apply(lambda r: _lookup(r, 'test'), axis=1)
    results_df['cv_var']   = results_df.apply(lambda r: _lookup(r, 'cv'),   axis=1)
    results_df['median_test_nmse'] = results_df['median_test_mse'] / results_df['test_var']
    results_df['median_cv_nmse']   = results_df['median_cv_mse']   / results_df['cv_var']
    return results_df, var_cache

def get_qrc_feature_space(params, time_series, train_fraction, seed, washout=100):
    """
    Generates the post-washout quantum feature space for a given set of QRC parameters.
    Works for both autoregressive (1D array) and input-driven ((s, y) tuple) tasks.
    """
    leakage, lambda_r, win_size, layers, lag = params
    n_qubits = win_size
    train_size = int(series_length(time_series) * train_fraction)
    train_segment = slice_series(time_series, 0, train_size)

    train_inputs, train_outputs = make_io_pairs(train_segment, win_size, lag)

    # We only need the quantum_features, so we ignore the other return values
    _, _, _, quantum_features = train_esn_reservoir(
        train_inputs, train_outputs, layers, n_qubits, leakage, lambda_r, seed,
        washout=washout
    )
    return quantum_features

def calculate_effective_dimension(feature_matrix, variance_threshold=0.95):
    """
    Calculates the effective dimensionality of a feature space using PCA.
    
    The effective dimension is the number of principal components needed to
    explain a certain amount of the total variance.
    """
    if feature_matrix is None or feature_matrix.shape[0] < 2:
        return np.nan # Cannot perform PCA on empty or single-sample data
        
    pca = PCA()
    pca.fit(feature_matrix)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the first index where cumulative variance exceeds the threshold
    eff_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    return eff_dim