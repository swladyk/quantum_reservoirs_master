# src/models.py

import pennylane as qml
from pennylane import numpy as np


def get_q_device(n_qubits):
    """
    Creates a PennyLane quantum device.

    Parameters
    ----------
    n_qubits : int
        The number of qubits for the device.

    Returns
    -------
    qml.Device
        A PennyLane device instance.
    """
    return qml.device("default.qubit", wires=n_qubits, shots=None)


def quantum_feature_map(inputs, weights, biases, n_layers, n_qubits, dev):
    """
    Defines the quantum circuit that acts as a feature map.

    Parameters
    ----------
    inputs : np.ndarray
        Input data for the circuit.
    weights : np.ndarray
        Trainable weights for the quantum layers.
    biases : np.ndarray
        Biases added to the input data.
    n_layers : int
        Number of quantum layers in the circuit.
    n_qubits : int
        Number of qubits in the circuit.
    dev : qml.Device
        The PennyLane device to run the circuit on.

    Returns
    -------
    np.ndarray
        The expectation values of the observables, serving as quantum features.
    """
    @qml.qnode(dev)
    def circuit(inputs, weights, biases):
        for i in range(n_qubits):
            total_angle = inputs[i] + biases[i]
            qml.RX(total_angle, wires=i)

        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        observables = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
                      [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] + \
                      [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return observables

    return circuit(inputs, weights, biases)


def train_esn_reservoir(train_inputs, train_outputs, n_layers, n_qubits, leakage_rate, lambda_reg, seed, washout=100):
    """
    Trains the quantum reservoir computer.

    The first `washout` reservoir states are discarded before fitting the readout
    so that the regression sees only post-transient dynamics. The returned
    `quantum_features` matrix is likewise trimmed.

    Returns
    -------
    tuple
        (W_out, weights, biases, quantum_features_post_washout)
    """
    np.random.seed(seed)
    weights = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    biases = np.random.uniform(-0.5, 0.5, n_qubits)

    n_observables = 3 * n_qubits
    n_samples = len(train_inputs)
    dev = get_q_device(n_qubits)

    # 1. Compute classical ESN states
    classical_states = np.zeros((n_samples, n_qubits))
    current_classical_state = np.zeros(n_qubits)
    for t in range(n_samples):
        current_classical_state = (1 - leakage_rate) * current_classical_state + leakage_rate * train_inputs[t]
        classical_states[t] = current_classical_state

    # 2. Map classical states to quantum features
    quantum_features = np.zeros((n_samples, n_observables))
    for t in range(n_samples):
        quantum_features[t] = quantum_feature_map(
            inputs=classical_states[t], weights=weights, biases=biases,
            n_layers=n_layers, n_qubits=n_qubits, dev=dev
        )

    # 3. Drop transient and train the readout layer (Ridge Regression)
    quantum_features = quantum_features[washout:]
    R = quantum_features
    Y = train_outputs[washout:].reshape(-1, 1)
    I = np.identity(n_observables)
    W_out = np.linalg.solve(R.T @ R + lambda_reg * I, R.T @ Y).flatten()

    return W_out, weights, biases, quantum_features


def predict_esn(test_inputs, weights, biases, W_out, n_layers, n_qubits, leakage_rate, washout=100):
    """
    Makes one-step-ahead predictions using the trained QRC-ESN model.

    The first `washout` inputs only advance the reservoir state — no prediction is
    computed for them. The returned array has length len(test_inputs) - washout.
    """
    predictions = []
    current_classical_state = np.zeros(n_qubits)
    dev = get_q_device(n_qubits)

    for t, input_val in enumerate(test_inputs):
        current_classical_state = (1 - leakage_rate) * current_classical_state + leakage_rate * input_val
        if t < washout:
            continue
        q_features = quantum_feature_map(
            inputs=current_classical_state, weights=weights, biases=biases,
            n_layers=n_layers, n_qubits=n_qubits, dev=dev)
        y_pred = np.dot(W_out, q_features)
        predictions.append(y_pred)

    return np.array(predictions)



# --- Classical ESN Model ---

def initialize_classical_reservoir(reservoir_size, input_dim, spectral_radius=0.9, sparsity=0.1, seed=2025):
    """
    Initializes the weight matrices for a classical Echo State Network.

    Parameters
    ----------
    reservoir_size : int
        The number of neurons in the reservoir.
    input_dim : int
        The dimensionality of the input signal (e.g., window_size).
    spectral_radius : float, optional
        The spectral radius of the reservoir weight matrix, by default 0.9.
    sparsity : float, optional
        The fraction of connections to set to zero in the reservoir matrix, by default 0.1.
    seed : int, optional
        Seed for the random number generator, by default 2025.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the input weight matrix (W_in) and the reservoir weight matrix (W_res).
    """
    np.random.seed(seed)

    W_in = np.random.uniform(-0.1, 0.1, (reservoir_size, input_dim))

    W_res = np.random.randn(reservoir_size, reservoir_size)
    W_res[np.random.rand(*W_res.shape) < sparsity] = 0

    eigenvalues = np.linalg.eigvals(W_res)
    current_spectral_radius = np.max(np.abs(eigenvalues))
    if current_spectral_radius > 1e-9:  # Avoid division by zero
        W_res = (spectral_radius / current_spectral_radius) * W_res

    return W_in, W_res


def update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate):
    """
    Updates the state of the classical reservoir for a single time step.

    Parameters
    ----------
    input_seq : np.ndarray
        The input vector at the current time step.
    W_in : np.ndarray
        The input weight matrix.
    W_res : np.ndarray
        The reservoir weight matrix.
    reservoir_state : np.ndarray
        The current state of the reservoir.
    leakage_rate : float
        The leakage rate (alpha) of the reservoir.

    Returns
    -------
    np.ndarray
        The updated reservoir state.
    """
    return (1 - leakage_rate) * reservoir_state + \
           leakage_rate * np.tanh(W_in @ input_seq + W_res @ reservoir_state)


def train_classical_reservoir(train_inputs, train_outputs, W_in, W_res, reservoir_size, leakage_rate, lambda_reg=1e-6, washout=100):
    """
    Trains the readout layer of the classical ESN.

    The first `washout` reservoir states are discarded before fitting the readout
    so that the regression sees only post-transient dynamics.
    """
    reservoir_states = []
    reservoir_state = np.zeros(reservoir_size)

    for input_seq in train_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate)
        reservoir_states.append(reservoir_state)

    R = np.vstack(reservoir_states[washout:])
    Y = train_outputs[washout:].reshape(-1, 1)
    I = np.identity(reservoir_size)
    W_out = np.linalg.solve(R.T @ R + lambda_reg * I, R.T @ Y).T

    return W_out, reservoir_state


def predict_esn_classical(test_inputs, W_in, W_res, W_out, reservoir_size, leakage_rate, initial_state, washout=100):
    """
    Makes one-step-ahead predictions with the trained classical ESN.

    The first `washout` inputs only advance the reservoir state — no prediction is
    computed for them. The returned array has length len(test_inputs) - washout.
    """
    predictions = []
    reservoir_state = initial_state.copy()

    for t, input_seq in enumerate(test_inputs):
        reservoir_state = update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate)
        if t < washout:
            continue
        y_pred = (W_out @ reservoir_state)[0]
        predictions.append(y_pred)

    return np.array(predictions)


def get_classical_reservoir_states(train_inputs, reservoir_size, leakage_rate, spectral_radius, sparsity, seed, input_dim, washout=100):
    """
    Drives the classical reservoir with training data and returns its post-washout
    internal states (first `washout` transient rows are discarded).
    """
    W_in_c, W_res_c = initialize_classical_reservoir(reservoir_size, input_dim, spectral_radius, sparsity, seed)
    reservoir_states = []
    reservoir_state = np.zeros(reservoir_size)

    for input_seq in train_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in_c, W_res_c, reservoir_state, leakage_rate)
        reservoir_states.append(reservoir_state)

    return np.vstack(reservoir_states[washout:])