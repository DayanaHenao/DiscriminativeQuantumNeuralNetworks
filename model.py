import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
import math

"""

Module that contains the quantum circuit (POVM-like classifier) and
helper utilities used by the notebooks and training scripts.

Functions and Objects:
- dev: the PennyLane device used by the QNode
- Circuit(params, rho): QNode returning outcome probabilities for a 4-wire circuit
- rho1(a), rho2(b): state constructors for the two families (return (rho, label))
- sample_loss, loss: per-sample and aggregated loss (requires alpha_err, alpha_inc)
- get_probabilities: wrapper around Circuit
- get_batches, batch_generator: simple RNG-backed batching utilities
- evaluate_model: compute average success/error/inconclusive rates on a dataset
- predict_label, confusion_matrix_inconclusive: helpers for prediction/confusion

Notes:
- The QNode uses the "default.mixed" device (density matrix support).
- We use pennylane.numpy (pnp) for arrays that participate in autodiff with PennyLane.
"""

# Device: density-matrix simulator with 4 wires
dev = qml.device("default.mixed", wires=4)


@qml.qnode(dev, interface="autograd")
def Circuit(params, rho):
    """QNode that prepares a 4-qubit density matrix and applies a parameterized
    circuit to realize a POVM-like classifier.

    Inputs:
    - params: array-like of 30 parameters (used by a collection of Rot and
      controlled operations in the circuit).
    - rho: 4x4 density matrix describing the (two-qubit) input family state.

    Returns:
    - probabilities over the measurement outcomes on wires [0,1] (length 4).

    Implementation notes:
    - The global state fed to QubitDensityMatrix is formed as kron(|0><0|, rho),
      where the first two wires (0,1) are the ancilla / measurement qubits.
    - The circuit intentionally mirrors the structure used in the notebooks and
      should not be changed lightly (it encodes the POVM via unitaries + ancilla).
    """
    # Construct |00><00| on wires (0,1) and place `rho` on wires (2,3) via kron
    base = pnp.zeros((4, 4), dtype=complex)
    base[0, 0] = 1.0
    full_state = pnp.kron(base, rho)

    # Load the density matrix into the device (wires 0,1,2,3)
    qml.QubitDensityMatrix(full_state, wires=[0, 1, 2, 3])

    # Parameterized single-qubit rotations
    qml.Rot(params[0], params[1], params[2], wires=0)
    qml.Rot(params[3], params[4], params[5], wires=1)
    qml.Rot(params[6], params[7], params[8], wires=2)
    qml.Rot(params[9], params[10], params[11], wires=3)

    # Entangling / controlled structure (kept as in original notebook)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[0, 3])
    qml.CNOT(wires=[3, 0])

    qml.PauliX(wires=0)

    # Controlled rotations conditioned on wire 0
    qml.ctrl(qml.Rot(params[12], params[13], params[14], wires=1), control=0)
    qml.ctrl(qml.Rot(params[15], params[16], params[17], wires=2), control=0)
    qml.ctrl(qml.Rot(params[18], params[19], params[20], wires=3), control=0)

    # Controlled CNOTs (again, mirroring the original model)
    qml.ctrl(qml.CNOT(wires=[1, 2]), control=0)
    qml.ctrl(qml.CNOT(wires=[1, 3]), control=0)
    qml.ctrl(qml.CNOT(wires=[3, 1]), control=0)

    qml.PauliX(wires=0)

    # Second block of controlled rotations
    qml.ctrl(qml.Rot(params[21], params[22], params[23], wires=1), control=0)
    qml.ctrl(qml.Rot(params[24], params[25], params[26], wires=2), control=0)
    qml.ctrl(qml.Rot(params[27], params[28], params[29], wires=3), control=0)

    # Final controlled entangling operations
    qml.ctrl(qml.CNOT(wires=[1, 2]), control=0)
    qml.ctrl(qml.CNOT(wires=[1, 3]), control=0)
    qml.ctrl(qml.CNOT(wires=[3, 1]), control=0)

    # Return probabilities on the measurement wires (wire indices 0 and 1)
    return qml.probs(wires=[0, 1])


# State / family constructors
def rho1(a):
    """Construct a pure-state density matrix for family 1.

    The underlying state is |psi> = [sqrt(1-a^2), 0, a, 0]^T. Returns (rho, label).
    """
    psi1 = pnp.array([pnp.sqrt(1 - a * a), 0.0, a, 0.0], dtype=complex)
    rho = pnp.outer(psi1, pnp.conj(psi1))
    return rho, 1


def rho2(b):
    """Construct the (mixed) density matrix for family 2.

    Family 2 is modeled as the equal mixture of two pure states which differ by
    a relative sign; returns (rho, label=2).
    """
    psi2 = pnp.array([0.0, pnp.sqrt(1 - b * b), b, 0.0], dtype=complex)
    psi3 = pnp.array([0.0, -pnp.sqrt(1 - b * b), b, 0.0], dtype=complex)
    rho = 0.5 * pnp.outer(psi2, pnp.conj(psi2)) + 0.5 * pnp.outer(psi3, pnp.conj(psi3))
    return rho, 2


# Loss functions
# The loss is composed of three terms: (i) penalty for failing to succeed,
# (ii) penalty for erroneous classification, and (iii) penalty for producing
# an inconclusive outcome. The notebook / training script passes `alpha_err`
# and `alpha_inc` to weigh the error and inconclusive penalties.
def sample_loss(params, rho, label, alpha_err, alpha_inc, eps=1e-8):
    """Compute the (smoothed) loss for a single (rho, label) example.

    - params: circuit parameters
    - rho: 4x4 density matrix
    - label: 1 or 2 indicating the true family
    - alpha_err, alpha_inc: scalar weights applied to the error/inconclusive terms
    - eps: small constant used for smoothing the absolute value (to avoid
      non-differentiable points at 0)
    """
    probs = Circuit(params, rho)
    # Map the probability vector to success/error/inconclusive depending on true label
    if label == 1:
        p_suc = probs[0] + probs[2]
        p_err = probs[1]
        p_inc = probs[3]
    else:
        p_suc = probs[1]
        p_err = probs[0] + probs[2]
        p_inc = probs[3]

    def smooth_abs(x, eps=1e-6):
        # Smooth approximation to |x|: sqrt(x^2 + eps)
        return pnp.sqrt(x**2 + eps)

    # Encourage p_suc -> 1, and penalize p_err and p_inc according to their alphas
    return smooth_abs(p_suc - 1.0, eps) + float(alpha_err) * smooth_abs(p_err, eps) + float(alpha_inc) * smooth_abs(p_inc, eps)


def loss(params, samples, alpha_err, alpha_inc):
    """Aggregate sample_loss over a list of (rho, label) samples.

    Returns the mean loss across the provided samples. The API requires the
    caller to pass alpha_err and alpha_inc explicitly for reproducibility.
    """
    total = pnp.array(0.0)
    for (rho, label) in samples:
        total = total + sample_loss(params, rho, label, alpha_err, alpha_inc)
    return total / len(samples)


def get_probabilities(params, rho):
    """Convenience wrapper: return Circuit(params, rho) probabilities."""
    return Circuit(params, rho)


# Batching utilities 
def get_batches(data, batch_size, rng=None):
    """Return a list of shuffled batches (one epoch) using a numpy RNG.

    If rng is None a new default_rng() is created. The returned value is a
    Python list of lists; each inner list contains up to batch_size elements.
    """
    batch_size = max(1, int(batch_size))
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.permutation(len(data))
    shuffled = [data[i] for i in idx]
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


def batch_generator(data, batch_size, rng):
    """Infinite generator that yields shuffled batches every epoch.

    Useful for training loops that iterate over many epochs and want a fresh
    shuffle each epoch. If the dataset is empty this yields empty batches.
    """
    batch_size = max(1, int(batch_size))
    n = len(data)
    if n == 0:
        while True:
            yield []
    while True:
        idx = rng.permutation(n)
        shuffled = [data[i] for i in idx]
        for i in range(0, n, batch_size):
            yield shuffled[i:i + batch_size]


# Evaluation 
def evaluate_model(params, samples):
    """Compute average success / error / inconclusive rates on `samples`.

    `samples` is expected to be a list of (rho, label) tuples. The function
    returns a triple (suc, err, inc) with each value between 0 and 1.
    """
    S1 = [rho for (rho, label) in samples if label == 1]
    S2 = [rho for (rho, label) in samples if label == 2]

    total_suc_1 = total_err_1 = total_inc_1 = 0.0
    total_suc_2 = total_err_2 = total_inc_2 = 0.0

    # Sum probabilities for each class's examples
    for rho in S1:
        probs = get_probabilities(params, rho)
        total_suc_1 += float(probs[0] + probs[2])
        total_err_1 += float(probs[1])
        total_inc_1 += float(probs[3])

    for rho in S2:
        probs = get_probabilities(params, rho)
        total_suc_2 += float(probs[1])
        total_err_2 += float(probs[0] + probs[2])
        total_inc_2 += float(probs[3])

    n1, n2 = len(S1), len(S2)
    if (n1 + n2) == 0:
        return 0.0, 0.0, 0.0
    suc = (total_suc_1 + total_suc_2) / (n1 + n2)
    err = (total_err_1 + total_err_2) / (n1 + n2)
    inc = (total_inc_1 + total_inc_2) / (n1 + n2)
    return suc, err, inc


def predict_label(params, rho, actual_label=None):
    """Return a discrete prediction for a single example.

    The classifier chooses max over [p_fam1, p_fam2, p_inc] and returns 1, 2
    or 3 (3 indicates the inconclusive outcome).
    """
    probs = get_probabilities(params, rho)
    p_fam1 = float(probs[0] + probs[2])
    p_fam2 = float(probs[1])
    p_inc = float(probs[3])
    idx = int(np.argmax([p_fam1, p_fam2, p_inc]))
    return [1, 2, 3][idx]


def confusion_matrix_inconclusive(params, samples):
    """Build a confusion matrix for the dataset including an 'inconclusive' col.

    Returns a 2x3 integer array where rows correspond to true labels (1,2) and
    columns correspond to predicted [1,2,3(inconclusive)].
    """
    cm = np.zeros((2, 3), dtype=int)
    for (rho, actual_label) in samples:
        pred = predict_label(params, rho, actual_label)
        row = actual_label - 1
        col = pred - 1
        cm[row, col] += 1
    return cm