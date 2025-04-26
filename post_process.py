import itertools


def get_prob0(result, n_qubits):
    """
    Extract the probability of measuring all qubits in state |0> from the result object.
    Args:
        result: The result object from the quantum circuit execution.
        n_qubits: The number of qubits in the circuit.
    Returns:
        prob0: The probability of measuring all qubits in state |0>.
    """
    meas_counts = result.data.meas.get_counts()
    num_shots = result.data.meas.num_shots

    prob0 = meas_counts.get("0" * n_qubits, 0) / num_shots

    return prob0


def extract_probs(probs_dict, probs_dict_qpu):
    """
    Extracts the probabilities from the given dictionaries.
    Args:
        probs_dict: A dictionary containing the probabilities from the simulator.
        probs_dict_qpu: A dictionary containing the probabilities from the QPU.
    Returns:
        A list of probabilities extracted from the dictionaries.
    """
    probs_extracted = []

    for sample_id, probs in probs_dict.items():
        if sample_id in probs_dict_qpu.keys():
            probs_extracted.append(probs.values())

    return list(itertools.chain.from_iterable(probs_extracted))
