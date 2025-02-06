import itertools


def get_prob0(result, n_qubits):
    meas_counts = result.data.meas.get_counts()
    num_shots = result.data.meas.num_shots

    prob0 = meas_counts.get("0" * n_qubits, 0) / num_shots

    return prob0


def extract_probs(probs_dict, probs_dict_qpu):
    probs_extracted = []

    for sample_id, probs in probs_dict.items():
        if sample_id in probs_dict_qpu.keys():
            probs_extracted.append(probs.values())

    return list(itertools.chain.from_iterable(probs_extracted))
