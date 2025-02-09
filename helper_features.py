import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import os
from heisenberg import (
    HeisenbergModel,
    get_graph,
)
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import pickle
import networkx as nx
from datetime import datetime, timezone
import json
import math
import warnings

warnings.filterwarnings("ignore")


def run_job(config):
    circuits_phase0 = {}
    circuits_phase1 = {}
    circuits_phase2 = {}
    circuits_phase3 = {}
    exec_circuits_phase0 = {}
    exec_circuits_phase1 = {}
    exec_circuits_phase2 = {}
    exec_circuits_phase3 = {}

    n_qubits = config["n_qubits"]
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    times = config["times"]
    all_Js = config["all_Js"]
    n_step_array = config["n_step_array"]
    backend = config["backend"]

    # Whether the backend is a simulator or not
    is_sim = True if backend.name == "aer_simulator" else False

    for i in range(n_samples):
        print(f"Preparing circuits for sample {i}/{n_samples}")
        Js = all_Js[i]
        G = get_graph(n_qubits, Js)

        heisenberg = HeisenbergModel(n_qubits, G)

        circuits_phase0[f"sample{i}"] = {}
        circuits_phase1[f"sample{i}"] = {}
        circuits_phase2[f"sample{i}"] = {}
        circuits_phase3[f"sample{i}"] = {}
        exec_circuits_phase0[f"sample{i}"] = {}
        exec_circuits_phase1[f"sample{i}"] = {}
        exec_circuits_phase2[f"sample{i}"] = {}
        exec_circuits_phase3[f"sample{i}"] = {}

        # Compute the Fourier features for different times
        for k in range(n_features):
            n_step = n_step_array[k]
            circuit_phase0 = heisenberg.get_circuit(times[k], n_step, phase=0)
            circuit_phase1 = heisenberg.get_circuit(times[k], n_step, phase=1)
            circuit_phase2 = heisenberg.get_circuit(times[k], n_step, phase=2)
            circuit_phase3 = heisenberg.get_circuit(times[k], n_step, phase=3)

            initial_layout = list(
                range(n_qubits)
            )  # Use physical qubits [0, 1, ..., n_qubits-1]
            exec_circuit_phase0 = transpile(
                circuit_phase0, backend, initial_layout=initial_layout
            )
            exec_circuit_phase1 = transpile(
                circuit_phase1, backend, initial_layout=initial_layout
            )
            exec_circuit_phase2 = transpile(
                circuit_phase2, backend, initial_layout=initial_layout
            )
            exec_circuit_phase3 = transpile(
                circuit_phase3, backend, initial_layout=initial_layout
            )

            circuits_phase0[f"sample{i}"][f"f_{k}"] = circuit_phase0
            circuits_phase1[f"sample{i}"][f"f_{k}"] = circuit_phase1
            circuits_phase2[f"sample{i}"][f"f_{k}"] = circuit_phase2
            circuits_phase3[f"sample{i}"][f"f_{k}"] = circuit_phase3
            exec_circuits_phase0[f"sample{i}"][f"f_{k}"] = exec_circuit_phase0
            exec_circuits_phase1[f"sample{i}"][f"f_{k}"] = exec_circuit_phase1
            exec_circuits_phase2[f"sample{i}"][f"f_{k}"] = exec_circuit_phase2
            exec_circuits_phase3[f"sample{i}"][f"f_{k}"] = exec_circuit_phase3
    print()

    # check a circuit
    sample_id = 2
    feature_id = 5
    print("Circuit example:")
    print("before transpile")
    print(
        f"circuit depth: {circuits_phase0[f'sample{sample_id}'][f'f_{feature_id}'].depth()}"
    )
    print(
        f"count_ops: {circuits_phase0[f'sample{sample_id}'][f'f_{feature_id}'].count_ops()}\n"
    )
    circuits_phase0[f"sample{sample_id}"][f"f_{feature_id}"].draw(
        output="mpl",
        idle_wires=False,
        fold=-1,  # fold=-1 is used to disable folding
    )
    print("after transpile")
    print(
        f"circuit depth: {exec_circuits_phase0[f'sample{sample_id}'][f'f_{feature_id}'].depth()}"
    )
    print(
        f"count_ops: {exec_circuits_phase0[f'sample{sample_id}'][f'f_{feature_id}'].count_ops()}\n"
    )
    exec_circuits_phase0[f"sample{sample_id}"][f"f_{feature_id}"].draw(
        output="mpl",
        idle_wires=False,
        fold=-1,  # fold=-1 is used to disable folding
    )

    # Run jobs in batch
    job_ids = []  # For QPU
    jobs = (
        []
    )  # For AerSimulator. we can't use job ids with AerSimulator. Instead, we store the jobs in a list.

    with Batch(backend=backend) as batch:
        sampler = Sampler()

        for i in range(n_samples):

            print(f"Submitting circuits to backend for sample {i}/{n_samples}")
            exec_circuits_per_sample = []
            exec_circuits_per_sample += [
                exec_circuits_phase0[f"sample{i}"][f"f_{k}"] for k in range(n_features)
            ]
            exec_circuits_per_sample += [
                exec_circuits_phase1[f"sample{i}"][f"f_{k}"] for k in range(n_features)
            ]
            exec_circuits_per_sample += [
                exec_circuits_phase2[f"sample{i}"][f"f_{k}"] for k in range(n_features)
            ]
            exec_circuits_per_sample += [
                exec_circuits_phase3[f"sample{i}"][f"f_{k}"] for k in range(n_features)
            ]

            job = sampler.run(exec_circuits_per_sample)

            if is_sim:
                jobs.append(job)
            else:
                job_ids.append(job.job_id())

    return batch, jobs, job_ids


def _get_prob0(result, n_qubits):
    meas_counts = result.data.meas.get_counts()
    num_shots = result.data.meas.num_shots

    prob0 = meas_counts.get("0" * n_qubits, 0) / num_shots

    return prob0


def get_features(config, jobs):
    data = []
    probs_phase0 = {}
    probs_phase1 = {}
    probs_phase2 = {}
    probs_phase3 = {}

    n_qubits = config["n_qubits"]
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    times = config["times"]
    backend = config["backend"]
    all_expected_values = config["all_expected_values"]

    # Whether the backend is a simulator or not
    is_sim = True if backend.name == "aer_simulator" else False

    # Whether the backend is noisy simulator or not
    is_sim_noisy = True if backend.options.noise_model else False

    for i in range(n_samples):
        print(f"Post-processing sample {i}/{n_samples}")
        features = []
        probs_phase0[f"sample{i}"] = {}
        probs_phase1[f"sample{i}"] = {}
        probs_phase2[f"sample{i}"] = {}
        probs_phase3[f"sample{i}"] = {}

        lambda_ref = np.sum(config["all_Js"][i])

        for k in range(n_features):
            # Get results of each phase in a batch
            results_phase0 = jobs[i].result()[:n_features]
            results_phase1 = jobs[i].result()[n_features : 2 * n_features]
            results_phase2 = jobs[i].result()[2 * n_features : 3 * n_features]
            results_phase3 = jobs[i].result()[3 * n_features :]

            prob_phase0 = _get_prob0(results_phase0[k], n_qubits)
            prob_phase1 = _get_prob0(results_phase1[k], n_qubits)
            prob_phase2 = _get_prob0(results_phase2[k], n_qubits)
            prob_phase3 = _get_prob0(results_phase3[k], n_qubits)

            probs_phase0[f"sample{i}"][f"f_{k}"] = prob_phase0
            probs_phase1[f"sample{i}"][f"f_{k}"] = prob_phase1
            probs_phase2[f"sample{i}"][f"f_{k}"] = prob_phase2
            probs_phase3[f"sample{i}"][f"f_{k}"] = prob_phase3

            inner_product = np.exp(-1j * lambda_ref * times[k]) * (
                (prob_phase0 - prob_phase1) + 1j * (prob_phase2 - prob_phase3)
            )

            features.append(inner_product.real)
            if k != 0:
                features.append(inner_product.imag)
        data.append([i, *features, all_expected_values[i]])

    # Create column names for the DataFrame
    columns = []
    columns.append("sample_id")
    for k in range(n_features):
        columns.append(f"f_{k} Re")
        if k != 0:
            columns.append(f"f_{k} Im")
    columns.append("expected_value")

    # Convert to a DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Config path
    if is_sim_noisy:  # noisy simulator
        path = f"results/fourier_feature_sim_noisy/{n_qubits}Q"
    elif is_sim:  # noiseless simulator
        path = f"results/fourier_feature_sim_noiseless/{n_qubits}Q"
    else:  # QPU
        path = f"results/fourier_feature_qpu/{n_qubits}Q"

    # Save the DataFrame
    df.to_json(f"{path}/features.json", orient="records", indent=4)

    # Save probabilties (for reference)
    with open(f"{path}/probs_phase0.json", "w") as f:
        json.dump(probs_phase0, f, indent=4)
    with open(f"{path}/probs_phase1.json", "w") as f:
        json.dump(probs_phase1, f, indent=4)
    with open(f"{path}/probs_phase2.json", "w") as f:
        json.dump(probs_phase2, f, indent=4)
    with open(f"{path}/probs_phase3.json", "w") as f:
        json.dump(probs_phase3, f, indent=4)

    return df
