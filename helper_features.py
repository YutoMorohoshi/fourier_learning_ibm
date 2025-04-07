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
from qiskit.visualization import plot_circuit_layout
import pickle
import networkx as nx
from datetime import datetime, timezone
import json
import math
import warnings
import time

warnings.filterwarnings("ignore")


def run_job(config, backend_qpu, sim_type="noiseless"):
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
    n_shots = config["n_shots"]
    n_features = config["n_features"]
    times = config["times"]
    all_Js = config["all_Js"]
    n_step_array = config["n_step_array"]
    backend = config["backend"]

    if sim_type == "noiseless":
        path = f"results/fourier_feature_sim_noiseless/"
    elif sim_type == "noisy":
        path = f"results/fourier_feature_sim_noisy/"
    elif sim_type == "qpu":
        path = f"results/fourier_feature_qpu/"

    initial_layout_52 = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        19,
        35,
        34,
        33,
        32,
        31,
        30,
        29,
        28,
        27,
        26,
        25,
        37,
        45,
        46,
        47,
        57,
        67,
        66,
        65,
        77,
        85,
        86,
        87,
        97,
        107,
        106,
        105,
        117,
        125,
        124,
        123,
        136,
        143,
        144,
        145,
    ]
    if n_qubits <= 15:
        initial_layout = list(
            range(n_qubits)
        )  # Use physical qubits [0, 1, ..., n_qubits-1]
    else:
        initial_layout = initial_layout_52[:n_qubits]

    # 保存用のファイルを初期化
    with open(path + "temp_progress.txt", "w") as f:
        f.write("")  # ファイルを空にする

    for i in range(n_samples):
        # 開始時刻を記録
        sample_start = time.time()
        # 途中経過を表示 + ファイルに保存
        progress_report = f"Preparing circuits for sample {i}/{n_samples}"
        with open(path + "temp_progress.txt", "a") as f:
            f.write(progress_report + "\n")

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

        # 各特徴量ごとの回路生成とトランスパイル時間を計測
        for k in range(n_features):
            feat_start = time.time()
            # Compute the Fourier features for different times
            n_step = n_step_array[k]
            circuit_phase0 = heisenberg.get_circuit(times[k], n_step, phase=0)
            circuit_phase1 = heisenberg.get_circuit(times[k], n_step, phase=1)
            circuit_phase2 = heisenberg.get_circuit(times[k], n_step, phase=2)
            circuit_phase3 = heisenberg.get_circuit(times[k], n_step, phase=3)

            exec_circuit_phase0 = transpile(
                circuit_phase0,
                backend_qpu,
                initial_layout=initial_layout,
            )
            exec_circuit_phase1 = transpile(
                circuit_phase1,
                backend_qpu,
                initial_layout=initial_layout,
            )
            exec_circuit_phase2 = transpile(
                circuit_phase2,
                backend_qpu,
                initial_layout=initial_layout,
            )
            exec_circuit_phase3 = transpile(
                circuit_phase3,
                backend_qpu,
                initial_layout=initial_layout,
            )
            circuits_phase0[f"sample{i}"][f"f_{k}"] = circuit_phase0
            circuits_phase1[f"sample{i}"][f"f_{k}"] = circuit_phase1
            circuits_phase2[f"sample{i}"][f"f_{k}"] = circuit_phase2
            circuits_phase3[f"sample{i}"][f"f_{k}"] = circuit_phase3
            exec_circuits_phase0[f"sample{i}"][f"f_{k}"] = exec_circuit_phase0
            exec_circuits_phase1[f"sample{i}"][f"f_{k}"] = exec_circuit_phase1
            exec_circuits_phase2[f"sample{i}"][f"f_{k}"] = exec_circuit_phase2
            exec_circuits_phase3[f"sample{i}"][f"f_{k}"] = exec_circuit_phase3

            feat_end = time.time()
            duration = feat_end - feat_start
            with open(path + "temp_progress.txt", "a") as f:
                f.write(
                    f"Sample {i}, feature {k}: generation and transpile took {duration:.2f} sec\n"
                )

            # 最初のサンプルの 5 つ目の特徴量のみをサンプリング
            if i == 0 and k == 5:
                # 使用する量子ビットの配置を pdf で保存
                plot_circuit_layout(
                    exec_circuit_phase0,
                    backend_qpu,
                    view="virtual",  # 論理量子ビットの配置
                ).savefig(path + f"{n_qubits}Q/circuit_layout_virtual.pdf")
                plot_circuit_layout(
                    exec_circuit_phase0,
                    backend_qpu,
                    view="physical",  # 物理量子ビットの配置
                ).savefig(path + f"{n_qubits}Q/circuit_layout_physical.pdf")

                # 量子回路を pdf で保存
                circuit_phase0.draw(
                    output="mpl",
                    idle_wires=False,
                    fold=-1,
                    filename=path + f"{n_qubits}Q/circuit.pdf",
                )
                exec_circuit_phase0.draw(
                    output="mpl",
                    idle_wires=False,
                    fold=-1,
                    filename=path + f"{n_qubits}Q/exec_circuit.pdf",
                )

                with open(path + "temp_progress.txt", "a") as f:
                    f.write(f"\nCircuit example:\n")
                    f.write(f"before transpile\n")
                    f.write(f"circuit depth: {circuit_phase0.depth()}\n")
                    f.write(f"count_ops: {circuit_phase0.count_ops()}\n\n")
                    f.write(f"after transpile\n")
                    f.write(f"circuit depth: {exec_circuit_phase0.depth()}\n")
                    f.write(f"count_ops: {exec_circuit_phase0.count_ops()}\n\n")
        sample_end = time.time()
        with open(path + "temp_progress.txt", "a") as f:
            f.write(
                f"Sample {i}: total circuit generation and transpile time {sample_end - sample_start:.2f} sec\n"
            )
    print()

    # Run jobs in batch
    job_ids = []  # For QPU
    jobs = (
        []
    )  # For AerSimulator. we can't use job ids with AerSimulator. Instead, we store the jobs in a list.

    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)

        for i in range(n_samples):
            start = time.time()
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

            # Run the circuits
            job = sampler.run(exec_circuits_per_sample, shots=n_shots)

            if "simulator" in backend.name:  # AerSimulator
                jobs.append(job)
            else:  # QPU
                job_ids.append(job.job_id())

            end = time.time()
            elapsed_time = end - start
            with open(path + "temp_progress.txt", "a") as f:
                f.write(
                    f"Submitted circuits to backend for sample {i}/{n_samples} in {elapsed_time:.2f} sec\n"
                )

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
    is_sim = True if "aer_simulator" in backend.name else False

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
