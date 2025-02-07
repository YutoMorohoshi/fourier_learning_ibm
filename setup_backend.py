import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import os
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import pickle
import networkx as nx
from datetime import datetime, timezone
import json
import math


def setup_backend(qpu_name: str = "ibm_fez", method: str = None):
    ########################################################
    # Option1: Use IBM Quantum backend.
    ########################################################

    # Set up the Qiskit Runtime service (this is a one-time setup)
    # QiskitRuntimeService.save_account(
    #     token="YOUR_API_TOKEN",
    #     channel="ibm_quantum",
    # )

    # Load saved credentials
    service = QiskitRuntimeService()
    # backend_qpu = service.least_busy(simulator=False, interactional=True)
    backend_qpu = service.backend(qpu_name)
    print(f"Using backend QPU: {backend_qpu}")

    ########################################################
    # Option2: Use local AerSimulator as the backend.
    ########################################################

    # Noiseless simulator
    if method == "matrix_product_state":
        backend_sim_noiseless = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=100,
            matrix_product_state_truncation_threshold=1e-8,
        )
    else:
        backend_sim_noiseless = AerSimulator()
    print(f"Using backend noiseless simulator: {backend_sim_noiseless}")
    print()

    # Noise model
    noise_backend = NoiseModel.from_backend(backend_qpu)
    print(noise_backend)
    print()

    # Noisy simulator
    if method == "matrix_product_state":
        backend_sim_noisy = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=100,
            matrix_product_state_truncation_threshold=1e-8,
            noise_model=noise_backend,
        )
    else:
        backend_sim_noisy = AerSimulator(noise_model=noise_backend)
    print(f"Using backend noisy simulator: {backend_sim_noisy}")

    return backend_qpu, backend_sim_noiseless, backend_sim_noisy
