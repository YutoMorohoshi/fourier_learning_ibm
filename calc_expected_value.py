import numpy as np
import scipy
import time
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp, Statevector
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import tebd
from heisenberg import (
    HeisenbergModel,
    get_graph,
)

import cupy as cp
import cupyx.scipy.linalg

# import cupyx.scipy.sparse as cusparse
# import cupyx.scipy.sparse.linalg as cusparse_linalg
import time
import numpy as np


def imag_tebd(model_params, L, beta):
    chi_max = 100
    M = SpinChain(model_params)
    dt = 0.01

    # prepare the initial state
    # |psi> = |0011...1100> (center qubits are 1 and the rest are 0)
    product_state = ["up"] * (L // 4) + ["down"] * (L // 2) + ["up"] * (L // 4)
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, "finite")

    # configure the TEBD engine
    tebd_params = {
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": 1e-8,
        },
    }
    eng = tebd.TEBDEngine(psi, M, tebd_params)

    # calculate |psi'> = exp(-beta/2 * H) |psi>
    start = time.time()
    for _ in range(int(beta / 2 / dt)):
        eng.calc_U(order=2, delta_t=dt, type_evo="imag")
        eng.update_imag(N_steps=1)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for TEBD: {elapsed_time:.2f}[s]")

    # calculate the expected value as overlap of |psi'>, i.e., <psi'|psi'> = <psi|exp(-beta * H)|psi>
    return eng.psi.overlap(eng.psi)


def full_diag(n_qubits, Js, beta):
    # prepare the initial state
    # |psi> = |0011...1100> (center qubits are 1 and the rest are 0)
    index = ["0"] * (n_qubits // 4) + ["1"] * (n_qubits // 2) + ["0"] * (n_qubits // 4)
    index = "".join(index)
    psi = Statevector.from_label(index)

    # prepare the Hamiltonian
    G = get_graph(n_qubits, Js)
    heisenberg = HeisenbergModel(n_qubits, G)
    H = heisenberg.H

    # state is big endian, so we need to reverse the qubits of the Hamiltonian
    H = Operator(H).reverse_qargs().to_matrix()
    H = scipy.sparse.csr_matrix(H)

    # calculate f(H) = exp(-beta * H)
    start = time.time()
    fH = scipy.sparse.linalg.expm(-beta * H)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for diagonalization: {elapsed_time:.2f}[s]")

    # calculate the expected value <psi|exp(-beta * H)|psi>
    expected_value = np.vdot(psi, fH @ psi).real

    return expected_value


def full_diag_gpu(n_qubits, Js, beta):
    # 初期状態の準備
    # |psi> = |0011...1100> (中心の量子ビットが1で、その他は0)
    index = ["0"] * (n_qubits // 4) + ["1"] * (n_qubits // 2) + ["0"] * (n_qubits // 4)
    index = "".join(index)
    psi = Statevector.from_label(index)

    # GPU上で計算するため、状態ベクトルを cupy 配列に変換
    psi_gpu = cp.asarray(psi.data)

    # ハミルトニアンの準備
    G = get_graph(n_qubits, Js)
    heisenberg = HeisenbergModel(n_qubits, G)
    # H = heisenberg.H
    H = heisenberg.get_hamiltonian()

    # state is big endian, so we need to reverse the qubits of the Hamiltonian
    H = Operator(H).reverse_qargs().to_matrix()

    # cupy の密行列に変換
    H = cp.asarray(H)

    # # calculate f(H) = exp(-beta * H)
    start = time.time()
    fH = cupyx.scipy.linalg.expm(-beta * H)  # GPU 上で計算

    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for diagonalization: {elapsed_time:.2f}[s]")

    # 期待値 <psi|exp(-beta * H)|psi> を計算
    expected_value = cp.vdot(psi_gpu, fH @ psi_gpu).real

    # 必要に応じてホスト側の numpy スカラーに変換し、さらに float に変換
    return cp.asnumpy(expected_value).item()


def calculate_expected_value(L, Js):
    """
    L: int
        The number of sites
    Js: float array
        The array of the coupling constants
    Returns
    -------
    data_tebd: float
        The expected_value of the final state of the TEBD
    """
    beta = 1.0

    L_diag_upper_bound = 12

    # モデルの設定
    model_params = {
        "L": L,
        "S": 0.5,
        "conserve": "Sz",  # 粒子数保存
        # TenPy では 1/2 スピンのモデルを考えるため、厳密対角化との帳尻を合わせるために Js は 4 倍する
        "Jx": Js * 4,
        "Jy": Js * 4,
        "Jz": Js * 4,
        "hz": 0,
        "bc_MPS": "finite",
    }

    expected_value_tebd = imag_tebd(model_params, L, beta)

    if L <= L_diag_upper_bound:
        # expected_value_diag = full_diag(n_qubits=L, Js=Js, beta=beta)
        expected_value_diag = full_diag_gpu(n_qubits=L, Js=Js, beta=beta)
        diff = abs(expected_value_tebd - expected_value_diag)
    else:
        expected_value_diag = None
        diff = None
    print()

    return expected_value_tebd, expected_value_diag, diff
