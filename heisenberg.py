import numpy as np
import scipy
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    SamplerV2 as Sampler,
    EstimatorV2 as Estimator,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator


def get_n_steps(t):
    # interval per division
    max_delta_t = 0.1

    n_steps = int(t / max_delta_t) + 1

    return n_steps


class HeisenbergModel:
    def __init__(self, n_qubits, Js, backend=None):
        self.n_qubits = n_qubits
        self.Js = Js
        self.first_half_pairs = [(i, i + 1) for i in range(0, self.n_qubits - 1, 2)]
        self.second_half_pairs = [(i, i + 1) for i in range(1, self.n_qubits - 1, 2)]
        self.pairs = self.first_half_pairs + self.second_half_pairs

        self.H = self.get_hamiltonian()

        self.backend = backend

    def get_pauli_strings(self):
        paulis = ["X", "Y", "Z"]
        strings = []

        for i, j in self.pairs:
            for pauli in paulis:
                pauli_string = ["I"] * self.n_qubits
                pauli_string[i] = pauli
                pauli_string[j] = pauli
                strings.append("".join(pauli_string))

        return strings

    def get_hamiltonian(self):
        pauli_strings = self.get_pauli_strings()
        coefficients = np.repeat(self.Js, 3)  # Js を各パウリ演算子に対応させる

        hamiltonian = SparsePauliOp.from_list(
            [(pauli_string, J) for pauli_string, J in zip(pauli_strings, coefficients)]
        )

        return hamiltonian.simplify().to_matrix()

    def add_heisenberg_interaction(self, qc, pairs, Js, t):
        for (i, j), J in zip(pairs, Js):
            # Function to add Heisenberg interactions between specific pairs of qubits
            # this corresponds to exp(-i J t (X_i X_j + Y_i Y_j + Z_i Z_j))
            theta = 2 * J * t
            qc.cx(i, j)
            qc.rx(theta, i)
            qc.rz(theta, j)

            # Note: order of qubits
            qc.rzx(-theta, j, i)

            qc.cx(i, j)

    def get_ghz_circuit(self, phase=0):
        """Prepare the GHZ state on the first half of the qubits.
        phase = 0 corresponds to |0...0> + |1...1>
        phase = 1 corresponds to |0...0> - |1...1>
        phase = 2 corresponds to |0...0> + i|1...1>
        phase = 3 corresponds to |0...0> - i|1...1>

        Args:
            phase (int): phase of the GHZ state (0, 1, 2, or 3)
        """
        ghz_n_qubits = self.n_qubits // 2

        qc = QuantumCircuit(ghz_n_qubits)
        qc.h(0)
        if phase == 1:
            qc.z(0)
        elif phase == 2:
            qc.s(0)
        elif phase == 3:
            qc.s(0)
            qc.z(0)
        for i in range(ghz_n_qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def exact_simulation(self, t, phase=0):
        """Compute the exact evolution of the Heisenberg model using the GHZ state.

        Args:
            t (float): time for the evolution

        Returns:
            float: probability of measuring the |0...0> state
        """
        initial_state = np.zeros(2**self.n_qubits, dtype=complex)
        initial_state[0] = 1
        U = scipy.linalg.expm(-1j * self.H * t)

        ghz_circuit = self.get_ghz_circuit(phase=0)
        ghz_unitary = np.kron(
            Operator(ghz_circuit).data,
            np.eye(2 ** (self.n_qubits - self.n_qubits // 2)),
        )

        ghz_circuit_with_phase = self.get_ghz_circuit(phase=phase)
        ghz_unitary_with_phase = np.kron(
            Operator(ghz_circuit_with_phase).data,
            np.eye(2 ** (self.n_qubits - self.n_qubits // 2)),
        )

        final_state = (
            np.conjugate(ghz_unitary_with_phase.T) @ U @ ghz_unitary @ initial_state
        )

        prob0 = np.abs(final_state[0]) ** 2

        return prob0

    def get_trotter_simulation_pub(self, total_time, n_steps, phase=0):
        t = total_time / n_steps
        first_half_Js = self.Js[: len(self.first_half_pairs)]
        second_half_Js = self.Js[len(self.first_half_pairs) :]

        qc = QuantumCircuit(self.n_qubits)

        # Create GHZ state
        qc.compose(self.get_ghz_circuit(phase=0), inplace=True)
        # qc.barrier()

        # Apply time-evolution operator
        self.add_heisenberg_interaction(qc, self.first_half_pairs, first_half_Js, t / 2)
        for _ in range(n_steps - 1):
            self.add_heisenberg_interaction(qc, self.first_half_pairs, first_half_Js, t)
            self.add_heisenberg_interaction(
                qc, self.second_half_pairs, second_half_Js, t
            )
        self.add_heisenberg_interaction(qc, self.second_half_pairs, second_half_Js, t)
        self.add_heisenberg_interaction(qc, self.first_half_pairs, first_half_Js, t / 2)
        # qc.barrier()

        # Uncompute GHZ state
        qc.compose(self.get_ghz_circuit(phase=phase).inverse(), inplace=True)

        # Measure
        qc.measure_all()

        # Convert to an ISA circuit and layout-mapped observables.
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        isa_qc = pm.run(qc)

        # if t == 0:
        #     print(" > Circuit for t=0")
        #     qc.draw("mpl")
        #     isa_qc.draw("mpl", idle_wires=False)

        # Only necessary for Estimator
        # isa_obs =

        return isa_qc


def get_prob0(result, n_qubits):
    if "0" * n_qubits not in result.data.meas.get_counts():
        print(" > No counts for |0...0> state")
        prob0 = 0
    else:
        prob0 = (
            result.data.meas.get_counts()["0" * n_qubits] / result.data.meas.num_shots
        )

    return prob0
