import numpy as np
import scipy
import networkx as nx
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import cupy as cp
import cupyx.scipy.linalg as csl


def get_graph(n_qubits, Js):
    # Create a graph
    G = nx.Graph()

    # add nodes
    nodes = list(range(n_qubits))
    G.add_nodes_from(nodes)

    # add 'hadamard' attribute to the nodes
    for node in nodes:
        if node == n_qubits // 2 - 1:
            G.nodes[node]["hadamard"] = True
        else:
            G.nodes[node]["hadamard"] = False

    # add edges between nearest neighbors
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    G.add_edges_from(edges)

    # add 'J' attribute (interaction strength) to the edges
    for edge in G.edges:
        G.edges[edge]["J"] = Js[edge[0]]

    # add 'cnot' attribute to the edges
    # State |0011...1100> (center qubits are 1 and the rest are 0)
    leftmost = n_qubits // 4
    rightmost = leftmost + n_qubits // 2 - 1
    hadamard = list(filter(lambda x: G.nodes[x]["hadamard"], G.nodes))[0]
    left_start = hadamard
    right_start = hadamard

    for i in range(hadamard - leftmost + 1):
        if i == 0:
            # processing on the right side of Hadamard
            control = hadamard
            target = hadamard + 1
            right_start = target
            G.edges[(control, target)]["cnot"] = {
                "order": i,
                "control": control,
                "target": target,
            }
        else:
            # processing on the left side of Hadamard
            control = left_start
            target = left_start - 1

            # apply CNOT only if target is not less than leftmost
            if target >= leftmost:
                G.edges[(control, target)]["cnot"] = {
                    "order": i,
                    "control": control,
                    "target": target,
                }
                left_start = target

            # processing on the right side of Hadamard
            control = right_start
            target = right_start + 1

            # apply CNOT only if target is not greater than rightmost
            if target <= rightmost:
                G.edges[(control, target)]["cnot"] = {
                    "order": i,
                    "control": control,
                    "target": target,
                }
                right_start = target

        # add None of 'cnot_order' attribute to the other edges
        for edge in G.edges:
            if "cnot" not in G.edges[edge]:
                G.edges[edge]["cnot"] = {"order": None, "control": None, "target": None}

    return G


def get_positions(n_qubits):
    # define custom coordinates to place qubits at equal intervals
    positions = {i: (i, 0) for i in range(n_qubits)}

    return positions


class HeisenbergModel:
    def __init__(self, n_qubits, graph):
        self.n_qubits = n_qubits
        self.G = graph
        self.Js = [self.G.edges[edge]["J"] for edge in self.G.edges]

        # sparse big-endian matrix
        self.H = self.get_hamiltonian()

        # first_half_pairs starts from 1 if n_qubits % 4 == 0, otherwise starts from 0
        if self.n_qubits % 4 == 0:
            self.first_half_pairs = [(i, i + 1) for i in range(1, self.n_qubits - 1, 2)]
            self.first_Js = [self.Js[i] for i in range(1, self.n_qubits - 1, 2)]
            self.second_half_pairs = [
                (i, i + 1) for i in range(0, self.n_qubits - 1, 2)
            ]
            self.second_Js = [self.Js[i] for i in range(0, self.n_qubits - 1, 2)]
        else:
            self.first_half_pairs = [(i, i + 1) for i in range(0, self.n_qubits - 1, 2)]
            self.first_Js = [self.Js[i] for i in range(0, self.n_qubits - 1, 2)]
            self.second_half_pairs = [
                (i, i + 1) for i in range(1, self.n_qubits - 1, 2)
            ]
            self.second_Js = [self.Js[i] for i in range(1, self.n_qubits - 1, 2)]

    def get_pauli_strings(self):
        paulis = ["X", "Y", "Z"]
        strings = []

        for i, j in self.G.edges:
            for pauli in paulis:
                pauli_string = ["I"] * self.n_qubits
                pauli_string[i] = pauli
                pauli_string[j] = pauli

                # Convert to little-endian for Qiskit's SparsePauliOp
                # See: https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp#from_list
                # This is necessary because Qiskit uses little-endian order for qubits
                pauli_string.reverse()

                strings.append("".join(pauli_string))

        return strings

    def get_hamiltonian(self):
        pauli_strings = self.get_pauli_strings()
        coefficients = np.repeat(self.Js, 3)  # 3 for X, Y, Z

        H = SparsePauliOp.from_list(
            [(pauli_string, J) for pauli_string, J in zip(pauli_strings, coefficients)]
        )

        return H

    def add_heisenberg_interaction(self, qc, pairs, Js, t):
        for (i, j), J in zip(pairs, Js):
            # Check if the qubits are entangled
            is_entangled = (
                self.G.nodes[i]["is_entangled"] or self.G.nodes[j]["is_entangled"]
            )
            if not is_entangled:
                continue

            # Function to add Heisenberg interactions between specific pairs of qubits
            # this corresponds to exp(-i J t (X_i X_j + Y_i Y_j + Z_i Z_j))
            theta = 2 * J * t
            qc.cx(i, j)
            qc.rx(theta, i)
            qc.rz(theta, j)

            # Note: order of qubits
            qc.rzx(-theta, j, i)

            qc.cx(i, j)

            # update the 'is_entangled' attribute of the nodes
            self.G.nodes[i]["is_entangled"] = True
            self.G.nodes[j]["is_entangled"] = True

    def get_ghz_circuit(self, phase=0):
        """Prepare the GHZ state on the first half of the qubits.
        phase = 0 corresponds to |0...0> + |1...1>
        phase = 1 corresponds to |0...0> - |1...1>
        phase = 2 corresponds to |0...0> + i|1...1>
        phase = 3 corresponds to |0...0> - i|1...1>

        Args:
            phase (int): phase of the GHZ state (0, 1, 2, or 3)
        """
        max_order = max(
            [
                self.G.edges[edge]["cnot"]["order"]
                for edge in self.G.edges
                if self.G.edges[edge]["cnot"]["order"] is not None
            ]
        )

        # qc = QuantumCircuit(self.n_qubits // 2)
        qc = QuantumCircuit(self.n_qubits)

        # extract the node with 'hadamard' attribute
        hadamard_node = None
        for node in self.G.nodes:
            if self.G.nodes[node]["hadamard"]:
                hadamard_node = node
                break

        qc.h(hadamard_node)
        if phase == 1:
            qc.z(hadamard_node)
        elif phase == 2:
            qc.s(hadamard_node)
        elif phase == 3:
            qc.s(hadamard_node)
            qc.z(hadamard_node)
        # determine the order of cnot gates based on the 'cnot' 'order' attribute
        for i in range(max_order + 1):
            for j, k in self.G.edges:
                if self.G.edges[(j, k)]["cnot"]["order"] == i:
                    control = self.G.edges[(j, k)]["cnot"]["control"]
                    target = self.G.edges[(j, k)]["cnot"]["target"]
                    qc.cx(control, target)

                    # ノードの 'is_entangled' 属性を更新
                    self.G.nodes[j]["is_entangled"] = True
                    self.G.nodes[k]["is_entangled"] = True

        return qc

    def exact_simulation(self, t, phase=0):
        # Compute the exact evolution of the Heisenberg model using the GHZ state.
        # Exact means that we compute the matrix exponential of the Hamiltonian, not Trotterized.
        initial_state = Statevector.from_label("0" * self.n_qubits)

        U = scipy.sparse.linalg.expm(-1j * self.H.to_matrix(sparse=True) * t)

        ghz_circuit = self.get_ghz_circuit(phase=0)
        ghz_op = Operator.from_circuit(ghz_circuit)

        ghz_circuit_with_phase = self.get_ghz_circuit(phase=phase)
        ghz_op_with_phase = Operator.from_circuit(ghz_circuit_with_phase)

        U_all = ghz_op_with_phase.adjoint().data @ U @ ghz_op.data

        # initial_state is big endian, but when using evolve(), we don't need to reverse the qubits
        final_state = initial_state.evolve(U_all)

        return final_state, U_all

    def get_trotter_circuit(self, t, n_step):
        qc = QuantumCircuit(self.n_qubits)

        # Apply time-evolution
        self.add_heisenberg_interaction(qc, self.first_half_pairs, self.first_Js, t / 2)
        self.add_heisenberg_interaction(qc, self.second_half_pairs, self.second_Js, t)
        for _ in range(n_step - 1):
            self.add_heisenberg_interaction(qc, self.first_half_pairs, self.first_Js, t)
            self.add_heisenberg_interaction(
                qc, self.second_half_pairs, self.second_Js, t
            )
        self.add_heisenberg_interaction(qc, self.first_half_pairs, self.first_Js, t / 2)
        # qc.barrier()

        return qc

    def get_circuit(self, total_time, n_step, phase=0):
        t = total_time / n_step

        # add 'is_entangled' attribute to the nodes
        for node in self.G.nodes:
            self.G.nodes[node]["is_entangled"] = False

        qc = QuantumCircuit(self.n_qubits)

        # Create GHZ state
        qc.compose(self.get_ghz_circuit(phase=0), inplace=True)
        # qc.barrier()

        # Apply time-evolution
        qc.compose(self.get_trotter_circuit(t, n_step), inplace=True)
        # qc.barrier()

        # Uncompute GHZ state
        qc.compose(self.get_ghz_circuit(phase=phase).inverse(), inplace=True)
        # qc.barrier()

        # Measure
        qc.measure_all()

        return qc


class HeisenbergModelGPU(HeisenbergModel):
    def exact_simulation(self, t, phase=0):
        """
        Compute the exact evolution of the Heisenberg model using the GHZ state.
        Exact means that we compute the matrix exponential of the Hamiltonian, not Trotterized.

        Args:
            t (float): total time
            phase (int): phase of the GHZ state (0, 1, 2, or 3)

        Returns:
            final_state (cupy.ndarray): final state of the system as a density matrix
        """
        initial_state = DensityMatrix.from_label("0" * self.n_qubits)
        initial_state = cp.asarray(initial_state.data)  # send to GPU

        # send Hamiltonian as dense matrix to GPU
        H_dense = self.H.to_matrix(sparse=False)
        U_evo = csl.expm(-1j * cp.asarray(H_dense) * t)  # send to GPU

        ghz_circuit = self.get_ghz_circuit(phase=0)
        ghz_op = Operator.from_circuit(ghz_circuit)
        U_ghz = cp.asarray(ghz_op.data)  # send to GPU

        ghz_circuit_with_phase = self.get_ghz_circuit(phase=phase)
        ghz_op_with_phase = Operator.from_circuit(ghz_circuit_with_phase)
        U_ghz_with_phase = cp.asarray(ghz_op_with_phase.data)  # send to GPU

        U_all = U_ghz_with_phase.conj().T @ U_evo @ U_ghz

        # calculate the final state
        final_state = U_all @ initial_state @ U_all.conj().T

        # send to CPU, and convert to DensityMatrix
        final_state = DensityMatrix(cp.asnumpy(final_state))

        # if the trace is not 1, normalize to 1
        trace_value = np.trace(final_state.data).real
        if not np.isclose(trace_value, 1):
            final_state = final_state / trace_value

        return final_state


def sqrtm_svd(rho, eps=1e-8):
    # create a Hermitian matrix
    rho = 0.5 * (rho + rho.conj().T)

    # Singular Value Decomposition (SVD)
    U, s, Vh = cp.linalg.svd(rho)
    # convert NaN to 0, and values less than eps to 0
    s = cp.nan_to_num(s, nan=0.0)
    s = cp.where(s < eps, 0, s)
    return (U * cp.sqrt(s)) @ U.conj().T


def fidelity_gpu(rho1, rho2, eps=1e-8):
    """
    calculate F(rho1, rho2) = Tr[ sqrt( sqrt(rho1) rho2 sqrt(rho1) ) ] for rho1, rho2 of CuPy arrays

    rho1, rho2: DensityMatrix objects of Qiskit
    """

    # convert rho1, rho2 to CuPy arrays
    rho1 = cp.asarray(rho1.data)
    rho2 = cp.asarray(rho2.data)
    sqrt_rho1 = sqrtm_svd(rho1, eps=eps)
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    product_sqrt = sqrtm_svd(product, eps=eps)
    trace_val = cp.trace(product_sqrt)
    return cp.real(trace_val) ** 2
