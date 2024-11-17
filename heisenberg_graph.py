import numpy as np
import scipy
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def get_n_steps(t):
    # interval per division
    # max_delta_t = 0.1

    # n_steps = int(t / max_delta_t) + 1
    # print(f" > Total time: {t}, n_steps: {n_steps}")
    n_steps = 1

    return n_steps


def get_graph(n_qubits, rng, ghz_qubits, graph_type="tree"):
    # グラフを作成
    G = nx.Graph()

    # ノードを追加
    nodes = list(range(n_qubits))
    G.add_nodes_from(nodes)

    # ノードに 'GHZ' 属性を追加
    for node in nodes:
        if node in ghz_qubits:
            G.nodes[node]["GHZ"] = True
        else:
            G.nodes[node]["GHZ"] = False

    # エッジを追加
    if graph_type == "tree":
        if n_qubits == 10:
            edges = [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (2, 8),
                (2, 9),
                (4, 5),
                (5, 6),
                (5, 7),
            ]
        elif n_qubits == 8:
            edges = [(0, 1), (0, 3), (0, 4), (1, 2), (2, 7), (3, 5), (5, 6)]
        elif n_qubits == 4:
            edges = [(0, 1), (0, 3), (1, 2)]
    elif graph_type == "line":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]

    G.add_edges_from(edges)

    # エッジにランダムな重み (J_{ij}) を追加
    for edge in G.edges:
        G.edges[edge]["J"] = rng.uniform(-1, 1)

    # エッジに 'operation_order' 属性を追加
    if graph_type == "tree":
        if n_qubits == 10:
            operation_orders = [
                [(0, 1)],
                [(1, 2), (0, 4)],
                [(0, 3), (2, 9), (4, 5)],
                [(5, 6), (2, 8)],
                [(5, 7)],
            ]
        elif n_qubits == 8:
            operation_orders = [
                [(0, 1)],
                [(1, 2), (0, 3)],
                [(0, 4), (2, 7), (3, 5)],
                [(5, 6)],
            ]
        elif n_qubits == 4:
            operation_orders = [[(0, 1)], [(1, 2), (0, 3)]]
    elif graph_type == "line":
        operation_orders = [[(i, i + 1)] for i in range(n_qubits - 1)]

    for i in range(len(operation_orders)):
        for j in range(len(operation_orders[i])):
            G.edges[operation_orders[i][j]]["operation_order"] = i

    return G


def get_prob0(result, n_qubits, mit=None):
    meas_counts = result.data.meas.get_counts()
    num_shots = result.data.meas.num_shots
    prob0_mit = 0

    if "0" * n_qubits not in meas_counts:
        print(" > No counts for |0...0> state")
        prob0_nmit = 0
    else:
        prob0_nmit = meas_counts["0" * n_qubits] / num_shots

    if mit is not None:
        quasis = mit.apply_correction(meas_counts, range(n_qubits))
        prob0_mit = quasis["0" * n_qubits]
        return prob0_nmit, prob0_mit
    else:
        return prob0_nmit


class HeisenbergModel:
    def __init__(self, n_qubits, graph, backend=None):
        self.n_qubits = n_qubits
        self.G = graph
        self.Js = [self.G.edges[edge]["J"] for edge in self.G.edges]
        self.ghz_qubits = [node for node in self.G.nodes if self.G.nodes[node]["GHZ"]]

        self.H = self.get_hamiltonian()
        self.backend = backend

    def get_pauli_strings(self):
        paulis = ["X", "Y", "Z"]
        strings = []

        for i, j in self.G.edges:
            for pauli in paulis:
                pauli_string = ["I"] * self.n_qubits
                pauli_string[i] = pauli
                pauli_string[j] = pauli
                strings.append("".join(pauli_string))

        return strings

    def get_hamiltonian(self):
        pauli_strings = self.get_pauli_strings()
        coefficients = np.repeat(self.Js, 3)

        hamiltonian = SparsePauliOp.from_list(
            [(pauli_string, J) for pauli_string, J in zip(pauli_strings, coefficients)]
        )

        return hamiltonian.simplify().to_matrix()

    def add_heisenberg_interaction(self, qc, t):
        # 'operation_order' に従って、エッジの順番を決める
        max_order = max(
            [self.G.edges[edge]["operation_order"] for edge in self.G.edges]
        )
        for i in range(max_order + 1):
            for edge in self.G.edges:
                if self.G.edges[edge]["operation_order"] == i:
                    j, k = edge
                    J = self.G.edges[edge]["J"]

                    # Function to add Heisenberg interactions between specific pairs of qubits
                    # this corresponds to exp(-i J t (X_j X_k + Y_j Y_k + Z_j Z_k))
                    theta = 2 * J * t
                    qc.cx(j, k)
                    qc.rx(theta, j)
                    qc.rz(theta, k)

                    # Note: order of qubits
                    qc.rzx(-theta, k, j)

                    qc.cx(j, k)

    def get_ghz_circuit(self, phase=0):
        """Prepare the GHZ state on the first half of the qubits.
        phase = 0 corresponds to |0...0> + |1...1>
        phase = 1 corresponds to |0...0> - |1...1>
        phase = 2 corresponds to |0...0> + i|1...1>
        phase = 3 corresponds to |0...0> - i|1...1>

        Args:
            phase (int): phase of the GHZ state (0, 1, 2, or 3)
        """
        max_order = (
            max([self.G.edges[edge]["operation_order"] for edge in self.G.edges]) // 2
        )

        qc = QuantumCircuit(len(self.ghz_qubits))
        qc.h(0)
        if phase == 1:
            qc.z(0)
        elif phase == 2:
            qc.s(0)
        elif phase == 3:
            qc.s(0)
            qc.z(0)
        # 'operation_order' に従って、エッジの順番を決める
        for i in range(max_order + 1):
            for edge in self.G.edges:
                if self.G.edges[edge]["operation_order"] == i:
                    j, k = edge
                    if j in self.ghz_qubits and k in self.ghz_qubits:
                        qc.cx(j, k)

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

    def get_trotter_simulation_pub(
        self, total_time, n_steps, phase=0, initial_layout=None
    ):
        t = total_time / n_steps

        qc = QuantumCircuit(self.n_qubits)

        # Create GHZ state
        qc.compose(self.get_ghz_circuit(phase=0), inplace=True)
        # qc.barrier()

        self.add_heisenberg_interaction(qc, t)
        # qc.barrier()

        # Uncompute GHZ state
        qc.compose(self.get_ghz_circuit(phase=phase).inverse(), inplace=True)

        # Measure
        qc.measure_all()

        # Convert to an ISA circuit and layout-mapped observables.
        pm = generate_preset_pass_manager(
            backend=self.backend, optimization_level=1, initial_layout=initial_layout
        )
        isa_qc = pm.run(qc)

        return qc, isa_qc
