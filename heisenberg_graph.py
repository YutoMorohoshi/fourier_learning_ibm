import numpy as np
import scipy
import networkx as nx
import itertools
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def get_n_steps(t):
    # interval per division
    # max_delta_t = 0.5

    # n_steps = int(t / max_delta_t) + 1
    n_steps = 1

    return n_steps


def get_graph(n_qubits, rng, graph_type="tree"):
    # グラフを作成
    G = nx.Graph()

    # ノードを追加
    nodes = list(range(n_qubits))
    G.add_nodes_from(nodes)

    # ノードに 'hadamard' 属性を追加
    if graph_type == "tree":
        print("ToDo: hadamard")
    elif graph_type == "line":
        for node in nodes:
            if node == n_qubits // 2 - 1:
                G.nodes[node]["hadamard"] = True
            else:
                G.nodes[node]["hadamard"] = False

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
        # G.edges[edge]["J"] = 1

    # エッジに 'cnot' 属性を追加
    if graph_type == "tree":
        print("ToDo: cnot_order")
    elif graph_type == "line":
        # State |0011...1100> (center qubits are 1 and the rest are 0)
        leftmost = n_qubits // 4
        rightmost = leftmost + n_qubits // 2 - 1
        hadamard = list(filter(lambda x: G.nodes[x]["hadamard"], G.nodes))[0]
        left_start = hadamard
        right_start = hadamard

        for i in range(hadamard - leftmost + 1):
            if i == 0:
                # Hadamard より右側の処理
                control = hadamard
                target = hadamard + 1
                right_start = target
                G.edges[(control, target)]["cnot"] = {
                    "order": i,
                    "control": control,
                    "target": target,
                }
            else:
                # Hadamard より左側の処理
                control = left_start
                target = left_start - 1

                # target が leftmost を超えていない場合のみ CNOT を作用させる
                if target >= leftmost:
                    G.edges[(control, target)]["cnot"] = {
                        "order": i,
                        "control": control,
                        "target": target,
                    }
                    left_start = target

                # Hadamard より右側の処理
                control = right_start
                target = right_start + 1

                # target が rightmost を超えていない場合のみ CNOT を作用させる
                if target <= rightmost:
                    G.edges[(control, target)]["cnot"] = {
                        "order": i,
                        "control": control,
                        "target": target,
                    }
                    right_start = target

        """
        # State |111000> 
        rightmost = n_qubits // 2 - 1
        for i in range(math.ceil(n_qubits / 4)):
            if i == 0:
                if n_qubits == 4:
                    control = 1
                    target = 0
                else:
                    control = math.ceil(n_qubits / 4)
                    target = math.ceil(n_qubits / 4) - 1
                G.edges[(control, target)]["cnot"] = {
                    "order": i,
                    "control": control,
                    "target": target,
                }
            else:
                left_control = math.ceil(n_qubits / 4) - i
                left_target = math.ceil(n_qubits / 4) - i - 1
                G.edges[(left_control, left_target)]["cnot"] = {
                    "order": i,
                    "control": left_control,
                    "target": left_target,
                }
                # CNOT を作用させる右ノードが、全体の左半分に収まっていれば、左半分のノードに対しても CNOT を作用させる
                if math.ceil(n_qubits / 4) + i <= rightmost:
                    right_control = math.ceil(n_qubits / 4) + i - 1
                    right_target = math.ceil(n_qubits / 4) + i
                    G.edges[(right_control, right_target)]["cnot"] = {
                        "order": i,
                        "control": right_control,
                        "target": right_target,
                    }
        """

        # それ以外のエッジには None の 'cnot_order' 属性を追加
        for edge in G.edges:
            if "cnot" not in G.edges[edge]:
                G.edges[edge]["cnot"] = {"order": None, "control": None, "target": None}

    return G


def get_positions(n_qubits, graph_type):
    # 等間隔に配置するためのカスタム座標を定義
    if graph_type == "tree":
        if n_qubits == 10:
            positions = {
                0: (3, 1),
                1: (4, 1),
                2: (5, 1),
                3: (3, 2),
                4: (2, 1),
                5: (1, 1),
                6: (0, 1),
                7: (1, 0),
                8: (5, 0),
                9: (6, 1),
            }
        elif n_qubits == 8:
            positions = {
                0: (3, 0),
                1: (4, 0),
                2: (5, 0),
                3: (2, 0),
                4: (3, 1),
                5: (1, 0),
                6: (0, 0),
                7: (6, 0),
            }
        elif n_qubits == 4:
            positions = {
                0: (0, 0),
                1: (1, 0),
                2: (2, 0),
                3: (0, 1),
            }
    elif graph_type == "line":
        positions = {i: (i, 0) for i in range(n_qubits)}

    return positions


def get_initial_layout(n_qubits, graph_type, qpu_name):
    if graph_type == "tree":
        if qpu_name == "ibm_brisbane":
            if n_qubits == 10:
                initial_layout = [60, 61, 62, 53, 59, 58, 57, 71, 72, 63]
            elif n_qubits == 8:
                initial_layout = [60, 61, 62, 59, 53, 58, 57, 63]
            elif n_qubits == 4:
                initial_layout = [60, 61, 62, 53]
        elif qpu_name == "ibm_torino":
            if n_qubits == 10:
                initial_layout = [103, 104, 105, 93, 102, 101, 100, 111, 112, 106]
        elif qpu_name == "ibm_marrakesh":
            if n_qubits == 10:
                initial_layout = [103, 104, 105, 96, 102, 101, 100, 116, 117, 106]
            elif n_qubits == 8:
                initial_layout = [103, 104, 105, 102, 96, 101, 100, 106]
    elif graph_type == "line":
        initial_layout = list(range(n_qubits))

    return initial_layout


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


def extract_probs(probs_dict, successful_samples):
    probs_extracted = []

    for sample_id, probs in probs_dict.items():
        if sample_id in successful_samples:
            probs_extracted.append(probs.values())

    return list(itertools.chain.from_iterable(probs_extracted))


class HeisenbergModel:
    def __init__(self, n_qubits, graph, backend=None):
        self.n_qubits = n_qubits
        self.G = graph
        self.Js = [self.G.edges[edge]["J"] for edge in self.G.edges]
        self.backend = backend

        # sparse big-endian matrix
        self.H = self.get_hamiltonian()

        # first_half_pairs の始まりは、n_qubits を 4 で割った余りが 0 なら 1, そうでなければ 0
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
                # これは必要
                pauli_string.reverse()

                strings.append("".join(pauli_string))

        return strings

    def get_hamiltonian(self):
        pauli_strings = self.get_pauli_strings()
        coefficients = np.repeat(self.Js, 3)  # Js を各パウリ演算子に対応させる

        H = SparsePauliOp.from_list(
            [(pauli_string, J) for pauli_string, J in zip(pauli_strings, coefficients)]
        )

        return H

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
        max_order = max(
            [
                self.G.edges[edge]["cnot"]["order"]
                for edge in self.G.edges
                if self.G.edges[edge]["cnot"]["order"] is not None
            ]
        )

        # qc = QuantumCircuit(self.n_qubits // 2)
        qc = QuantumCircuit(self.n_qubits)

        # hadamard 属性を持つノードを抽出
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
        # 'cnot' 'order' に従って、cnot をかける順番を決める
        for i in range(max_order + 1):
            for j, k in self.G.edges:
                if self.G.edges[(j, k)]["cnot"]["order"] == i:
                    # print(f"i: {i}, j:{j}, k: {k}")
                    control = self.G.edges[(j, k)]["cnot"]["control"]
                    target = self.G.edges[(j, k)]["cnot"]["target"]
                    qc.cx(control, target)

        return qc

    def exact_simulation(self, t, phase=0):
        # Compute the exact evolution of the Heisenberg model using the GHZ state.
        initial_state = Statevector.from_label("0" * self.n_qubits)

        U = scipy.sparse.linalg.expm(-1j * self.H.to_matrix(sparse=True) * t)

        # 有望
        # H = Operator(self.H).reverse_qargs().to_matrix()
        # H = scipy.sparse.csr_matrix(H)
        # U = scipy.sparse.linalg.expm(-1j * H * t)

        # ダメ
        # H = SparsePauliOp.from_operator(Operator(self.H).reverse_qargs())
        # U = PauliEvolutionGate(H, t)

        ghz_circuit = self.get_ghz_circuit(phase=0)
        ghz_op = Operator(ghz_circuit)

        ghz_circuit_with_phase = self.get_ghz_circuit(phase=phase)
        ghz_op_with_phase = Operator(ghz_circuit_with_phase)

        # This is correct
        final_state = initial_state.evolve(
            ghz_op_with_phase.adjoint().data @ U @ ghz_op.data
        )

        prob0 = final_state.probabilities_dict()["0" * self.n_qubits]

        return prob0

    def get_trotter_simulation_circuit(
        self, total_time, n_steps, phase=0, initial_layout=None
    ):
        t = total_time / n_steps

        qc = QuantumCircuit(self.n_qubits)

        # Create GHZ state
        qc.compose(self.get_ghz_circuit(phase=0), inplace=True)
        # qc.barrier()

        # Apply time-evolution
        self.add_heisenberg_interaction(qc, self.first_half_pairs, self.first_Js, t / 2)
        self.add_heisenberg_interaction(qc, self.second_half_pairs, self.second_Js, t)
        for _ in range(n_steps - 1):
            self.add_heisenberg_interaction(qc, self.first_half_pairs, self.first_Js, t)
            self.add_heisenberg_interaction(
                qc, self.second_half_pairs, self.second_Js, t
            )
        self.add_heisenberg_interaction(qc, self.first_half_pairs, self.first_Js, t / 2)
        # qc.barrier()

        # Uncompute GHZ state
        qc.compose(self.get_ghz_circuit(phase=phase).inverse(), inplace=True)

        # Measure
        qc.measure_all()

        # Convert to an ISA circuit and layout-mapped observables.
        if initial_layout is None:
            optimization_level = 1
        else:
            optimization_level = 3
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
        )
        exec_qc = pm.run(qc)

        return qc, exec_qc
