import pennylane as qml
from pennylane import qaoa as pl_qaoa
import torch

class QuantumSolver:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes())  # Сохраняем реальные ID узлов
        self.n_qubits = graph.number_of_nodes()
        self.cost_h, self.mixer_h = pl_qaoa.maxcut(graph)
        
        # ВАЖНО: Создаем устройство на текущее количество узлов
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def solve(self, layers=3, warm_start_params=None):
        @qml.qnode(self.dev, interface="torch")
        def circuit(params):
            # Используем wires=range(self.n_qubits) для соответствия индексу устройства
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for i in range(layers):
                pl_qaoa.cost_layer(params[i], self.cost_h)
                pl_qaoa.mixer_layer(params[i + layers], self.mixer_h)
            return qml.expval(self.cost_h)

        # Подбираем размер параметров: 2 * layers
        num_params = 2 * layers
        params = warm_start_params if warm_start_params is not None else torch.rand(num_params, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=0.1)
        
        loss_history = []
        for _ in range(50):
            optimizer.zero_grad()
            loss = circuit(params)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
        return params.detach(), loss_history