import networkx as nx
import csv
import gzip
from solver import QuantumSolver
from networkx.algorithms import community

class QuantumOrchestrator:
    """Оркестратор для масштабирования квантового решения на графы любой сложности."""
    
    def __init__(self, large_graph):
        self.large_graph = large_graph

    def solve_large_scale(self, output_file="epinions_solution.csv"):
        # 1. Разбиваем граф на сообщества для локальной оптимизации
        communities = list(community.greedy_modularity_communities(self.large_graph))
        partition_map = {}
        
        print(f"--- Оркестратор: Граф разбит на {len(communities)} кластеров ---")
        
        for i, community_nodes in enumerate(communities):
            subgraph = self.large_graph.subgraph(community_nodes)
            
            # Переиндексация для квантового решателя (QAOA работает с индексами 0..N)
            mapping = {node: idx for idx, node in enumerate(subgraph.nodes())}
            reindexed_subgraph = nx.relabel_nodes(subgraph, mapping)
            
            if reindexed_subgraph.number_of_nodes() < 2:
                continue
                
            print(f"Оптимизация кластера {i+1} (узлов: {reindexed_subgraph.number_of_nodes()})...")
            solver = QuantumSolver(reindexed_subgraph)
            
            # Запуск квантового алгоритма
            _, history = solver.solve(layers=2)
            
            # Запоминаем результат для этого кластера
            for node in community_nodes:
                partition_map[node] = i
                
            print(f"Кластер {i+1} оптимизирован. Энергия: {history[-1]:.4f}")
        
        self.export_to_csv(partition_map, output_file)
        return partition_map

    def export_to_csv(self, partition_map, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Node", "ClusterID"])
            for node, cluster_id in partition_map.items():
                writer.writerow([node, cluster_id])
        print(f"\nРезультаты успешно сохранены в {filename}")

def load_real_graph(file_path, max_nodes=100):
    """Распаковка и загрузка фрагмента графа из GZIP-файла."""
    G = nx.Graph()
    try:
        # 'rt' означает read text mode, gzip сам распакует файл
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'): continue # Пропускаем комментарии
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)
                        if G.number_of_nodes() >= max_nodes:
                            break
                    except ValueError:
                        continue 
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None
    return G

if __name__ == "__main__":
    # Убедись, что файл лежит в папке с этим скриптом
    FILE_PATH = 'soc-Epinions1.txt' 
    
    print("Загрузка реальных данных из Epinions...")
    real_graph = load_real_graph(FILE_PATH, max_nodes=20)
    
    if real_graph:
        orchestrator = QuantumOrchestrator(real_graph)
        orchestrator.solve_large_scale(output_file="epinions_solution.csv")