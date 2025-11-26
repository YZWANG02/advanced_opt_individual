from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

random.seed(42)
np.random.seed(42)


class IndependentCascadeModel:
    def __init__(self, graph: nx.Graph, p: float = 0.1, steps: int = 5, mc_runs: int = 30):
        self.graph = graph
        self.p = p
        self.steps = steps
        self.mc_runs = mc_runs

    def simulate_once(self, seed_nodes: List[int]) -> int:
        activated = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(self.steps):
            new_active = set()
            for node in frontier:
                for nbr in self.graph.neighbors(node):
                    if nbr in activated:
                        continue
                    if random.random() < self.p:
                        new_active.add(nbr)
            if not new_active:
                break
            activated |= new_active
            frontier = new_active
        return len(activated)
    
    def simulate_with_timesteps(self, seed_nodes: List[int]) -> List[int]:
        
        activated = set(seed_nodes)
        frontier = set(seed_nodes)
        timestep_counts = [len(activated)]  
        
        for _ in range(self.steps):
            new_active = set()
            for node in frontier:
                for nbr in self.graph.neighbors(node):
                    if nbr in activated:
                        continue
                    if random.random() < self.p:
                        new_active.add(nbr)
            if not new_active:
                break
            activated |= new_active
            frontier = new_active
            timestep_counts.append(len(activated))
        
        return timestep_counts

    def expected_spread(self, seed_nodes: List[int]) -> float:
        if not seed_nodes:
            return 0.0
        total = 0
        for _ in range(self.mc_runs):
            total += self.simulate_once(seed_nodes)
        return total / self.mc_runs




def node_cost(graph: nx.Graph, node: int, base: float = 1.0) -> float:
    return base + graph.degree[node]


def build_networks() -> Dict[str, nx.Graph]:
    graphs = {
        "BA_200": nx.barabasi_albert_graph(200, 3, seed=7),
        "BA_600": nx.barabasi_albert_graph(600, 4, seed=11),
        "ER_200": nx.erdos_renyi_graph(200, 0.05, seed=23),
        "ER_600": nx.erdos_renyi_graph(600, 0.03, seed=29),
    }
    for name, g in graphs.items():
        if not nx.is_connected(g):
            largest = max(nx.connected_components(g), key=len)
            graphs[name] = g.subgraph(largest).copy()
    return graphs


def budget_for_graph(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    avg_deg = sum(dict(graph.degree()).values()) / n
    base_budget = n * 0.35
    return base_budget + avg_deg * 5




class CostAwareGreedy:
    def __init__(
        self,
        graph: nx.Graph,
        ic_model: IndependentCascadeModel,
        budget: float,
        cost_map: Dict[int, float],
    ):
        self.graph = graph
        self.ic_model = ic_model
        self.budget = budget
        self.cost_map = cost_map

    def solve(self) -> Dict[str, object]:
        selected: List[int] = []
        remaining_budget = self.budget
        best_spread = 0.0
        history = []

        while True:
            best_node = None
            best_gain = 0.0
            for node in self.graph.nodes():
                if node in selected:
                    continue
                cost = self.cost_map[node]
                if cost > remaining_budget:
                    continue
                candidate = selected + [node]
                spread = self.ic_model.expected_spread(candidate)
                gain = (spread - best_spread) / cost
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
                    candidate_spread = spread
            if best_node is None:
                    break
            selected.append(best_node)
            remaining_budget -= self.cost_map[best_node]
            best_spread = candidate_spread
            history.append(best_spread)

        return {
            "seeds": selected,
            "spread": best_spread,
            "cost": self.budget - remaining_budget,
            "curve": history,
        }


class PageRankGreedy:
 
    def __init__(
        self,
        graph: nx.Graph,
        ic_model: IndependentCascadeModel,
        budget: float,
        cost_map: Dict[int, float],
    ):
        self.graph = graph
        self.ic_model = ic_model
        self.budget = budget
        self.cost_map = cost_map
        self.rank = nx.pagerank(graph)

    def solve(self) -> Dict[str, object]:
        ordering = sorted(
            self.graph.nodes(),
            key=lambda n: self.rank[n] / (self.cost_map[n] + 1e-8),
            reverse=True,
        )
        selected: List[int] = []
        remaining = self.budget
        history = []
        last_spread = 0.0

        for node in ordering:
            cost = self.cost_map[node]
            if cost > remaining:
                continue
            selected.append(node)
            remaining -= cost
            last_spread = self.ic_model.expected_spread(selected)
            history.append(last_spread)

        return {
            "seeds": selected,
            "spread": last_spread,
            "cost": self.budget - remaining,
            "curve": history,
        }




def repair_solution(bits: np.ndarray, cost_map: Dict[int, float], budget: float) -> np.ndarray:
    indices = np.where(bits == 1)[0]
    current_cost = sum(cost_map[i] for i in indices)
    if current_cost <= budget:
        return bits
    scores = sorted(indices, key=lambda idx: cost_map[idx], reverse=True)
    for idx in scores:
        bits[idx] = 0
        current_cost -= cost_map[idx]
        if current_cost <= budget:
            break
    if current_cost > budget:
        ones = np.where(bits == 1)[0]
        for idx in ones:
            bits[idx] = 0
            current_cost -= cost_map[idx]
            if current_cost <= budget:
                    break
    if current_cost == 0:
        fallback = min(cost_map, key=cost_map.get)
        bits[fallback] = 1
    return bits


@dataclass
class GAParams:
    pop_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float


class GeneticAlgorithmBase:
    def __init__(
        self,
        graph: nx.Graph,
        ic_model: IndependentCascadeModel,
        cost_map: Dict[int, float],
        budget: float,
        params: GAParams,
    ):
        self.graph = graph
        self.ic_model = ic_model
        self.cost_map = cost_map
        self.budget = budget
        self.params = params
        self.n = graph.number_of_nodes()
        self.population: List[np.ndarray] = []
        self.fitness: List[float] = []
        self.best_history: List[float] = []
        self.avg_history: List[float] = []

    def initialize(self):
        self.population = []
        for _ in range(self.params.pop_size):
            bits = np.random.rand(self.n) < 0.05
            bits = bits.astype(int)
            bits = repair_solution(bits, self.cost_map, self.budget)
            self.population.append(bits)

    def evaluate_population(self):
        self.fitness = []
        for individual in self.population:
            seeds = np.where(individual == 1)[0].tolist()
            spread = self.ic_model.expected_spread(seeds)
            self.fitness.append(spread)

    def select_parents(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.population), size=4, replace=False)
        best_two = sorted(idx, key=lambda i: self.fitness[i], reverse=True)[:2]
        return (self.population[best_two[0]].copy(), self.population[best_two[1]].copy())

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.params.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, self.n - 2)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(self, bits: np.ndarray, rate: float) -> np.ndarray:
     
        adaptive_rate = rate * (0.5 + random.random())
        mask = np.random.rand(self.n) < adaptive_rate
        bits = np.bitwise_xor(bits, mask.astype(int))
        return bits

    def elitism_enabled(self) -> bool:
        return False
    
    def get_elite_count(self) -> int:
        
        return 0

    def record_stats(self):
        if not self.fitness:
            self.best_history.append(0.0)
            self.avg_history.append(0.0)
            return
        self.best_history.append(float(np.max(self.fitness)))
        self.avg_history.append(float(np.mean(self.fitness)))

    def run(self) -> Dict[str, object]:
        self.initialize()
        self.evaluate_population()
        self.record_stats()

        best_idx = int(np.argmax(self.fitness))
        best_bits = self.population[best_idx].copy()
        best_spread = self.fitness[best_idx]

        for gen in range(self.params.generations):
            new_population: List[np.ndarray] = []
            if self.elitism_enabled():
                elite_count = self.get_elite_count()
                
                sorted_indices = sorted(
                    range(len(self.population)),
                    key=lambda i: self.fitness[i],
                    reverse=True
                )
                for i in range(min(elite_count, len(sorted_indices))):
                    new_population.append(self.population[sorted_indices[i]].copy())

            while len(new_population) < self.params.pop_size:
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, self.params.mutation_rate)
                child2 = self.mutate(child2, self.params.mutation_rate)
                child1 = repair_solution(child1, self.cost_map, self.budget)
                child2 = repair_solution(child2, self.cost_map, self.budget)
                new_population.append(child1)
                if len(new_population) < self.params.pop_size:
                    new_population.append(child2)

            self.population = new_population[: self.params.pop_size]
            self.evaluate_population()
            self.record_stats()
            gen_best_idx = int(np.argmax(self.fitness))
            gen_best_spread = self.fitness[gen_best_idx]
            if gen_best_spread > best_spread:
                best_spread = gen_best_spread
                best_bits = self.population[gen_best_idx].copy()
            best_idx = gen_best_idx

        seeds = np.where(best_bits == 1)[0].tolist()
        total_cost = sum(self.cost_map[i] for i in seeds)
        return {
            "seeds": seeds,
            "spread": best_spread,
            "cost": total_cost,
            "best_curve": self.best_history,
            "avg_curve": self.avg_history,
        }


class MultiPointUniformGA(GeneticAlgorithmBase):
    """Uniform crossover without elitism."""

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.params.crossover_rate:
            return parent1.copy(), parent2.copy()
        mask = np.random.rand(self.n) < 0.5
        # ensure at least two positions swap
        if mask.sum() == 0 or mask.sum() == self.n:
            flip_indices = np.random.choice(self.n, size=2, replace=False)
            mask[flip_indices] = ~mask[flip_indices]
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2


class EliteUniformGA(MultiPointUniformGA):
   

    def elitism_enabled(self) -> bool:
        return True
    
    def get_elite_count(self) -> int:
     
        return max(5, int(self.params.pop_size * 0.1))




def run_algorithms_on_graph(name: str, graph: nx.Graph) -> Dict[str, Dict[str, object]]:
    budget = budget_for_graph(graph)
    ic_model = IndependentCascadeModel(graph, p=0.12, steps=5, mc_runs=25)
    cost_map = {node: node_cost(graph, node) for node in graph.nodes()}
    params_low = GAParams(pop_size=30, generations=50, mutation_rate=0.25, crossover_rate=0.6)
    params_high = GAParams(pop_size=50, generations=80, mutation_rate=0.25, crossover_rate=0.6)
    params = params_low if graph.number_of_nodes() <= 200 else params_high

    greedy_cost = CostAwareGreedy(graph, ic_model, budget, cost_map).solve()
    greedy_pr = PageRankGreedy(graph, ic_model, budget, cost_map).solve()

    ga_no_elite = MultiPointUniformGA(graph, ic_model, cost_map, budget, params).run()
    ga_elite = EliteUniformGA(graph, ic_model, cost_map, budget, params).run()

    return {
        "HA-Benefit": greedy_cost,
        "HA-PageRank": greedy_pr,
        "GA-w/o-Elite": ga_no_elite,
        "GA-Elite": ga_elite,
        "budget": budget,
    }



def plot_influence(results: Dict[str, Dict[str, object]]):
    algs = ["HA-Benefit", "HA-PageRank", "GA-Elite", "GA-w/o-Elite"]
    colors = ["#4c72b0", "#1b9e77", "#d95f02", "#7570b3"]
    n_nets = len(results)
    cols = 2
    rows = math.ceil(n_nets / cols)
    plt.figure(figsize=(12, 5 * rows))
    for idx, (net_name, data) in enumerate(results.items()):
        plt.subplot(rows, cols, idx + 1)
        spreads = [data[alg]["spread"] for alg in algs]
        plt.bar(algs, spreads, color=colors)
        plt.title(net_name)
        plt.ylabel("Activated nodes")
        plt.xticks(rotation=20)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ic_influence_comparison.png", dpi=400)
    plt.close()


def plot_ga_curves(results: Dict[str, Dict[str, object]]):
    
    for net_name, data in results.items():
        plt.figure(figsize=(7, 4.5))
        for alg_name in ["GA-w/o-Elite", "GA-Elite"]:
            if alg_name not in data:
                continue
            best_curve = data[alg_name]["best_curve"]
            best_curve = [max(best_curve[:i+1]) for i in range(len(best_curve))]
            if not best_curve:
                continue
            gens = range(len(best_curve))
            plt.plot(gens, best_curve, label=alg_name, linewidth=2)
        plt.xlabel("Generation")
        plt.ylabel("Best Spread")
        plt.title(f"{net_name} - GA Convergence")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{net_name}_ga_curve.png", dpi=400)
        plt.close()


def plot_propagation_timesteps(results: Dict[str, Dict[str, object]], graphs: Dict[str, nx.Graph]):
    
    ic_model = IndependentCascadeModel(graphs[list(graphs.keys())[0]], p=0.12, steps=5, mc_runs=25)
    
    for net_name, data in results.items():
        graph = graphs[net_name]
        ic_model.graph = graph
        plt.figure(figsize=(8, 5))
        
        alg_names = ["HA-Benefit", "HA-PageRank", "GA-Elite", "GA-w/o-Elite"]
        colors = ["#f2cb5d", "#1b9e77", "#d95f02", "#7570b3"]
        
        for alg_name, color in zip(alg_names, colors):
            if alg_name not in data:
                continue
            seeds = data[alg_name]["seeds"]
            if not seeds:
                continue
            
            
            all_timesteps = []
            max_len = 0
            for _ in range(20):  
                timesteps = ic_model.simulate_with_timesteps(seeds)
                all_timesteps.append(timesteps)
                max_len = max(max_len, len(timesteps))
            
            padded = []
            for ts in all_timesteps:
                padded.append(ts + [ts[-1]] * (max_len - len(ts)))
            
            avg_timesteps = np.mean(padded, axis=0)
            timestep_indices = range(len(avg_timesteps))
            
            plt.plot(timestep_indices, avg_timesteps, label=alg_name, 
                    color=color, linewidth=2, marker='o', markersize=4)
        
        plt.xlabel("Time Step")
        plt.ylabel("Activated Nodes")
        plt.title(f"{net_name} - Propagation Over Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{net_name}_propagation_timesteps.png", dpi=400)
        plt.close()
        print(f"Saved propagation timesteps plot: {net_name}_propagation_timesteps.png")


def print_summary(results: Dict[str, Dict[str, object]]):
    print("\nSummary (spread / cost / seeds)")
    for net_name, data in results.items():
        budget = data["budget"]
        print(f"\nNetwork: {net_name} | Budget: {budget:.1f}")
        for alg in ["HA-Benefit", "HA-PageRank", "GA-Elite", "GA-w/o-Elite"]:
            entry = data[alg]
            print(
                f"  {alg:<15} Spread: {entry['spread']:.2f} "
                f"Cost: {entry['cost']:.1f} Seeds: {len(entry['seeds'])}"
            )




def main():
    graphs = build_networks()
    results = {}
    for name, graph in graphs.items():
        start = time.time()
        results[name] = run_algorithms_on_graph(name, graph)
        results[name]["time"] = time.time() - start
    plot_influence(results)
    plot_ga_curves(results)
    plot_propagation_timesteps(results, graphs)
    print_summary(results)
    print("\nPlots saved:")
    print("  - ic_influence_comparison.png")
    print("  - *_ga_curve.png")
    print("  - *_propagation_timesteps.png")


if __name__ == "__main__":
    main()

