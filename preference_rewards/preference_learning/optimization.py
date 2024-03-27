import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from preference_learning.dynamics import Dynamics
from preference_learning.costs import Cost, InfoCost
from preference_learning.training_data import TrajectoryDataset, BootstrapSampler
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import CMAES, CEM
from evotorch.logging import StdOutLogger


class InfoGainProblem(Problem, ABC):
    def __init__(self, N: int, dynamics: Dynamics, cost_module: Cost, dataset: TrajectoryDataset, n_ensembles: int, parameters_path: str, device: str):
        self.N = N
        self.dynamics = dynamics
        self.cost = InfoCost(N=N, K=n_ensembles, cost_module=cost_module, parameters_path=parameters_path, device=device)
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=1, sampler=BootstrapSampler(self.dataset))

        super().__init__(
            objective_sense="min",
            solution_length=N * len(dynamics.lbu),
            initial_bounds=(dynamics.lbu.tile(N), dynamics.ubu.tile(N)),
            dtype=torch.float32,
            device=device,
        )

        self.x0 = None
        self.p = None

    @abstractmethod
    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def visualize(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        self.x0 = self.dynamics.generate_initial_state()
        self.p = self.generate_p(self.x0)

    def get_label(self, solution: torch.Tensor):
        u_values = solution.unsqueeze(0)
        x_values = self.dynamics.propagate(self.x0, u_values)
        answer = self.visualize(x_values.cpu(), u_values.cpu(), self.p.cpu())
        self.dataset.add_entry(x_values.cpu()[0], answer, self.p.cpu()[0])

    def train(self):
        self.cost.training_loop(self.dataloader)

    def _evaluate_batch(self, solutions: SolutionBatch):
        with torch.no_grad():
            u_values = solutions.values.view(-1, self.N, len(self.dynamics.lbu))
            x_values = self.dynamics.propagate(self.x0, u_values)
            C = self.cost.get_cost(x_values, self.p)
            penalty = 1e2 * self.dynamics.compute_constraint_violation(x_values, u_values)
            solutions.set_evals(C + penalty)


class EvoTorchWrapper:
    def __init__(self, problem: InfoGainProblem):
        self.problem = problem

    def solve(self, popsize=int(1e6), parents=1e3, std=1, iters=50):
        while True:
            self.problem.reset()

            # searcher = CMAES(self.problem, popsize=popsize, stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
            searcher = CEM(self.problem, parenthood_ratio=parents/popsize, popsize=popsize, stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
            logger = StdOutLogger(searcher, interval=10)

            searcher.run(iters)
            best_discovered_solution = searcher.status["pop_best"].values.clone().view(self.problem.N, len(self.problem.dynamics.lbu))
            self.problem.get_label(best_discovered_solution)
            self.problem.train()