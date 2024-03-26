import torch
import numpy as np
from dynamics import Dynamics, DualAvantDynamics, AvantDynamics
from costs import Cost, EmpiricalCost, InfoCost
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import CMAES, CEM
from evotorch.logging import StdOutLogger
from training_data import TrajectoryDataset, BootstrapSampler
from torch.utils.data import DataLoader
from utils import draw_2d_solution


class InfoGainProblem(Problem):
    def __init__(self, N: int, dynamics: Dynamics, cost: Cost, dataset: TrajectoryDataset):
        self.N = N
        self.dynamics = dynamics
        self.cost = cost
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=1, sampler=BootstrapSampler(self.dataset))

        super().__init__(
            objective_sense="min",
            solution_length=N * len(dynamics.lbu),
            initial_bounds=(dynamics.lbu.tile(N), dynamics.ubu.tile(N)),
            dtype=torch.float32,
            device="cuda:0",
        )

        self.x0 = None
        self.p = None

    def reset(self):
        self.x0 = self.dynamics.generate_initial_state()
        self.p = self.cost.generate_p(self.x0)

    def visualize(self, solution) -> torch.Tensor:
        u_values = solution.unsqueeze(0)
        x_values = self.dynamics.propagate(self.x0, u_values)
        answer = draw_2d_solution(x_values.cpu(), u_values.cpu(), self.p.cpu())
        self.dataset.add_entry(x_values.cpu()[0], answer, self.p.cpu()[0])

    def train(self):
        self.cost.training_loop(self.dataloader)

    def _evaluate_batch(self, solutions: SolutionBatch):
        with torch.no_grad():
            u_values = solutions.values.view(-1, self.N, len(self.dynamics.lbu))
            x_values = self.dynamics.propagate(self.x0, u_values)
            C = self.cost(x_values, self.p)
            penalty = 1e2 * self.dynamics.compute_constraint_violation(x_values, u_values)
            solutions.set_evals(C + penalty)


class EvoTorchWrapper:
    def __init__(self, problem: InfoGainProblem):
        self.problem = problem

    def solve(self, popsize=int(1e6), std=1, iters=1):
        while True:
            self.problem.reset()

            # searcher = CMAES(self.problem, popsize=popsize, stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
            searcher = CEM(self.problem, parenthood_ratio=0.001, popsize=popsize, stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
            logger = StdOutLogger(searcher, interval=10)

            searcher.run(iters)
            best_discovered_solution = searcher.status["center"].clone().view(self.problem.N, len(self.problem.dynamics.lbu))
            self.problem.visualize(best_discovered_solution)
            self.problem.train()

    
if __name__ == "__main__":
    dynamics = DualAvantDynamics(dt=1/10)
    cost = InfoCost(N=40, K=10).cuda()
    dataset = TrajectoryDataset("trajectories.pt")
    problem = InfoGainProblem(N=40, cost=cost, dynamics=dynamics, dataset=dataset)
    solver = EvoTorchWrapper(problem)
    solver.solve()
