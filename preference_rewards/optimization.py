import torch
import numpy as np
from dynamics import Dynamics, AvantDynamics
from costs import Cost, EmpiricalCost
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import CMAES, CEM
from evotorch.logging import StdOutLogger
from utils import draw_2d_solution



class InfoGainProblem(Problem):
    def __init__(self, N: int, dynamics: Dynamics, cost: Cost):
        self.N = N
        self.dynamics = dynamics
        self.cost = cost

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

    def visualize(self, solution):
        u_values = solution.unsqueeze(0)
        x_values = self.dynamics.propagate(self.x0, u_values)
        draw_2d_solution(x_values.cpu(), u_values.cpu(), self.p.cpu())

    def _evaluate_batch(self, solutions: SolutionBatch):
        with torch.no_grad():
            u_values = solutions.values.view(-1, self.N, len(self.dynamics.lbu))
            x_values = self.dynamics.propagate(self.x0, u_values)
            C = self.cost(x_values, u_values, self.p)
            penalty = 1e2 * self.dynamics.compute_constraint_violation(x_values, u_values)
            solutions.set_evals(C + penalty)


class EvoTorchWrapper:
    def __init__(self, problem: Problem):
        self.problem = problem

    def solve(self, popsize=50000, std=1, iters=100):
        self.problem.reset()

        # searcher = CMAES(self.problem, popsize=popsize, stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
        searcher = CEM(self.problem, parenthood_ratio=0.05, popsize=popsize, stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
        logger = StdOutLogger(searcher, interval=10)

        searcher.run(iters)
        best_discovered_solution = searcher.status["center"].clone().view(self.problem.N, len(self.problem.dynamics.lbu))
        self.problem.visualize(best_discovered_solution)

    

if __name__ == "__main__":
    dynamics = AvantDynamics(dt=1/10)
    cost = EmpiricalCost(N=30)
    cost.init_test_weights()
    cost = cost.cuda()
    problem = InfoGainProblem(N=30, cost=cost, dynamics=dynamics)
    solver = EvoTorchWrapper(problem)
    solver.solve()
