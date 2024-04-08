import torch
import numpy as np
import time
from evotorch import Problem, SolutionBatch
from evotorch.algorithms import CMAES, CEM
from evotorch.logging import StdOutLogger


class MPCProblem(Problem):
    def __init__(self, N: int, dynamics, cost, device: str):
        self.N = N
        self.dynamics = dynamics
        self.cost = cost

        super().__init__(
            objective_sense="min",
            solution_length=N * len(self.dynamics.lbu),
            initial_bounds=(self.dynamics.lbu.tile(N), self.dynamics.ubu.tile(N)),
            dtype=torch.float32,
            device=device,
        )

        self.x0 = None
        self.p = None

    def reset(self, M, x0, p):
        self.x0 = x0
        self.p = p

    def _compute_cost(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        M, N, _ = u_values.shape
        stage_costs = self.cost.stage_cost(x_values[:, :self.N, :].reshape(M*N, -1), u_values[:, :, :].view(M*N, -1), p_values)
        stage_costs = stage_costs.view(M, N).sum(dim=1)
        terminal_costs = self.cost.terminal_cost(x_values[:, self.N, :], p_values).view(M)
        return stage_costs + terminal_costs

    def _compute_constraint_violation(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        # Compute absolute state violations
        state_lower_violations = torch.clamp(self.dynamics.lbx - x_values, min=0).sum(dim=[1, 2])
        state_upper_violations = torch.clamp(x_values - self.dynamics.ubx, min=0).sum(dim=[1, 2])
        # Compute absolute control violations
        control_lower_violations = torch.clamp(self.dynamics.lbu - u_values, min=0).sum(dim=[1, 2])
        control_upper_violations = torch.clamp(u_values - self.dynamics.ubu, min=0).sum(dim=[1, 2])
        # Aggregate violations
        total_violations = state_lower_violations + state_upper_violations + control_lower_violations + control_upper_violations
        return total_violations

    def _evaluate_batch(self, solutions: SolutionBatch):
        u_values = solutions.values.view(-1, self.N, len(self.dynamics.lbu))
        x_values = self.dynamics.propagate(self.x0, u_values)
        C = self._compute_cost(x_values, u_values, self.p)
        penalty = 1e2 * self._compute_constraint_violation(x_values, u_values)
        solutions.set_evals(C + penalty)

    def get_horizon(self, solution):
        u_values = solution.unsqueeze(0)
        x_values = self.dynamics.propagate(self.x0, u_values)
        return x_values.squeeze(0)


class EvoTorchSolver:
    def __init__(self, problem):
        self.problem = problem
        self.u0 = None

    def solve(self, x0: torch.Tensor, p: torch.Tensor, popsize=5e4, parents=1e2, std=0.5, iters=7):
        with torch.no_grad():
            if self.u0 is None:
                u0 = torch.zeros(self.problem.solution_length)
            else:
                u0 = self.u0

            self.problem.reset(M=int(popsize), x0=x0, p=p)

            #searcher = CMAES(self.problem, popsize=int(popsize), stdev_init=std, center_init=torch.zeros(self.problem.solution_length))
            searcher = CEM(self.problem, parenthood_ratio=parents/popsize, popsize=int(popsize), stdev_init=std, center_init=u0)
            # logger = StdOutLogger(searcher, interval=10)
            searcher.run(iters)

            best_discovered_solution = searcher.status["pop_best"].values.clone().view(self.problem.N, len(self.problem.dynamics.lbu))
            horizon = self.problem.get_horizon(best_discovered_solution)

            self.u0 = best_discovered_solution.cpu().numpy()
            self.u0[:-1] = self.u0[1:]
            self.u0[-1, :] = 0
            self.u0 = self.u0.flatten()

        return horizon.cpu().numpy().astype(np.float64)
