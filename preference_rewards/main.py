import argparse
import torch
from preference_learning.dynamics import DualDynamics
from preference_learning.optimization import EvoTorchWrapper
from preference_learning.training_data import TrajectoryDataset

def check_valid_device(device_string):
    try:
        torch.device(device_string)
        return True
    except RuntimeError as e:
        raise ValueError(f"Invalid device {device_string}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, help="Specifies which problem instance is solved.", required=True)
    parser.add_argument('--N', type=int, help="The length of the optimization horizon in time steps.", required=True)
    parser.add_argument('--dt', type=float, help="The length of each time step in seconds.", required=True)
    parser.add_argument('--evaluate', action="store_true", help="Whether to run in evaluation mode, i.e. to optimize over the actual cost rather than the information gain metric. Can be used to visualize the optimal behaviour under the current cost.")
    parser.add_argument('--device', type=str, help="Selects the device: cpu or cuda:N (N=0,1,2,...)", default="cuda:0")
    args = parser.parse_args()

    check_valid_device(args.device)

    if args.problem == "cartpole":        
        from cartpole_preferences.dynamics import CartPoleDynamics
        from cartpole_preferences.costs import CartPoleCost
        from cartpole_preferences.problem import CartPoleInfoGainProblem
        dynamics = CartPoleDynamics(dt=args.dt, device=args.device) 
        cost = CartPoleCost
        dataset = TrajectoryDataset(directory="cartpole_data")
        problem = CartPoleInfoGainProblem(N=args.N, cost_module=cost, dynamics=dynamics, dataset=dataset, n_ensembles=10, parameters_path="cartpole_parameters", device=args.device)

    elif args.problem == "avant":
        from avant_preferences.dynamics import AvantDynamics
        from avant_preferences.costs import AvantEmpiricalCost
        from avant_preferences.problem import AvantInfoGainProblem
        dynamics = AvantDynamics(dt=args.dt, device=args.device) 
        cost = AvantEmpiricalCost
        dataset = TrajectoryDataset(directory="avant_data")
        problem = AvantInfoGainProblem(N=args.N, cost_module=cost, dynamics=dynamics, dataset=dataset, n_ensembles=10, parameters_path="avant_parameters", device=args.device)
    else:
        raise ValueError(f"problem={args.problem} not defined")

    solver = EvoTorchWrapper(problem)
    solver.solve()
