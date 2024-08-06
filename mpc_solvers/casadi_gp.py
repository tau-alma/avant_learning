import casadi as cs
import numpy as np
import torch
import json

def rbf_ard_kernel(x, x_prime, sigma_f, length_scales):
    if len(x_prime.shape) != 2:
        x_prime = x_prime.unsqueeze(0)
    x_scaled = x / torch.sqrt(length_scales)
    x_prime_scaled = x_prime / torch.sqrt(length_scales)
    xx = (x_scaled ** 2).sum(dim=1).unsqueeze(1)
    yy = (x_prime_scaled ** 2).sum(dim=1).unsqueeze(0)
    xy = x_scaled @ x_prime_scaled.t()
    sqdist = xx + yy - 2 * xy
    return sigma_f * torch.exp(-0.5 * sqdist)

class CasadiGP:
    def __init__(self, n_inducing: int, gp_inputs: torch.Tensor, gp_targets: torch.Tensor, gp_param_json_path: str):
        self.n_inducing = n_inducing
        self.gp_inputs = gp_inputs
        self.gp_targets = gp_targets

        with open(gp_param_json_path, 'r') as infile:
            params = json.load(infile)
        
        self.gp_sigma_f = params["sigma_f"]
        self.gp_lengthscale = torch.tensor([params["lengthscale"]])
        self.gp_sigma_n = params["sigma_n"]

        knn = rbf_ard_kernel(
            self.gp_inputs,
            self.gp_inputs,
            self.gp_sigma_f, self.gp_lengthscale
        )
        self.knn_diag = knn.diagonal(dim1=0, dim2=1)

    def _symbolic_rbf_ard_kernel_sx(self):
        x = cs.SX.sym("x", self.n_inducing, self.gp_inputs.shape[1])
        x_prime = cs.SX.sym("x_prime", 1, self.gp_inputs.shape[1])
        sigma_f = cs.SX.sym("sigma_f", 1)
        lengthscale = cs.SX.sym("lengthscale", 1, self.gp_inputs.shape[1])

        diff = ((x.T - x_prime.T)**2 / lengthscale.T).T
        diff = cs.sum2(diff)
        kernel = sigma_f * cs.exp(-0.5 * diff)

        return cs.Function("kernel", [x, x_prime, sigma_f, lengthscale], [kernel])

    def get_mu_symbolic_expression(self, z):
        '''
        Creates a parametric expression for the sparse GP outputs, which is then used to construct the MPC problem
        '''
        Zind = cs.MX.sym("Zind", self.n_inducing, self.gp_inputs.shape[1])
        outputs = cs.vertcat([])
        sym_params_list = [Zind]

        kernel_creation_fn = self._symbolic_rbf_ard_kernel_sx()
        alpha = cs.MX.sym("alpha", self.n_inducing)
        sigma_f = cs.MX.sym("sigma_f", 1)
        lengthscale = cs.MX.sym("lengthscale", self.gp_inputs.shape[1])

        sym_params_list.extend([alpha, sigma_f, lengthscale])
        K_Zind_z = kernel_creation_fn(Zind, z, sigma_f, lengthscale)
        outputs = cs.vertcat(outputs, K_Zind_z.T @ alpha)

        params = cs.vcat([cs.reshape(mx, np.prod(mx.shape), 1) for mx in sym_params_list])
        return outputs, params
    
    def get_params(self, Zind):
        '''
        Computes the sparse GP params for the casadi expression embedded to the MPC
        '''
        Zind = torch.from_numpy(Zind).to(torch.float64)
        params_list = [Zind]

        Kmm = rbf_ard_kernel(Zind, Zind, self.gp_sigma_f, self.gp_lengthscale)
        Knm = rbf_ard_kernel(
            self.gp_inputs, Zind,
            self.gp_sigma_f, self.gp_lengthscale
        )
        L = torch.linalg.cholesky(Kmm + 0.001 * torch.eye(len(Kmm)))
        terms = torch.linalg.solve_triangular(L, Knm.T, upper=False)
        lamda_diag = self.knn_diag - torch.sum(terms**2, dim=0)
        lamda_term = 1/(lamda_diag + self.gp_sigma_n)
        Qm_intermediate = Knm.T * lamda_term
        Qm = Kmm + Qm_intermediate @ Knm

        L = torch.linalg.cholesky(Qm + 0.001 * torch.eye(len(Kmm)))
        alpha = torch.linalg.solve_triangular(L.T, torch.linalg.solve_triangular(L, Knm.T, upper=False), upper=True)
        alpha = alpha * lamda_term @ self.gp_targets

        params_list.extend([alpha, self.gp_sigma_f, self.gp_lengthscale])
        params = np.hstack([p.numpy().flatten(order="F") for p in params_list])
        return params, Kmm, Qm, alpha