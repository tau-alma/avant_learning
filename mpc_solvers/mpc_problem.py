import numpy as np
import torch
import casadi as cs
import l4casadi as l4c
from dataclasses import dataclass
from typing import Dict, Callable
from abc import ABC, abstractmethod


class SymbolicMPCSolver(ABC):
    @abstractmethod
    def solve():
        pass


@dataclass
class SymbolicMPCProblem:
    N: int
    h: float

    ocp_x: cs.MX
    ocp_u: cs.MX
    ocp_p: cs.MX
    terminal_ocp_p: cs.MX
    
    dynamics_fun: cs.Function  
    lbx_vec: np.ndarray
    ubx_vec: np.ndarray
    ocp_x_slacks: Dict[int, float] 
    lbu_vec: np.ndarray
    ubu_vec: np.ndarray
    cost_fun: cs.Function  

    # Nonlinear constraints:
    lbg_vec: np.ndarray | None = None
    ubg_vec: np.ndarray | None = None
    g_fun: cs.Function | None = None
    
    # Terminal cost
    terminal_cost_fun: cs.Function | None = None
    # For learned neural terminal cost:
    ocp_x_to_terminal_state_fun: cs.Function | None = None
    terminal_cost_params: cs.MX | None = None
    get_terminal_cost_params_fun: Callable | None = None
    
    # Terminal nonlinear constraints:
    terminal_lbg_vec: np.ndarray | None = None
    terminal_ubg_vec: np.ndarray | None = None
    terminal_g_fun: cs.Function | None = None

    _l4c_model = None
    has_neural_cost = False
    model_external_shared_lib_dir: str | None = None
    model_external_shared_lib_name: str | None = None

    def add_terminal_neural_cost(self, model: torch.nn.Module, model_state: cs.MX, linearize=False):
        assert self.terminal_cost_fun is None, "Trying to assign a terminal neural cost to a problem with an existing terminal cost"
        
        self.has_neural_cost = True

        # Casadi complains about vertcat(MX, MX) inputs to a MXfunction, ugly fix is to use a temprorary MX input here:
        tmp_neural_cost_inputs = cs.MX.sym("tmp", model_state.size())
        # In the solver we can use a function to go from ocp_x to model_state:
        self.ocp_x_to_terminal_state_fun = cs.Function("ocp_x_to_terminal_state_fun", [self.ocp_x, self.terminal_ocp_p], [model_state])

        if linearize:
            self._l4c_model = l4c.realtime.RealTimeL4CasADi(model, approximation_order=2)
        else:
            self._l4c_model = l4c.l4casadi.L4CasADi(model, device="cpu")

        sym_cost_output = self._l4c_model(tmp_neural_cost_inputs)

        if linearize:
            self.terminal_cost_params = self._l4c_model.get_sym_params()
            self.terminal_cost_fun = cs.Function("l_t", [tmp_neural_cost_inputs, self.terminal_cost_params, self.terminal_ocp_p], [-sym_cost_output])
            self.get_terminal_cost_params_fun = self._l4c_model.get_params
        else:
            self.terminal_cost_fun = cs.Function("l_t", [tmp_neural_cost_inputs, self.terminal_ocp_p], [-sym_cost_output])
            self.model_external_shared_lib_dir = self._l4c_model.shared_lib_dir
            self.model_external_shared_lib_name = self._l4c_model.name