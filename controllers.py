
import numpy as np

from scipy import interpolate

from abc import ABC
from abc import ABCMeta, abstractmethod
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def reset_solver_initial_guess(solver_):
    """Reset solver to zero initial guess"""
    N = solver_.acados_ocp.dims.N
    nx = solver_.acados_ocp.model.x.shape[0]
    nu = solver_.acados_ocp.model.u.shape[0]
    
    for i in range(N + 1):
        solver_.set(i, "x", np.zeros(nx))
    for i in range(N):
        solver_.set(i, "u", np.zeros(nu))


class Controller(ABC):
    def __init__(self):
        pass
    
    def mpc_control(self):
        return False
    
    @abstractmethod
    def action(self, state):
        pass

    def get_data(self):
        return 0

class PID_Controller(Controller):
    def __init__(self, K_fb = np.array([0.5**2, 0.5]), wheelbase = 2.5, yaw_rate_conpensation = False):
        self.K_fb = K_fb
        self.wheelbase = wheelbase
        self.yaw_rate_conpensation = yaw_rate_conpensation
    def action(self, x0):
        d, psi = x0[:2]
        #print("action")
        #print(self.K_fb, x0)
        uff = self.curv  * np.cos(psi)/(1 - self.curv * d)  *self.wheelbase 
        yaw_rate_fb = 0
        if(self.yaw_rate_conpensation and len(self.K_fb) > 2):
            w = x0[3]
            yaw_rate_fb = self.K_fb[2] * (w - self.curv * self.v * np.cos(psi)/(1 - self.curv  * d))
            yaw_rate_fb = self.K_fb[2] * (w - self.curv * self.v)
        u_opt =  -self.K_fb[:2]@x0[:2] - yaw_rate_fb  + uff
        
        return np.array([u_opt])
    
    def set_params(self, params):
        v, c = params[:2]
        self.curv = c
        self.v = v

    def get_fb_gain(self):
        return self.K_fb

class MPC_Controller(Controller):
    def __init__(self, solver: AcadosOcpSolver, diff_model):
        self.solver = solver
        self.diff_model = diff_model

    def reinit(self):
        self.solver.reset()
        reset_solver_initial_guess(self.solver)
        
    def mpc_control(self) -> bool:
        return True
    
    def is_diff_model(self) -> bool:
        return self.diff_model
    
    def action(self, state):
        self.solver.set(0, "lbx", state)
        self.solver.set(0, "ubx", state)
        status = self.solver.solve()
        if(status != 0):
            print("status bad", status)
        #u_opt = np.array([self.solver.get(1, 'x')[2]])
        #return u_opt
    
    def set_params(self, stage, param):
        self.solver.set(stage, 'p', param)
    
    def get_model(self) -> AcadosModel:
        return self.solver.acados_ocp.model