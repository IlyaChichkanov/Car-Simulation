
from scipy.integrate import solve_ivp
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, jacobian, Function
from casadi import SX, vertcat, sin, cos, fmax, fmin
import scipy
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy import linalg
from abc import ABC, abstractmethod
from jaxadi import convert
from jax import numpy as jnp
from jax.experimental.ode import odeint

class System:
    def __init__(self):
        pass

    @abstractmethod
    def get_system(self):
        pass

    @abstractmethod
    def __call__(self):
        return self.get_system()
    
    def observation(self):
        observed = vertcat(*[*self.state])
        return observed
    
    def get_input_signals(self, t):
        return []


class KinematicBycicleDiffPertubation(System):
    def __init__(self):
        self.state = [SX.sym('d'), SX.sym('psi'),  SX.sym('rwa')]
        self.input_signals = [SX.sym('du')]

        self.params = [SX.sym('v'), SX.sym('c'), SX.sym('vy')]
  
    def get_system(self,):
        v, c , vy = self.params
        du  = self.input_signals[0] #2.65
        d, psi, rwa = self.state

        wheelbase = 2.5
        w = v * np.tan(rwa)/ wheelbase

        d_dot = v * np.sin(psi) + vy * np.cos(psi)
        psi_dot = w - v * c * np.cos(psi) / (1 - c * d)
        rwa_dot = du
        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params), vertcat(d_dot, psi_dot, rwa_dot) 





class KinematicBycicle(System):
    def __init__(self):
        self.state = [SX.sym('d'), SX.sym('psi')]
        self.input_signals = [SX.sym('u')]
        self.params = [SX.sym('v'), SX.sym('c')]
        self.wheelbase = 2.5

    def get_system(self):
        v, c  = self.params
        u  = self.input_signals[0] #2.65
        d, psi = self.state

        rwa = u
        w = v * np.tan(rwa)/ self.wheelbase

        d_dot = v * np.sin(psi) 
        psi_dot = w - v * c * np.cos(psi) / (1 - c * d) 
        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params), vertcat(d_dot, psi_dot) 

class KinematicBycicleDiff(System):
    def __init__(self):
        self.state = [SX.sym('tau'), SX.sym('psi'),  SX.sym('rwa')]
        self.input_signals = [SX.sym('du')]

        self.params = [SX.sym('v'), SX.sym('c')]
        self.wheelbase = 2.5
    def get_system(self,):
        v, c  = self.params
        du  = self.input_signals[0] #2.65
        tau, psi, rwa = self.state
        
        w = v * np.tan(rwa)/ self.wheelbase

        tau_dot = np.sin(psi) 
        psi_dot = v * np.tan(rwa) / self.wheelbase - v * c * np.cos(psi) / (1 - c * tau * v)
        rwa_dot = du
        return vertcat(*self.state), \
                vertcat(*self.input_signals), \
                vertcat(*self.params), \
                vertcat(tau_dot, psi_dot, rwa_dot) 
    

class KinematicBycicleTau(System):
    def __init__(self):
        self.state = [SX.sym('d'), SX.sym('tau')]
        self.input_signals = [SX.sym('u')]
        self.params = [SX.sym('v'), SX.sym('c')]
        self.wheelbase = 2.5

    def get_system(self):
        v, c  = self.params
        u  = self.input_signals[0] #2.65
        tau, psi = self.state

        rwa = u
        w = v * np.tan(rwa)/ self.wheelbase

        tau_dot = np.sin(psi) 
        psi_dot = w - v * c #* np.cos(psi) / (1 - c * d) 
        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params), vertcat(tau_dot, psi_dot) 



    
class DynamicBycicle(System):
    def __init__(self, m = 1580, J = 3650, a = 1.44, wheelbase = 2.65, Cf = 50*1e3, Cr = 50*1e3):
        self.state = [SX.sym('d'), SX.sym('psi'), SX.sym('wz'), SX.sym('vy')]
        self.input_signals = [SX.sym('u')]
        self.params = [SX.sym('vx'), SX.sym('c')]
        self.a = a
        self.b = wheelbase - a
        self.rot_m = np.linalg.inv(np.array([[1, self.b], [0, 1]]))
        self.rot_m = np.array([[1, self.b], [0, 1]])
        self.Cf = Cf
        self.Cr = Cr
        self.wheelbase = wheelbase
        self.m = m
        self.J = J

    def get_system(self):
        d, psi, wz, vy = self.state
        vx, c = self.params
        delta = self.input_signals[0]

        alfa_f = np.arctan2(vy + self.a * wz, vx) - delta
        alfa_r =  np.arctan2(vy - self.b * wz, vx)
        Ff = -alfa_f * self.Cf 
        Fr = -alfa_r * self.Cr 

        wz_dot = (self.a * Ff - self.b * Fr)/self.J
        vy_dot = (Ff + Fr)/self.m - vx * wz #- wz_dot
        

        d_dot = vx * np.sin(psi)
        psi_dot = wz -  vx * c * np.cos(psi)/(1 - c * d)

        #vy_dot, wz_dot = self.rot_m@np.array([vy_dot, wz_dot])
        state_dot = vertcat(d_dot, psi_dot, wz_dot, vy_dot)
        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params), state_dot

class DynamicBycicleDiff(System):
    def __init__(self, m = 1580, J = 3650, a = 1.44, wheelbase = 2.65, Cf = 50*1e3, Cr = 50*1e3):
        self.state = [SX.sym('d'), SX.sym('psi'), SX.sym('rwa'), SX.sym('wz'), SX.sym('vy')]
        self.input_signals = [SX.sym('u')]
        self.params = [SX.sym('vx'), SX.sym('c')]
        self.a = a
        self.b = wheelbase - a
        self.rot_m = np.linalg.inv(np.array([[1, self.b], [0, 1]]))
        self.rot_m = np.array([[1, self.b], [0, 1]])
        self.Cf = Cf
        self.Cr = Cr
        self.wheelbase = wheelbase
        self.m = m
        self.J = J

    def get_system(self):
        d, psi, rwa, wz, vy = self.state
        vx, c = self.params
        u = self.input_signals[0]

        alfa_f = np.arctan2(vy + self.a * wz, vx) - rwa
        alfa_r =  np.arctan2(vy - self.b * wz, vx)
        Ff = -alfa_f * self.Cf 
        Fr = -alfa_r * self.Cr 

        vy_dot = (Ff + Fr)/self.m - vx * wz
        wz_dot = (self.a * Ff - self.b * Fr)/self.J

        d_dot = vx * np.sin(psi)
        psi_dot = wz - vx * c * np.cos(psi)/(1 - c * d)
        rwa_dot = u
        #vy_dot, wz_dot = self.rot_m@np.array([vy_dot, wz_dot])
        state_dot = vertcat(d_dot, psi_dot, rwa_dot, wz_dot, vy_dot)
        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params), state_dot
    
class KinematicBycicleActuator(System):
    def __init__(self):
        self.state = [SX.sym('d'), SX.sym('psi'), SX.sym('delta'), SX.sym('delta_dot')]
        self.input_signals = [SX.sym('u')]

        self.params = [SX.sym('v'), SX.sym('c')]
        self.wheelbase = 2.5
    def get_system(self):
        v, c  = self.params
        u  = self.input_signals[0] #2.65
        d, psi, delta, delta_dot = self.state
        kp = 420.9
        kv = 40.61
        d_dot = v * np.sin(psi) 
        psi_dot = v * (np.tan(delta)) / (self.wheelbase) - v * c * np.cos(psi) / (1 - c * d) 
        f = vertcat(d_dot, psi_dot, delta_dot, kp * (u - delta) - kv * delta_dot) 
        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params),  f



class KinematicBycicleDiffDelay(System):
    def __init__(self,  n_delay = 2, delay_time: float = 0.2):
        self.n_delay = n_delay
        self.delay_time = delay_time
        self.delay_states = SX.sym('delay_states', n_delay)
        self.state = [SX.sym('d'), SX.sym('psi'),  SX.sym('rwa'), self.delay_states]
        self.input_signals = [SX.sym('du')]

        self.params = [SX.sym('v'), SX.sym('c')]
        self.wheelbase = 2.5
    def get_system(self,):
        v, c  = self.params
        du  = self.input_signals[0] #2.65
        d, psi, rwa, delay_states = self.state

        tau_delay = self.delay_time / self.n_delay
        delay_dynamics = []
        for i in range(self.n_delay):
            if i == 0:
                dz_dt = (rwa - delay_states[i]) / tau_delay
            else:
                dz_dt = (delay_states[i-1] - delay_states[i]) / tau_delay
            delay_dynamics.append(dz_dt)
        
        actual_rwa = delay_states[-1]


        w = v * np.tan(actual_rwa)/ self.wheelbase

        d_dot = v * np.sin(psi) 
        psi_dot = w - v * c * np.cos(psi) / (1 - c * d)
        rwa_dot = du
        return vertcat(*self.state), \
                vertcat(*self.input_signals), \
                vertcat(*self.params), \
                vertcat(d_dot, psi_dot, rwa_dot, *delay_dynamics) 
    
class KinematicBycicleDelay(System):
    def __init__(self, n_delay = 2, delay_time: float = 0.2):
        self.n_delay = n_delay
        self.delay_time = delay_time
        self.delay_states = SX.sym('delay_states', n_delay)
        self.state = [SX.sym('d'), SX.sym('psi'), self.delay_states]
        self.input_signals = [SX.sym('u')]
        self.params = [SX.sym('v'), SX.sym('c')]
        self.wheelbase = 2.5

    def get_system(self):
        v, c  = self.params
        u  = self.input_signals[0] #2.65
        d, psi, delay_states = self.state


        if delay_states.shape[0] > 0:
            tau_delay = self.delay_time / self.n_delay
        # Correct delay dynamics with time constants
        delay_dynamics = []
        for i in range(self.n_delay):
            if i == 0:
                dz_dt = (u - delay_states[i]) / tau_delay
            else:
                dz_dt = (delay_states[i-1] - delay_states[i]) / tau_delay
            delay_dynamics.append(dz_dt)
        
        actual_rwa = delay_states[-1] if delay_states.shape[0] > 0 else u


        w = v * np.tan(actual_rwa)/ self.wheelbase

        d_dot = v * np.sin(psi) 
        psi_dot = w - v * c * np.cos(psi) / (1 - c * d) 

        f_expl = vertcat(d_dot, psi_dot, *delay_dynamics)

        return vertcat(*self.state), vertcat(*self.input_signals), vertcat(*self.params), f_expl
    
    
class Integrator:
    def __init__(self, model: AcadosModel):
        self.df_dt = Function('func', [*model.x.elements(), *model.u.elements(), *model.p.elements()], [model.f_expl_expr])
        self.df_dt_jax = convert(self.df_dt)
        
        self.model = model
        JacA = jacobian(model.f_expl_expr, model.x)
        self.jacA = Function('J_p', vertcat(model.x, model.u, model.p).elements(), [JacA])
        JacB = jacobian(model.f_expl_expr, model.u)
        self.jacB = Function('J_p', vertcat(model.x, model.u, model.p).elements(), [JacB])
        self.x_len = len(model.x.elements())
        self.control_len = len(model.u.elements())
        self.param_len = len(model.p.elements())
        self.use_distortion = False

    def df_dx(self, state, u, params):
        return np.array(self.df_dt(*[*state, *u, *params])).T[0]
    
    def df_dx_jax(self, state, t, *params):
        return jnp.array(self.df_dt_jax(*state, *params)).flatten()
    
    def step(self, c0, u, params, dt):
        system = lambda t, y: self.df_dx(y, u, params)   
        solution1 = solve_ivp(
            system,
            (0, dt),
            c0,
            method='RK45' 
        )
        return solution1.y.T[-1]
    
    def step_jax(self, c0, u, params, dt):
        solution = odeint(
            self.df_dx_jax,
            c0,
            np.array([0, dt]),
            *u, *params
        )
        return np.array(solution[-1])
    
    def integrate(self, c0, u, params, t_span):
        system = lambda t, y: self.df_dx(y, u, params)   
        solution1 = solve_ivp(
            system,
            t_span,
            c0,
            method='RK45' 
        )
        return solution1.y.T
    
    def get_lin_system_dynamics(self, state, control_inputs, params):
        assert len(state) == self.x_len
        assert len(control_inputs) == self.control_len
        assert len(params) == self.param_len
        A = np.array(self.jacA(*state, *control_inputs, *params))#[0]
        B = np.array(self.jacB(*state, *control_inputs, *params))#[0]
        return A, B
    
def get_close_loop_matrix(A, B, K_fb):
    state_length = A.shape[0]
    K = np.array([0.0]*state_length)
    K[:state_length] = np.copy(K_fb[:state_length])
    Ar = A - B*K
    return Ar

def get_lapynov_matrix(A, B, R, K_fb):
    Ar = get_close_loop_matrix(A, B, K_fb)
    state_length = Ar.shape[0]
    P = linalg.solve_continuous_lyapunov(Ar.T, np.diag(R[:state_length]))
    return P
        
def make_acados_model(system: System, model_name: str) -> AcadosModel:
    x, input_signals, params, f_expl = system.get_system()
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x
    model.p = params
    model.u = input_signals
    model.name = model_name
    return model


