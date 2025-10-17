from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, tan, cos, sin, mtimes, fmax, fmin
import numpy as np

from dataclasses import dataclass
from utils import simulate_car_control, TrajectoryTimeProfile, plot_results

def quadform(x, matrix):
    return mtimes([x.T, matrix, x])

@dataclass
class MpcParams:
    """Class for keeping track of an item in inventory."""
    mpc_horizont: int
    n_delay: int
    ts: float
    du_max: float
    u_max: float
    r_dist: float
    r_ang: float
    r_u: float
    r_u_diff: float
    a_comf: float
    jerk_max: float

    def __init__(self, mpc_horizont: int,
                n_delay: int,
                ts: float,
                du_max: float,
                u_max: float,
                r_dist: float,
                r_ang: float,
                r_u: float,
                r_u_diff: float,
                a_comf: float,
                jerk_max: float):
        self.mpc_horizont = mpc_horizont
        self.n_delay = n_delay
        self.ts = ts
        self.du_max = du_max
        self.u_max = u_max
        self.r_dist = r_dist
        self.r_ang = r_ang
        self.r_u = r_u
        self.r_u_diff = r_u_diff
        self.a_comf = a_comf
        self.jerk_max = jerk_max

def is_discrete(model):
    try:
        if(model.disc_dyn_expr == []):
            return False
    except RuntimeError:
        return True
    return True

def create_car_solver(model: AcadosModel, params: MpcParams, increment_mode):
    # Create OCP object
    ocp = AcadosOcp()
    ocp.parameter_values = np.array([0, 0])
    ocp.model = model

    # Dimensions
    Tf = params.mpc_horizont * params.ts  # prediction horizon [s]
    N = params.mpc_horizont    # number of shooting nodes

    ocp.dims.N = N

    # Use EXTERNAL cost to avoid y_expr issues
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # Define cost function directly using states and controls
    x, u = model.x, model.u
    # tau, psi, rwa = model.x
    v, c = model.p.elements()

    # Weighting matrices
    Q = np.diag([params.r_dist, params.r_ang])  # state weights
    R = np.diag([params.r_u])
    R_diff = np.diag([params.r_u_diff])
    cost_expr = quadform(model.x[:2], Q) #+ quadform(du, R)

    cost_expr_e = 10 * quadform(model.x[:2], Q)

    ocp.constraints.idxbu = np.array([0])  # Constrain Δu

    if (increment_mode):
        ocp.constraints.lbu = np.array([-params.du_max])
        ocp.constraints.ubu = np.array([params.du_max])

        rwa = x[2]
        jerk = v * v * u
        ocp.model.con_h_expr = vertcat(rwa, jerk)
        ocp.constraints.lh = np.array([-params.u_max, -params.jerk_max])  # Lower bounds
        ocp.constraints.uh = np.array([params.u_max, params.jerk_max])   # Upper bounds

        cost_expr += quadform(rwa - c, R)
        cost_expr += quadform(u, R_diff)

    else:
        ocp.constraints.lbu = np.array([-params.u_max])
        ocp.constraints.ubu = np.array([params.u_max])

        if (params.n_delay > 0):
            u_actual = x[-1]
        else:
            u_actual = u

        cost_expr += quadform(u_actual - c, R)
        cost_expr += quadform(u - c, R / 10)

    ocp.constraints.x0 = np.zeros(len(model.x.elements()))

    discrete: bool = is_discrete(model)
    #ocp.solver_options.hessian_approx = 'EXACT'

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    if (discrete):
        ocp.solver_options.integrator_type = 'DISCRETE'
    else:
        ocp.solver_options.integrator_type = 'ERK'

    ocp.model.cost_expr_ext_cost = cost_expr
    ocp.model.cost_expr_ext_cost_e = cost_expr_e
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_warm_start = True

    # ocp.solver_options.code_export_directory = str(generated_folder)
    # ocp.code_export_directory = str(generated_folder)
    # ocp.solver_options.json_file = str(generated_folder / 'acados_ocp_nlp2.json')
    solver = AcadosOcpSolver(ocp)
    return solver

def create_discrete_bicycle_model_rk4(params: MpcParams, model_mane: str):
    tau = SX.sym('tau')
    psi = SX.sym('psi')
    rwa = SX.sym('rwa')
    delayed_buf_u = SX.sym('delayed_u', params.n_delay)
    du =  SX.sym('du')
    v = SX.sym('v')
    c = SX.sym('c')

    p = vertcat(v, c)
    Ts = params.ts
    def continuous_dynamics(x, rwa, p):
        tau, psi = x[0], x[1]
        v, c = p[0], p[1]
        dtau = np.sin(psi)
        dpsi = v * rwa - v * c * np.cos(psi) / (1 - c * tau * v)
        return vertcat(dtau, dpsi)
    
    delayed_u = delayed_buf_u[-1] if delayed_buf_u.shape[0] > 0 else rwa

    x = vertcat(tau, psi)
    k1 = continuous_dynamics(x, delayed_u, p)
    k2 = continuous_dynamics(x + 0.5 * Ts * k1, delayed_u, p)
    k3 = continuous_dynamics(x + 0.5 * Ts * k2, delayed_u, p)
    k4 = continuous_dynamics(x + Ts * k3, delayed_u, p)
    x_next = x + (Ts / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    x = vertcat(x, rwa)
    rwa_newt = rwa + du * Ts
    x_next = vertcat(x_next, rwa_newt)

    delay_dynamics = []
    for i in range(params.n_delay):
        if i == 0:
            delay_dynamics.append(rwa)
        else:
            delay_dynamics.append(delayed_buf_u[i-1])

    x_next = vertcat(x_next, *delay_dynamics)
    x = vertcat(x, delayed_buf_u)

    model = AcadosModel()
    model.disc_dyn_expr = x_next
    model.x = x
    model.u = du
    model.p = p
    model.name = model_mane
    return model


def car_model_kinematic_diff(params: MpcParams, model_name: str):
    # Define model variables
    tau = SX.sym('tau')          # d/v
    psi = SX.sym('psi')      # heading angle
    rwa = SX.sym('rwa')
    x = vertcat(tau, psi, rwa)

    du = SX.sym('du')
    # Fixed parameters
    v = SX.sym('v')
    c = SX.sym('с')                # curvature [1/m]

    # Dynamics
    f_expl = vertcat(
        np.sin(psi),
        v * rwa  - v * c * np.cos(psi) / (1 - c * tau * v),
        du
    )

    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x
    model.p = vertcat(v, c)
    model.u = du
    model.name = model_name
    return model