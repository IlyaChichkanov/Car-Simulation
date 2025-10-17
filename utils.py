from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, tan, cos, sin, mtimes
import numpy as np

from systems import *
from controllers import *
from dataclasses import dataclass



class TrajectoryTimeProfile:
    def __init__(self, f_vel, f_curv):
        self.vel_profile = f_vel
        self.curv_profile = f_curv

def simulate_car_control(real_model:AcadosModel, 
                         controller: Controller, 
                         trajectory: TrajectoryTimeProfile,
                         dt, t_list, delay_cicles = 5):

    integrator = Integrator(real_model)
    wheelbase = 2.5
    # Simulation parameters
    N_sim = len(t_list)
    u_hist = [np.array([0.0])] * max(delay_cicles, 1)


    # Initial state
    STATE_LENGTH = len(real_model.x.elements())
    x0 = np.zeros(STATE_LENGTH)
    x0[0] = 0.1
    x0[1] = 0.02

    # Storage
    states = np.zeros((N_sim + 1, STATE_LENGTH))
    controls = np.zeros((N_sim, 1))
    controls_actual = np.zeros((N_sim, 1))
    states[0, :STATE_LENGTH] = x0
    x_current = x0
    if(controller.mpc_control()):
        k = controller.get_model().x.shape[0]
        controller.reinit()
    else:
        k = 2
        
    x0_solver = np.zeros(k)
    x0_solver[0] = x_current[0]/trajectory.vel_profile(t_list[0])
    x0_solver[1] = x_current[1]
    for i, t in enumerate(t_list[:-1]):
        # Set initial state
        curr_curv = trajectory.curv_profile(t)
        curr_vel = trajectory.vel_profile(t)
        params = np.hstack((curr_vel, curr_curv))
        d, psi = x_current[:2]
        
        if(controller.mpc_control()):
            for stage in range(controller.solver.acados_ocp.dims.N):
                t_ = t + (stage) * dt
                if(t_ >= t_list[-1]):
                    t_ = t_list[-1] 
                #p = np.hstack((trajectory.vel_profile(t_), trajectory.curv_profile(t_), -x_current[3]/9))

                p = np.hstack((trajectory.vel_profile(t_), trajectory.curv_profile(t_)))
                # if(t > 9.5 and t < 10):
                #     print("aa")
                controller.set_params(stage, p)
           
        
            # a = np.hstack(u_hist[::-1])
            x0_solver = np.hstack((d/curr_vel, psi, controller.solver.get(1, 'x')[2:])) 

            #controller.action(x0_solver)
            #reset_solver_initial_guess(controller.solver)
            controller.solver.set(0, "lbx", x0_solver)
            controller.solver.set(0, "ubx", x0_solver)
            status = controller.solver.solve()
            if(status != 0):
                print("status bad", status, x0_solver)
                controller.solver.reset()
                raise BaseException("This is a custom BaseException message.") 
                
                
            if(controller.is_diff_model()):
                u_opt = controller.solver.get(1, 'x')[2:3]
                #u_opt = controller.solver.get(0, 'x')[2] + controller.solver.get(0, 'u') * dt
            else:
                u_opt = controller.solver.get(0, 'u')

        else:
            controller.set_params(params)
            x0_solver[0] = d/curr_vel
            x0_solver[1] = psi
            u_opt = controller.action(x0_solver[:2])

        
            

        
        #x_current = integrator.integrate(x_current, u_opt, params, (t, t + dt))[-1]
        u_hist.append(u_opt)
        u_hist.pop(0)
        u_delayed = u_hist[0]
        
        x_current = integrator.step_jax(x_current, u_delayed, params, dt)
        print("t", t, x_current[:2])

        if(controller.mpc_control()):
            controls[i] = controller.solver.get(0, 'u')
        else:
            controls[i] = 0
        controls_actual[i] = u_opt
        states[i+1, :] =  x_current
        
    return t_list, states, controls, controls_actual


def plot_results(times, states, controls, controls_actual):
    plt.clf()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    
    ax1.plot(times, states[:-1, 0], 'b-', linewidth=2)
    ax1.set_ylabel('Lateral deviation [m]')
    ax1.grid(True)
    ax1.axhline(y=0, color='r', linestyle='--')
    
    ax2.plot(times, states[:-1, 1], 'g-', linewidth=2)
    ax2.set_ylabel('Heading angle [rad]')
    ax2.grid(True)
    ax2.axhline(y=0, color='r', linestyle='--')
    
    #ax3.plot(times, states[:-1, 2], 'g-', linewidth=2)
    ax3.plot(times, controls_actual, 'k', linewidth=2)
    ax3.set_ylabel('Steering angle[rad]')
    ax3.grid(True)
    ax3.axhline(y=0, color='r', linestyle='--')

    
    # ax4.plot(times[:], states[:-1, 3], 'r-', linewidth=2)
    ax4.plot(times[:], controls, 'r-', linewidth=2)
    ax4.set_ylabel('v_y[m/s]')
    ax4.set_xlabel('Time [s]')
    ax4.grid(True)
    ax4.axhline(y=0, color='k', linestyle='--')
    
    plt.tight_layout()
    plt.show()