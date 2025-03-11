import numpy as np
import matplotlib.pyplot as plt
import pyoptsparse

# Define the system dynamics for position (x) and velocity (v)
def dynamics(z, u):
    x, v = z
    dxdt = v
    dvdt = u
    return np.array([dxdt, dvdt])

# Objective function: Minimize control effort (force squared)
def objective_func(dv):
    funcs = {}
    funcs['obj'] = np.sum(dv**2)  # Minimize sum of squared control inputs
    return funcs, False

# Constraint function: Ensure the system reaches the desired final state
def constraint_func(dv, z0, z1, dt, N):
    z = np.zeros((2, N+1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the control inputs
    for i in range(N):
        z[:, i+1] = z[:, i] + dynamics(z[:, i], dv[i]) * dt

    # Final state constraint: position = 2, velocity = 0
    funcs = {}
    funcs['con'] = np.array([z[0, -1] - z1[0], z[1, -1] - z1[1]])
    return funcs, False

# Solve using SNOPT
def solve_block_move_snopt():
    N = 50  # Number of time steps
    dt = 1.0 / N  # Time step size
    
    z0 = np.array([0, 0])  # Initial state
    z1 = np.array([1, 0])  # Final state
    
    u_guess = np.ones(N) * 0.1  # Initial guess
    
    opt_prob = pyoptsparse.Optimization("Block Move", objective_func)
    opt_prob.addVarGroup("dv", N, lower=-10, upper=10, value=u_guess)
    opt_prob.addConGroup("con", 2, lower=0.0, upper=0.0, wrt="dv")
    
    solver = pyoptsparse.SNOPT()
    solver.setOption('Major Iteration Limit', 100)
    solver.setOption('Minor Iteration Limit', 500)
    solver.setOption('Print File', 'snopt_output.txt')
    solver.setOption('Summary File', 'snopt_summary.txt')
    
    sol = solver(opt_prob, sens='FD')
    return sol

# Run the optimization
solution = solve_block_move_snopt()
print("Optimization completed.")
print("Optimal control inputs:", solution.xStar['dv'])