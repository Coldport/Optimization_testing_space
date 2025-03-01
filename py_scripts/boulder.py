import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Define the system dynamics
def dynamics(t, z, u):
    x, v = z
    dxdt = v
    dvdt = u(t)
    return [dxdt, dvdt]

# Boundary conditions
def boundary_conditions(z0, z1):
    x0, v0 = z0
    x1, v1 = z1
    return [x0 - 0, v0 - 0, x1 - 1, v1 - 0]  # x(0) = 0, v(0) = 0, x(1) = 1, v(1) = 0

# Objective function: Minimize control effort
def objective(u, times):
    total_cost = 0
    for i in range(len(times)-1):
        dt = times[i+1] - times[i]
        total_cost += 0.5 * dt * (u[i]**2 + u[i+1]**2)  # Trapezoidal integration for u^2
    return total_cost

# Trapezoidal collocation
def solve_block_move():
    N = 20  # Number of collocation points
    times = np.linspace(0, 1, N+1)
    z0 = np.zeros(2)  # Initial state
    z1 = np.array([1, 0])  # Final state
    
    # Optimization: Solve for control inputs (u)
    u_guess = np.zeros(N)  # Initial guess for control inputs
    res = minimize(objective, u_guess, args=(times), constraints={'type': 'eq', 'fun': boundary_conditions})
    
    # Return the optimized control inputs
    return res.x

# Run the optimization
optimized_controls = solve_block_move()
print("Optimized control inputs:", optimized_controls)
