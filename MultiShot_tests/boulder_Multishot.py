import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the system dynamics for position (x) and velocity (v)
def dynamics(z, u):
    x, v = z
    dxdt = v
    dvdt = u
    return np.array([dxdt, dvdt])

# Objective function: Minimize control effort (force squared)
def objective(u):
    return np.sum(u**2)  # Minimize the sum of squared control inputs

# Constraint function: Ensure continuity between segments and enforce final state
def constraint(u, z0, z1, dt, N, num_shots):
    z = np.zeros((2, N * num_shots + 1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the control inputs
    for shot in range(num_shots):
        for i in range(N):
            idx = shot * N + i
            z[:, idx+1] = z[:, idx] + dynamics(z[:, idx], u[idx]) * dt

    # Continuity constraints between shots
    continuity_constraints = []
    for shot in range(1, num_shots):
        idx = shot * N
        # Position continuity
        continuity_constraints.append(z[0, idx] - z[0, idx-1])
        # Velocity continuity
        continuity_constraints.append(z[1, idx] - z[1, idx-1])
        # Control input continuity (optional, but recommended)
        continuity_constraints.append(u[idx] - u[idx-1])

    # Final state constraint: position = 2, velocity = 0
    final_state_constraints = [z[0, -1] - z1[0], z[1, -1] - z1[1]]

    return np.array(continuity_constraints + final_state_constraints)

# Multi-shot optimization
def solve_block_move_multishot():
    N = 10  # Number of time steps per shot
    num_shots = 5  # Number of shots
    dt = 1.0 / (N * num_shots)  # Time step size
    times = np.linspace(0, 1, N * num_shots + 1)  # Time vector

    # Boundary conditions: Move from position 0 to position 2, with velocity 0 at both ends
    z0 = np.array([0, 0])  # Initial state: [0, 0] (position 0, velocity 0)
    z1 = np.array([2, 0])  # Final state: [2, 0] (position 2, velocity 0)

    # Initial guess for control inputs (force)
    u_guess = np.ones(N * num_shots) * 0.1  # Small initial force

    # Define the constraint dictionary
    cons = {
        'type': 'eq',
        'fun': constraint,
        'args': (z0, z1, dt, N, num_shots)
    }

    # Run the optimization to minimize control effort, subject to constraints
    result = minimize(objective, u_guess, constraints=cons, method='SLSQP')

    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed:", result.message)

    return result.x, times, z0, z1

# Run the optimization
optimized_controls, times, z0, z1 = solve_block_move_multishot()

# Simulate the system using the optimized control inputs
def simulate_system(u_optimized, z0, dt, N, num_shots):
    z = np.zeros((2, N * num_shots + 1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the optimized control inputs
    for shot in range(num_shots):
        for i in range(N):
            idx = shot * N + i
            z[:, idx+1] = z[:, idx] + dynamics(z[:, idx], u_optimized[idx]) * dt

    return z

# Simulate the system with the optimized control
num_shots = 5  # Number of shots
N = 10  # Number of time steps per shot
states = simulate_system(optimized_controls, z0, 1.0 / (N * num_shots), N, num_shots)
time = np.linspace(0, 1, N * num_shots + 1)
position = states[0, :]
velocity = states[1, :]

# Plot the results
plt.figure(figsize=(10, 6))

# Plot position over time
plt.subplot(3, 1, 1)
plt.plot(time, position, label="Position")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Optimal Trajectory: Position vs. Time")
plt.grid(True)

# Plot velocity over time
plt.subplot(3, 1, 2)
plt.plot(time, velocity, label="Velocity", color="orange")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Optimal Trajectory: Velocity vs. Time")
plt.grid(True)

# Plot control input over time
plt.subplot(3, 1, 3)
plt.plot(time[:-1], optimized_controls, label="Control (Force)", color="green", marker='o')
plt.xlabel("Time")
plt.ylabel("Control (Force)")
plt.title("Optimized Control Input vs. Time")
plt.grid(True)

plt.tight_layout()
plt.show()