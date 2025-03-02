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

# Constraint function: Ensure the system reaches the desired final state
def constraint(u, z0, z1, dt, N):
    z = np.zeros((2, N+1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the control inputs
    for i in range(N):
        z[:, i+1] = z[:, i] + dynamics(z[:, i], u[i]) * dt

    # Final state constraint: position = 2, velocity = 0
    return np.array([z[0, -1] - z1[0], z[1, -1] - z1[1]])

# One-shot optimization
def solve_block_move_oneshot():
    N = 50  # Number of time steps
    dt = 1.0 / N  # Time step size
    times = np.linspace(0, 1, N+1)  # Time vector

    # Boundary conditions: Move from position 0 to position 2, with velocity 0 at both ends
    z0 = np.array([0, 0])  # Initial state: [0, 0] (position 0, velocity 0)
    z1 = np.array([5, 0])  # Final state: [2, 0] (position 2, velocity 0)

    # Initial guess for control inputs (force)
    u_guess = np.ones(N) * 0.1  # Small initial force

    # Define the constraint dictionary
    cons = {
        'type': 'eq',
        'fun': constraint,
        'args': (z0, z1, dt, N)
    }

    # Run the optimization to minimize control effort, subject to the final state constraint
    result = minimize(objective, u_guess, constraints=cons, method='SLSQP')

    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed:", result.message)

    return result.x, times, z0, z1

# Run the optimization
optimized_controls, times, z0, z1 = solve_block_move_oneshot()

# Simulate the system using the optimized control inputs
def simulate_system(u_optimized, z0, dt, N):
    z = np.zeros((2, N+1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the optimized control inputs
    for i in range(N):
        z[:, i+1] = z[:, i] + dynamics(z[:, i], u_optimized[i]) * dt

    return z

# Simulate the system with the optimized control
states = simulate_system(optimized_controls, z0, 1.0 / len(optimized_controls), len(optimized_controls))
time = np.linspace(0, 1, len(optimized_controls)+1)
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