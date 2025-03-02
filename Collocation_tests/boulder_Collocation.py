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

# Constraint function: Ensure dynamics are satisfied at collocation points
def constraint(u, z0, z1, dt, N, num_shots):
    z = np.zeros((2, N * num_shots + 1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the control inputs
    for shot in range(num_shots):
        for i in range(N):
            idx = shot * N + i
            z[:, idx+1] = z[:, idx] + dynamics(z[:, idx], u[idx]) * dt

    # Collocation constraints: Ensure dynamics are satisfied at segment boundaries
    collocation_constraints = []
    for shot in range(1, num_shots):
        idx = shot * N
        # Dynamics at the boundary
        dz = dynamics(z[:, idx], u[idx])
        # Collocation constraint: Match the dynamics
        collocation_constraints.extend(z[:, idx] - z[:, idx-1] - dz * dt)

    # Final state constraint: position = target_position, velocity = 0
    final_state_constraints = [z[0, -1] - z1[0], z[1, -1] - z1[1]]

    return np.array(collocation_constraints + final_state_constraints)

# Multi-shot optimization with collocation
def solve_block_move_multishot(z0, z1, N, num_shots, T):
    dt = T / (N * num_shots)  # Time step size
    times = np.linspace(0, T, N * num_shots + 1)  # Time vector

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

    return result.x, times

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

# Main function
def main():
    # Define parameters
    N = 20  # Number of time steps per shot
    num_shots = 3  # Number of shots
    T = 20.0  # Total time duration (changed from 1.0 to 2.0)
    target_position = 20.0  # Target position
    z0 = np.array([0, 0])  # Initial state: [0, 0] (position 0, velocity 0)
    z1 = np.array([target_position, 0])  # Final state: [target_position, 0]

    # Run the optimization
    optimized_controls, times = solve_block_move_multishot(z0, z1, N, num_shots, T)

    # Simulate the system with the optimized control
    states = simulate_system(optimized_controls, z0, T / (N * num_shots), N, num_shots)
    time = np.linspace(0, T, N * num_shots + 1)
    position = states[0, :]
    velocity = states[1, :]

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot position over time
    plt.subplot(3, 1, 1)
    plt.plot(time, position, label="Position")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(f"Optimal Trajectory: Position vs. Time (Target = {target_position})")
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

# Run the main function
if __name__ == "__main__":
    main()