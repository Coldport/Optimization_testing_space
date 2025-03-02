import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the system dynamics for position (x) and velocity (v)
def system_dynamics(state, control_input):
    x, v = state
    dxdt = v
    dvdt = control_input
    return np.array([dxdt, dvdt])

# Objective function: Minimize control effort (force squared)
def minimize_control_effort(control_inputs):
    return np.sum(control_inputs**2)  # Minimize the sum of squared control inputs

# Constraint function: Ensure dynamics are satisfied at collocation points
def enforce_constraints(control_inputs, initial_state, target_state, dt, num_steps_per_shot, num_shots):
    state = np.zeros((2, num_steps_per_shot * num_shots + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the control inputs
    for shot in range(num_shots):
        for i in range(num_steps_per_shot):
            idx = shot * num_steps_per_shot + i
            state[:, idx+1] = state[:, idx] + system_dynamics(state[:, idx], control_inputs[idx]) * dt

    # Collocation constraints: Ensure dynamics are satisfied at segment boundaries
    collocation_constraints = []
    for shot in range(1, num_shots):
        idx = shot * num_steps_per_shot
        # Dynamics at the boundary
        dz = system_dynamics(state[:, idx], control_inputs[idx])
        # Collocation constraint: Match the dynamics
        collocation_constraints.extend(state[:, idx] - state[:, idx-1] - dz * dt)

    # Final state constraint: position = target_position, velocity = 0
    final_state_constraints = [state[0, -1] - target_state[0], state[1, -1] - target_state[1]]

    return np.array(collocation_constraints + final_state_constraints)


def optimize_trajectory(initial_state, target_state, num_steps_per_shot, num_shots, total_time):
    dt = total_time / (num_steps_per_shot * num_shots)  # Time step size
    times = np.linspace(0, total_time, num_steps_per_shot * num_shots + 1)  # Time vector

    # Initial guess for control inputs (force)
    initial_guess_controls = np.ones(num_steps_per_shot * num_shots) * 0.1  # Small initial force

    # Define the constraint dictionary
    constraints = {
        'type': 'eq',
        'fun': enforce_constraints,
        'args': (initial_state, target_state, dt, num_steps_per_shot, num_shots)
    }

    # Run the optimization to minimize control effort, subject to constraints
    result = minimize(minimize_control_effort, initial_guess_controls, constraints=constraints, method='SLSQP')

    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed:", result.message)

    return result.x, times

# Simulate the system using the optimized control inputs
def simulate_trajectory(optimized_controls, initial_state, dt, num_steps_per_shot, num_shots):
    state = np.zeros((2, num_steps_per_shot * num_shots + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the optimized control inputs
    for shot in range(num_shots):
        for i in range(num_steps_per_shot):
            idx = shot * num_steps_per_shot + i
            state[:, idx+1] = state[:, idx] + system_dynamics(state[:, idx], optimized_controls[idx]) * dt

    return state

# Main function
def main():
    # Define parameters
    num_steps_per_shot = 100 # Number of time steps per shot
    num_shots = 5  # Number of shots
    total_time = 1.0  # Total time duration
    target_position = 2.0  # Target position
    initial_state = np.array([0, 0])  # Initial state: [0, 0] (position 0, velocity 0)
    target_state = np.array([target_position, 0])  # Final state: [target_position, 0]

    # Run the optimization
    optimized_controls, times = optimize_trajectory(initial_state, target_state, num_steps_per_shot, num_shots, total_time)

    # Simulate the system with the optimized control
    states = simulate_trajectory(optimized_controls, initial_state, total_time / (num_steps_per_shot * num_shots), num_steps_per_shot, num_shots)
    time = np.linspace(0, total_time, num_steps_per_shot * num_shots + 1)
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

if __name__ == "__main__":
    main()