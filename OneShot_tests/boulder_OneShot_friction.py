import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)

# Define the system dynamics for position (x) and velocity (v)
def dynamics(state, control_input, mass, friction_coefficient):
    x, v = state
    dxdt = v
    # Frictional force opposes the direction of motion
    frictional_force = friction_coefficient * mass * g * np.sign(v) if v != 0 else 0
    dvdt = (control_input - frictional_force) / mass  # Include friction
    return np.array([dxdt, dvdt])

# Objective function: Minimize control effort (force squared)
def minimize_control_effort(control_inputs):
    return np.sum(control_inputs**2)  # Minimize the sum of squared control inputs

# Constraint function: Ensure the system reaches the desired final state
def enforce_final_state(control_inputs, initial_state, target_state, dt, num_steps, mass, friction_coefficient):
    state = np.zeros((2, num_steps + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the control inputs
    for i in range(num_steps):
        state[:, i + 1] = state[:, i] + dynamics(state[:, i], control_inputs[i], mass, friction_coefficient) * dt

    # Final state constraint: position = target_position, velocity = 0
    return np.array([state[0, -1] - target_state[0], state[1, -1] - target_state[1]])

# One-shot optimization
def optimize_trajectory(initial_state, target_state, num_steps, total_time, mass, friction_coefficient):
    dt = total_time / num_steps  # Time step size
    times = np.linspace(0, total_time, num_steps + 1)  # Time vector

    # Initial guess for control inputs (force)
    initial_guess_controls = np.ones(num_steps) * 0.1  # Small initial force

    # Define the constraint dictionary
    constraints = {
        'type': 'eq',
        'fun': enforce_final_state,
        'args': (initial_state, target_state, dt, num_steps, mass, friction_coefficient)
    }

    # Run the optimization to minimize control effort, subject to the final state constraint
    result = minimize(minimize_control_effort, initial_guess_controls, constraints=constraints, method='SLSQP')

    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed:", result.message)

    return result.x, times

# Simulate the system using the optimized control inputs
def simulate_system(optimized_controls, initial_state, dt, num_steps, mass, friction_coefficient):
    state = np.zeros((2, num_steps + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the optimized control inputs
    for i in range(num_steps):
        state[:, i + 1] = state[:, i] + dynamics(state[:, i], optimized_controls[i], mass, friction_coefficient) * dt

    return state

# Plot the results
def plot_results(time, position, velocity, optimized_controls, target_position):
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

# Main function
def main():
    # Define parameters
    num_steps = 50  # Number of time steps
    total_time = 10.0  # Total time duration
    target_position = 5.0  # Target position
    mass = 10.0  # Mass of the boulder (kg)
    friction_coefficient = 0.61  # Coefficient of friction
    initial_state = np.array([0, 0])  # Initial state: [0, 0] (position 0, velocity 0)
    target_state = np.array([target_position, 0])  # Final state: [target_position, 0]

    # Run the optimization
    optimized_controls, times = optimize_trajectory(initial_state, target_state, num_steps, total_time, mass, friction_coefficient)

    # Simulate the system with the optimized control
    states = simulate_system(optimized_controls, initial_state, total_time / num_steps, num_steps, mass, friction_coefficient)
    position = states[0, :]
    velocity = states[1, :]

    # Plot the results
    plot_results(times, position, velocity, optimized_controls, target_position)

# Entry point of the script
if __name__ == "__main__":
    main()