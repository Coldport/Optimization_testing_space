import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, if_else

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)

# Define the system dynamics for position (x) and velocity (v)
def dynamics(state, control_input, mass, friction_coefficient):
    x, v = state
    dxdt = v
    # Frictional force opposes the direction of motion
    frictional_force = friction_coefficient * mass * g * if_else(v != 0, np.sign(v), 0)
    dvdt = (control_input - frictional_force) / mass  # Include friction
    return vertcat(dxdt, dvdt)

# One-shot optimization
def optimize_trajectory(initial_state, target_state, num_steps, total_time, mass, friction_coefficient):
    dt = total_time / num_steps  # Time step size
    times = np.linspace(0, total_time, num_steps + 1)  # Time vector

    # Define symbolic variables
    x = MX.sym('x', num_steps + 1)  # Position
    v = MX.sym('v', num_steps + 1)  # Velocity
    u = MX.sym('u', num_steps)      # Control inputs (force)

    # Define the objective function: Minimize control effort (force squared)
    obj = 0
    for i in range(num_steps):
        obj += u[i]**2  # Sum of squared control inputs

    # Define the constraints
    g = []  # Constraint vector
    lbg = []  # Lower bounds for constraints
    ubg = []  # Upper bounds for constraints

    # Initial state constraints
    g.append(x[0] - initial_state[0])
    g.append(v[0] - initial_state[1])
    lbg += [0, 0]
    ubg += [0, 0]

    # Dynamics constraints
    for i in range(num_steps):
        # Compute the next state using dynamics
        state_next = dynamics([x[i], v[i]], u[i], mass, friction_coefficient) * dt + vertcat(x[i], v[i])
        g.append(x[i + 1] - state_next[0])
        g.append(v[i + 1] - state_next[1])
        lbg += [0, 0]
        ubg += [0, 0]

    # Final state constraints
    g.append(x[-1] - target_state[0])
    g.append(v[-1] - target_state[1])
    lbg += [0, 0]
    ubg += [0, 0]

    # Create the NLP problem
    nlp = {
        'x': vertcat(x, v, u),  # Decision variables
        'f': obj,               # Objective function
        'g': vertcat(*g)        # Constraints
    }

    # Create the solver with SNOPT
    solver = nlpsol('solver', 'snopt', nlp)  # Use SNOPT as the solver

    # Initial guess for decision variables
    x0 = np.zeros(num_steps + 1)  # Initial guess for position
    v0 = np.zeros(num_steps + 1)  # Initial guess for velocity
    u0 = np.ones(num_steps) * 0.1  # Initial guess for control inputs
    x0[0] = initial_state[0]
    v0[0] = initial_state[1]

    # Solve the problem
    result = solver(x0=vertcat(x0, v0, u0), lbg=lbg, ubg=ubg)

    # Extract the optimized control inputs
    optimized_controls = result['x'][-num_steps:].full().flatten()

    return optimized_controls, times

# Simulate the system using the optimized control inputs
def simulate_system(optimized_controls, initial_state, dt, num_steps, mass, friction_coefficient):
    state = np.zeros((2, num_steps + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the optimized control inputs
    for i in range(num_steps):
        # Compute the next state using dynamics
        state_dot = dynamics(state[:, i], optimized_controls[i], mass, friction_coefficient).full().flatten()
        state[:, i + 1] = state[:, i] + state_dot * dt

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
    num_steps = 100  # Number of time steps
    total_time = 1.0  # Total time duration
    target_position = 10.0  # Target position
    mass = 10.0  # Mass of the boulder (kg)
    friction_coefficient = 0.1  # Coefficient of friction
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