import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, if_else
from matplotlib.animation import FuncAnimation
import pandas as pd  # For CSV export

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)

# Define the system dynamics for the boulder and pendulum
def dynamics(state, control_input, mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction):
    x_b, v_b, theta, omega = state
    dxdt = v_b
    # Frictional force opposes the direction of motion
    frictional_force = friction_coefficient * mass_boulder * g * if_else(v_b != 0, np.sign(v_b), 0)
    dvdt = (control_input - frictional_force) / mass_boulder  # Boulder dynamics

    # Pendulum dynamics with rotation friction
    dthetadt = omega
    domegadt = -g / length_pendulum * np.sin(theta) - dvdt * np.cos(theta) / length_pendulum - rotation_friction * omega / (mass_pendulum * length_pendulum**2)

    return vertcat(dxdt, dvdt, dthetadt, domegadt)

# One-shot optimization
def optimize_trajectory(initial_state, target_state, num_steps, total_time, mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction):
    dt = total_time / num_steps  # Time step size
    times = np.linspace(0, total_time, num_steps + 1)  # Time vector

    # Define symbolic variables
    x_b = MX.sym('x_b', num_steps + 1)  # Boulder position
    v_b = MX.sym('v_b', num_steps + 1)  # Boulder velocity
    theta = MX.sym('theta', num_steps + 1)  # Pendulum angle
    omega = MX.sym('omega', num_steps + 1)  # Pendulum angular velocity
    u = MX.sym('u', num_steps)  # Control inputs (force)

    # Define the objective function: Minimize control effort (force squared)
    obj = 0
    for i in range(num_steps):
        obj += u[i]**2  # Sum of squared control inputs

    # Define the constraints
    g = []  # Constraint vector
    lbg = []  # Lower bounds for constraints
    ubg = []  # Upper bounds for constraints

    # Initial state constraints
    g.append(x_b[0] - initial_state[0])
    g.append(v_b[0] - initial_state[1])
    g.append(theta[0] - initial_state[2])
    g.append(omega[0] - initial_state[3])
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]

    # Dynamics constraints
    for i in range(num_steps):
        # Compute the next state using dynamics
        state_next = dynamics([x_b[i], v_b[i], theta[i], omega[i]], u[i], mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction) * dt + vertcat(x_b[i], v_b[i], theta[i], omega[i])
        g.append(x_b[i + 1] - state_next[0])
        g.append(v_b[i + 1] - state_next[1])
        g.append(theta[i + 1] - state_next[2])
        g.append(omega[i + 1] - state_next[3])
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Final state constraints
    g.append(x_b[-1] - target_state[0])
    g.append(v_b[-1] - target_state[1])
    g.append(theta[-1] - target_state[2])
    g.append(omega[-1] - target_state[3])
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]

    # Create the NLP problem
    nlp = {
        'x': vertcat(x_b, v_b, theta, omega, u),  # Decision variables
        'f': obj,                                  # Objective function
        'g': vertcat(*g)                           # Constraints
    }

    # Create the solver with SNOPT
    solver = nlpsol('solver', 'snopt', nlp)  # Use SNOPT as the solver

    # Initial guess for decision variables
    x0 = np.zeros(num_steps + 1)  # Initial guess for boulder position
    v0 = np.zeros(num_steps + 1)  # Initial guess for boulder velocity
    theta0 = np.zeros(num_steps + 1)  # Initial guess for pendulum angle
    omega0 = np.zeros(num_steps + 1)  # Initial guess for pendulum angular velocity
    u0 = np.ones(num_steps) * 0.1  # Initial guess for control inputs
    x0[0] = initial_state[0]
    v0[0] = initial_state[1]
    theta0[0] = initial_state[2]
    omega0[0] = initial_state[3]

    # Solve the problem
    result = solver(x0=vertcat(x0, v0, theta0, omega0, u0), lbg=lbg, ubg=ubg)

    # Extract the optimized control inputs and states
    optimized_controls = result['x'][-num_steps:].full().flatten()
    x_b_opt = result['x'][:num_steps + 1].full().flatten()
    theta_opt = result['x'][num_steps + 1:2 * (num_steps + 1)].full().flatten()

    return optimized_controls, x_b_opt, theta_opt, times

# Simulate the system using the optimized control inputs
def simulate_system(optimized_controls, initial_state, dt, num_steps, mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction):
    state = np.zeros((4, num_steps + 1))  # State vector (x_b, v_b, theta, omega)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the optimized control inputs
    for i in range(num_steps):
        # Compute the next state using dynamics
        state_dot = dynamics(state[:, i], optimized_controls[i], mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction).full().flatten()
        state[:, i + 1] = state[:, i] + state_dot * dt

    return state

# Plot the results
def plot_results(time, x_b, theta, optimized_controls, target_position):
    plt.figure(figsize=(10, 6))

    # Plot boulder position over time
    plt.subplot(3, 1, 1)
    plt.plot(time, x_b, label="Boulder Position")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title(f"Optimal Trajectory: Boulder Position vs. Time (Target = {target_position})")
    plt.grid(True)

    # Plot pendulum angle over time
    plt.subplot(3, 1, 2)
    plt.plot(time, theta, label="Pendulum Angle", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Angle (rad)")
    plt.title("Optimal Trajectory: Pendulum Angle vs. Time")
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

# Animate the trajectory
def animate_trajectory(time, x_b, theta, length_pendulum, start_position=0, target_position=1.0, pen_length=1.0):
    fig, ax = plt.subplots()
    ax.set_xlim(start_position - pen_length, target_position + pen_length)
    ax.set_ylim(-pen_length * 1.3, pen_length * 1.3)
    ax.set_aspect('equal')
    ax.grid(True)

    # Initialize pendulum and boulder
    boulder, = ax.plot([], [], 'bo', markersize=10)
    pendulum, = ax.plot([], [], 'r-', linewidth=2)

    def update(frame):
        # Update boulder position
        boulder.set_data([x_b[frame]], [0])  # Pass as lists

        # Update pendulum position
        x_p = x_b[frame] + length_pendulum * np.sin(theta[frame])
        y_p = -length_pendulum * np.cos(theta[frame])
        pendulum.set_data([x_b[frame], x_p], [0, y_p])  # Pass as lists

        return boulder, pendulum

    ani = FuncAnimation(fig, update, frames=len(time), interval=100, blit=True, repeat_delay=1000)
    plt.show()

# Export results to CSV
def export_to_csv(time, x_b, theta, optimized_controls, filename="results.csv"):
    # Create a DataFrame
    data = {
        "Time": time,
        "Boulder Position": x_b,
        "Pendulum Angle (rad)": theta,
        "Control Input": np.append(optimized_controls, np.nan)  # Append NaN to match length
    }
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")

# Main function
def main():
    # Define parameters
    num_steps = 50  # Number of time steps
    total_time = 10.0  # Total time duration
    target_position = 10.0  # Target position
    mass_boulder = 10.0  # Mass of the boulder (kg)
    friction_coefficient = 0.001  # Coefficient of friction
    mass_pendulum = 10.0  # Mass of the pendulum (kg)
    length_pendulum = 1.0  # Length of the pendulum (m)
    rotation_friction = 0.01  # Rotational friction coefficient
    initial_state = np.array([-5, 0, 0, 0])  # Initial state: [x_b, v_b, theta, omega]
    target_state = np.array([target_position, 0, np.deg2rad(180), 0])  # Final state: [x_b, v_b, theta, omega]

    # Run the optimization
    optimized_controls, x_b_opt, theta_opt, times = optimize_trajectory(
        initial_state, target_state, num_steps, total_time, mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction
    )

    # Simulate the system with the optimized control
    states = simulate_system(optimized_controls, initial_state, total_time / num_steps, num_steps, mass_boulder, friction_coefficient, mass_pendulum, length_pendulum, rotation_friction)
    x_b = states[0, :]
    theta = states[2, :]

    # Plot the results
    plot_results(times, x_b, theta, optimized_controls, target_position)

    # Animate the trajectory
    animate_trajectory(times, x_b, theta, length_pendulum, initial_state[0], target_position, length_pendulum)

    # Export results to CSV
    export_to_csv(times, x_b, theta, optimized_controls, filename="results.csv")

# Entry point of the script
if __name__ == "__main__":
    main()