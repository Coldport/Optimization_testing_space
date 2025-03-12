import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol
from matplotlib.animation import FuncAnimation

# Forward Kinematics: Calculate the end-effector position based on joint angles
def forward_kinematics(theta1, theta2, l1, l2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return x1, y1, x2, y2

# Inverse Kinematics: Calculate the joint angles to reach the target position
def inverse_kinematics(x, y, l1, l2):
    d = np.sqrt(x**2 + y**2)
    if d > l1 + l2 or d < np.abs(l1 - l2):
        raise ValueError("Target position is out of reach")
    
    # Calculate theta2 using the law of cosines
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = np.arccos(cos_theta2)
    
    # Two possible solutions: elbow-up and elbow-down
    theta2_up = theta2
    theta2_down = -theta2
    
    # Calculate theta1 for both solutions
    theta1_up = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_up), l1 + l2 * np.cos(theta2_up))
    theta1_down = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_down), l1 + l2 * np.cos(theta2_down))
    
    return (theta1_up, theta2_up), (theta1_down, theta2_down)

# Dynamics of the robot arm
def robot_dynamics(state, control, l1, l2, m1, m2, g=9.81):
    theta1 = state[0]
    theta2 = state[1]
    omega1 = state[2]
    omega2 = state[3]
    tau1 = control[0]
    tau2 = control[1]

    # Mass and length properties
    I1 = (1/12) * m1 * l1**2  # Moment of inertia for link 1
    I2 = (1/12) * m2 * l2**2  # Moment of inertia for link 2

    # Equations of motion (simplified dynamics)
    alpha1 = (tau1 - tau2 - m2 * l1 * l2 * (omega2**2 * np.sin(theta2) + 2 * omega1 * omega2 * np.sin(theta2))) / (I1 + I2 + m2 * l1**2)
    alpha2 = (tau2 + m2 * l1 * l2 * omega1**2 * np.sin(theta2)) / I2

    return vertcat(omega1, omega2, alpha1, alpha2)

# SNOPT-based optimization to find joint angles and velocities
def optimize_trajectory(initial_state, target_state, l1, l2, m1, m2, num_steps, total_time):
    dt = total_time / num_steps
    times = np.linspace(0, total_time, num_steps + 1)

    # Define symbolic variables
    theta1 = MX.sym('theta1', num_steps + 1)
    theta2 = MX.sym('theta2', num_steps + 1)
    omega1 = MX.sym('omega1', num_steps + 1)
    omega2 = MX.sym('omega2', num_steps + 1)
    tau1 = MX.sym('tau1', num_steps)
    tau2 = MX.sym('tau2', num_steps)

    # Objective function: Minimize control effort (torque squared)
    obj = 0
    for i in range(num_steps):
        obj += tau1[i]**2 + tau2[i]**2

    # Constraints
    g = []
    lbg = []
    ubg = []

    # Initial state constraints
    g.append(theta1[0] - initial_state[0])
    g.append(theta2[0] - initial_state[1])
    g.append(omega1[0] - initial_state[2])
    g.append(omega2[0] - initial_state[3])
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]

    # Dynamics constraints
    for i in range(num_steps):
        state = vertcat(theta1[i], theta2[i], omega1[i], omega2[i])
        control = vertcat(tau1[i], tau2[i])
        state_next = state + robot_dynamics(state, control, l1, l2, m1, m2) * dt
        g.append(theta1[i + 1] - state_next[0])
        g.append(theta2[i + 1] - state_next[1])
        g.append(omega1[i + 1] - state_next[2])
        g.append(omega2[i + 1] - state_next[3])
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Final state constraints
    g.append(theta1[-1] - target_state[0])
    g.append(theta2[-1] - target_state[1])
    g.append(omega1[-1] - target_state[2])
    g.append(omega2[-1] - target_state[3])
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]

    # Create the NLP problem
    nlp = {
        'x': vertcat(theta1, theta2, omega1, omega2, tau1, tau2),
        'f': obj,
        'g': vertcat(*g)
    }

    # Solve with SNOPT
    solver = nlpsol('solver', 'snopt', nlp)
    x0 = np.zeros(nlp['x'].shape[0])
    result = solver(x0=x0, lbg=lbg, ubg=ubg)

    # Extract results
    theta1_opt = result['x'][:num_steps + 1].full().flatten()
    theta2_opt = result['x'][num_steps + 1:2 * (num_steps + 1)].full().flatten()
    omega1_opt = result['x'][2 * (num_steps + 1):3 * (num_steps + 1)].full().flatten()
    omega2_opt = result['x'][3 * (num_steps + 1):4 * (num_steps + 1)].full().flatten()
    tau1_opt = result['x'][4 * (num_steps + 1):4 * (num_steps + 1) + num_steps].full().flatten()
    tau2_opt = result['x'][4 * (num_steps + 1) + num_steps:].full().flatten()

    return theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times

# Main function
def main():
    # Robot parameters
    l1 = 5
    l2 = 5
    m1 = 5
    m2 = 0.51
    num_steps = 100
    total_time = 10.0

    # Initial state (fully extended, 0 angle)
    initial_state = np.array([0, 0, 0, 0])  # [theta1, theta2, omega1, omega2]

    # Create a figure with subplots for animation and plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Robot Arm Simulation")

    # Initialize the robot arm plot
    ax1.set_xlim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_ylim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Robot Arm Animation")

    link1, = ax1.plot([], [], 'b-', linewidth=2)
    link2, = ax1.plot([], [], 'r-', linewidth=2)

    # Initialize the angular velocity plot
    ax2.set_xlim(0, total_time)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.set_title("Angular Velocities")
    ax2.grid(True)

    omega1_line, = ax2.plot([], [], label="Omega1 (Link 1)")
    omega2_line, = ax2.plot([], [], label="Omega2 (Link 2)")
    ax2.legend()

    # Initialize the torque plot
    ax3.set_xlim(0, total_time)
    ax3.set_ylim(-10, 10)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Torque (Nm)")
    ax3.set_title("Torques")
    ax3.grid(True)

    tau1_line, = ax3.plot([], [], label="Tau1 (Link 1)")
    tau2_line, = ax3.plot([], [], label="Tau2 (Link 2)")
    ax3.legend()

    # Store the target position and animation object
    target_x, target_y = None, None
    ani = None

    # Function to update the animation and plots
    def update_animation_and_plots(theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times):
        nonlocal ani

        # Update the angular velocity plot
        omega1_line.set_data(times, omega1_opt)
        omega2_line.set_data(times, omega2_opt)
        ax2.relim()
        ax2.autoscale_view()

        # Update the torque plot
        tau1_line.set_data(times[:-1], tau1_opt)
        tau2_line.set_data(times[:-1], tau2_opt)
        ax3.relim()
        ax3.autoscale_view()

        # Update the robot arm animation
        def update_arm(frame):
            x1, y1, x2, y2 = forward_kinematics(theta1_opt[frame], theta2_opt[frame], l1, l2)
            link1.set_data([0, x1], [0, y1])
            link2.set_data([x1, x2], [y1, y2])
            return link1, link2

        # Stop the previous animation if it exists
        if ani is not None:
            ani.event_source.stop()

        # Create a new animation
        ani = FuncAnimation(fig, update_arm, frames=len(times), interval=50, blit=True, repeat_delay=1000)

        plt.draw()

    # Function to handle mouse clicks
    def on_click(event):
        nonlocal target_x, target_y
        target_x, target_y = event.xdata, event.ydata
        print(f"Target position set to: ({target_x}, {target_y})")

        try:
            # Calculate inverse kinematics
            (theta1_up, theta2_up), (theta1_down, theta2_down) = inverse_kinematics(target_x, target_y, l1, l2)
            target_state = np.array([theta1_up, theta2_up, 0, 0])  # Target state (elbow-up solution)

            # Optimize trajectory
            theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times = optimize_trajectory(
                initial_state, target_state, l1, l2, m1, m2, num_steps, total_time
            )

            # Update the animation and plots
            update_animation_and_plots(theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times)

        except ValueError as e:
            print(e)

    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Entry point
if __name__ == "__main__":
    main()