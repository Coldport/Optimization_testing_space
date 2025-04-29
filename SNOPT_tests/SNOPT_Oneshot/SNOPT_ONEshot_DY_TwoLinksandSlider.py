import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle

# Forward Kinematics: Calculate the end-effector position based on joint angles and slider position
def forward_kinematics(slider_pos, theta1, theta2, l1, l2):
    # Debugging: Print inputs
    #print(f"Inputs: slider_pos={slider_pos}, theta1={theta1}, theta2={theta2}, l1={l1}, l2={l2}")
    
    # Calculate positions
    x1 = slider_pos + l1 * np.cos(theta1)
    y1 = 0 + l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    
    # Debugging: Print outputs
    #print(f"Forward Kinematics: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    return x1, y1, x2, y2

# Inverse Kinematics: Calculate the joint angles to reach the target position given the slider position
def inverse_kinematics(slider_pos, x, y, l1, l2):
    dx = x - slider_pos
    dy = y
    d = np.sqrt(dx**2 + dy**2)
    
    if d > l1 + l2 or d < np.abs(l1 - l2):
        raise ValueError("Target position is out of reach")
    
    cos_theta2 = (dx**2 + dy**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = np.arccos(cos_theta2)
    
    theta1 = np.arctan2(dy, dx) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    
    # Debugging output
    #print(f"Inverse Kinematics: theta1={theta1}, theta2={theta2}")
    
    return theta1, theta2

# Dynamics of the robot arm with slider
def robot_dynamics(state, control, l1, l2, m1, m2, g=9.81):
    slider_pos = state[0]
    theta1 = state[1]
    theta2 = state[2]
    slider_vel = state[3]
    omega1 = state[4]
    omega2 = state[5]
    force = control[0]
    tau1 = control[1]
    tau2 = control[2]  # Assuming a second torque is applied to the second link

    # Mass and length properties
    I1 = (1/12) * m1 * l1**2  # Moment of inertia for link 1
    I2 = (1/12) * m2 * l2**2  # Moment of inertia for link 2

    # Equations of motion (simplified dynamics)
    slider_acc = force / m1
    alpha1 = (tau1 + m2 * l2 * slider_acc * np.sin(theta1)) / I1
    alpha2 = (tau2 + m2 * l2 * slider_acc * np.sin(theta1 + theta2)) / I2

    return vertcat(slider_vel, omega1, omega2, slider_acc, alpha1, alpha2)

# SNOPT-based optimization to find joint angles and velocities with torque and speed limits
def optimize_trajectory(initial_state, target_state, l1, l2, m1, m2, num_steps, total_time, max_force, max_torque, max_speed):
    dt = total_time / num_steps
    times = np.linspace(0, total_time, num_steps + 1)

    # Define symbolic variables (Ensure proper shapes)
    slider_pos = MX.sym('slider_pos', num_steps + 1, 1)
    theta1 = MX.sym('theta1', num_steps + 1, 1)
    theta2 = MX.sym('theta2', num_steps + 1, 1)
    slider_vel = MX.sym('slider_vel', num_steps + 1, 1)
    omega1 = MX.sym('omega1', num_steps + 1, 1)
    omega2 = MX.sym('omega2', num_steps + 1, 1)
    force = MX.sym('force', num_steps, 1)
    tau1 = MX.sym('tau1', num_steps, 1)
    tau2 = MX.sym('tau2', num_steps, 1)

    # Objective function: Minimize control effort (force and torque squared)
    obj = sum(force[i]**2 + tau1[i]**2 + tau2[i]**2 for i in range(num_steps))

    # Constraints
    g, lbg, ubg = [], [], []

    # Initial state constraints (Ensure dimensions match)
    g += [slider_pos[0] - initial_state[0], 
          theta1[0] - initial_state[1], 
          theta2[0] - initial_state[2],
          slider_vel[0] - initial_state[3], 
          omega1[0] - initial_state[4], 
          omega2[0] - initial_state[5]]
    lbg += [0] * 6
    ubg += [0] * 6

    # Dynamics constraints (Ensure CasADi expressions remain symbolic)
    for i in range(num_steps):
        state = vertcat(slider_pos[i], theta1[i], theta2[i], slider_vel[i], omega1[i], omega2[i])
        control = vertcat(force[i], tau1[i], tau2[i])
        state_next = state + robot_dynamics(state, control, l1, l2, m1, m2) * dt
        
        g += [slider_pos[i + 1] - state_next[0],
              theta1[i + 1] - state_next[1],
              theta2[i + 1] - state_next[2],
              slider_vel[i + 1] - state_next[3],
              omega1[i + 1] - state_next[4],
              omega2[i + 1] - state_next[5]]
        lbg += [0] * 6
        ubg += [0] * 6

    # Final state constraints
    g += [slider_pos[-1] - target_state[0], 
          theta1[-1] - target_state[1], 
          theta2[-1] - target_state[2],
          slider_vel[-1] - target_state[3], 
          omega1[-1] - target_state[4], 
          omega2[-1] - target_state[5]]
    lbg += [0] * 6
    ubg += [0] * 6

    # Define the NLP problem
    nlp = {
        'x': vertcat(slider_pos, theta1, theta2, slider_vel, omega1, omega2, force, tau1, tau2), 
        'f': obj, 
        'g': vertcat(*g)
    }

    # Bounds (Ensure shape consistency)
    lbx = [-np.inf] * (3 * (num_steps + 1)) + [-max_speed] * (3 * (num_steps + 1)) + [-max_force] * num_steps + [-max_torque] * (2 * num_steps)
    ubx = [np.inf] * (3 * (num_steps + 1)) + [max_speed] * (3 * (num_steps + 1)) + [max_force] * num_steps + [max_torque] * (2 * num_steps)

    # Solve with SNOPT
    solver = nlpsol('solver', 'snopt', nlp)
    x0 = np.zeros(nlp['x'].shape[0])
    result = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    # Extract results (Ensure proper `.full().flatten()` usage)
    slider_pos_opt = result['x'][:num_steps + 1].full().flatten()
    theta1_opt = result['x'][num_steps + 1:2 * (num_steps + 1)].full().flatten()
    theta2_opt = result['x'][2 * (num_steps + 1):3 * (num_steps + 1)].full().flatten()
    slider_vel_opt = result['x'][3 * (num_steps + 1):4 * (num_steps + 1)].full().flatten()
    omega1_opt = result['x'][4 * (num_steps + 1):5 * (num_steps + 1)].full().flatten()
    omega2_opt = result['x'][5 * (num_steps + 1):6 * (num_steps + 1)].full().flatten()
    force_opt = result['x'][6 * (num_steps + 1):6 * (num_steps + 1) + num_steps].full().flatten()
    tau1_opt = result['x'][6 * (num_steps + 1) + num_steps:6 * (num_steps + 1) + 2 * num_steps].full().flatten()
    tau2_opt = result['x'][6 * (num_steps + 1) + 2 * num_steps:].full().flatten()

    return slider_pos_opt, theta1_opt, theta2_opt, slider_vel_opt, omega1_opt, omega2_opt, force_opt, tau1_opt, tau2_opt, times

# Main function
def main():
    # Robot parameters
    l1 = 5
    l2 = 5
    m1 = 0.5
    m2 = 0.5
    num_steps = 50
    total_time = 300.0
    max_force = 100.0  # Default maximum force (N)
    max_torque = 1000.0  # Default maximum torque (Nm)
    max_speed = 500.0    # Default maximum speed (m/s and rad/s)

    # Initial state (slider at 0, both links vertical)
    initial_state = np.array([0, 0, 0, 0, 0, 0])  # [slider_pos, theta1, theta2, slider_vel, omega1, omega2]

    # Initialize target position (set to a default value)
    target_x, target_y = 0, 0  # Example target position

    # Create a figure with subplots for animation and plots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Robot arm animation
    ax2 = fig.add_subplot(gs[1, 0])  # Velocity plot
    ax3 = fig.add_subplot(gs[2, 0])  # Force and torque plot
    ax_sliders = fig.add_subplot(gs[:, 1])  # Sliders
    ax_sliders.axis('off')  # Hide the axes

    # Initialize the robot arm plot
    ax1.set_xlim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_ylim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Robot Arm Animation")

    # Initialize the links and cart
    link1, = ax1.plot([], [], 'b-', linewidth=2, label="Link 1")  # Blue line for Link 1
    link2, = ax1.plot([], [], 'r-', linewidth=2, label="Link 2")  # Red line for Link 2
    target_point, = ax1.plot([], [], 'ro', markersize=8, label="Target Position")
    cart_width = 1.0
    cart_height = 0.5
    cart = Rectangle((0 - cart_width / 2, -cart_height / 2), cart_width, cart_height, color='g', label="Cart")
    ax1.add_patch(cart)

    # Initialize the velocity plot
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Velocity (m/s and rad/s)")
    ax2.set_title("Velocities")
    ax2.grid(True)

    slider_vel_line, = ax2.plot([], [], label="Slider Velocity")
    omega1_line, = ax2.plot([], [], label="Omega1 (Link 1)")
    omega2_line, = ax2.plot([], [], label="Omega2 (Link 2)")
    
    # Add speed limit lines
    speed_limit_upper, = ax2.plot([], [], 'k--', label="Speed Limit")
    speed_limit_lower, = ax2.plot([], [], 'k--')
    
    ax2.legend()

    # Initialize the force and torque plot
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Force (N) and Torque (Nm)")
    ax3.set_title("Force and Torque")
    ax3.grid(True)

    force_line, = ax3.plot([], [], label="Force (Slider)")
    tau1_line, = ax3.plot([], [], label="Tau1 (Link 1)")
    tau2_line, = ax3.plot([], [], label="Tau2 (Link 2)")
    
    # Add force and torque limit lines
    force_limit_upper, = ax3.plot([], [], 'k--', label="Force Limit")
    force_limit_lower, = ax3.plot([], [], 'k--')
    torque_limit_upper, = ax3.plot([], [], 'k--', label="Torque Limit")
    torque_limit_lower, = ax3.plot([], [], 'k--')
    
    ax3.legend()

    # Add vertical lines to track time in plots
    vline_ax2 = ax2.axvline(x=0, color='r', linestyle='--')
    vline_ax3 = ax3.axvline(x=0, color='r', linestyle='--')

    # Add sliders for time, link lengths, weights, force limit, torque limit, and speed limit
    slider_y_positions = [0.85, 0.78, 0.71, 0.64, 0.57, 0.50, 0.43, 0.36]
    slider_width = 0.14
    slider_height = 0.03
    slider_x = 0.82
    
    ax_time = plt.axes([slider_x, slider_y_positions[0], slider_width, slider_height])
    ax_l1 = plt.axes([slider_x, slider_y_positions[1], slider_width, slider_height])
    ax_l2 = plt.axes([slider_x, slider_y_positions[2], slider_width, slider_height])
    ax_m1 = plt.axes([slider_x, slider_y_positions[3], slider_width, slider_height])
    ax_m2 = plt.axes([slider_x, slider_y_positions[4], slider_width, slider_height])
    ax_force = plt.axes([slider_x, slider_y_positions[5], slider_width, slider_height])
    ax_torque = plt.axes([slider_x, slider_y_positions[6], slider_width, slider_height])
    ax_speed = plt.axes([slider_x, slider_y_positions[7], slider_width, slider_height])

    time_slider = Slider(ax_time, 'Time', 1, 30, valinit=total_time)
    l1_slider = Slider(ax_l1, 'L1', 1, 10, valinit=l1)
    l2_slider = Slider(ax_l2, 'L2', 1, 10, valinit=l2)
    m1_slider = Slider(ax_m1, 'M1', 0.1, 2, valinit=m1)
    m2_slider = Slider(ax_m2, 'M2', 0.1, 2, valinit=m2)
    force_slider = Slider(ax_force, 'Max Force', 1, 20, valinit=max_force)
    torque_slider = Slider(ax_torque, 'Max Torque', 1, 20, valinit=max_torque)
    speed_slider = Slider(ax_speed, 'Max Speed', 1, 10, valinit=max_speed)

    # Store the target position and animation object
    target_x, target_y = 6.95, 1.15  # Default target position
    ani = None

    # Function to update the limit lines on plots
    def update_limit_lines(max_force, max_torque, max_speed, times):
        # Update speed limit lines
        speed_limit_upper.set_data([times[0], times[-1]], [max_speed, max_speed])
        speed_limit_lower.set_data([times[0], times[-1]], [-max_speed, -max_speed])
        
        # Update force limit lines
        force_limit_upper.set_data([times[0], times[-1]], [max_force, max_force])
        force_limit_lower.set_data([times[0], times[-1]], [-max_force, -max_force])
        
        # Update torque limit lines
        torque_limit_upper.set_data([times[0], times[-1]], [max_torque, max_torque])
        torque_limit_lower.set_data([times[0], times[-1]], [-max_torque, -max_torque])

    # Function to update the animation and plots
    def update_animation_and_plots(slider_pos_opt, theta1_opt, theta2_opt, slider_vel_opt, omega1_opt, omega2_opt, force_opt, tau1_opt, tau2_opt, times, max_force, max_torque, max_speed):
        nonlocal ani

        # Update the velocity plot
        slider_vel_line.set_data(times, slider_vel_opt)
        omega1_line.set_data(times, omega1_opt)
        omega2_line.set_data(times, omega2_opt)
        ax2.set_xlim(0, times[-1])
        
        # Set y-limits to show the speed limits and data
        velocity_range = max(max(np.abs(slider_vel_opt)), max(np.abs(omega1_opt)), max(np.abs(omega2_opt)), max_speed)
        ax2.set_ylim(-velocity_range * 1.1, velocity_range * 1.1)
        
        # Update the force and torque plot
        force_line.set_data(times[:-1], force_opt)
        tau1_line.set_data(times[:-1], tau1_opt)
        tau2_line.set_data(times[:-1], tau2_opt)
        ax3.set_xlim(0, times[-1])
        
        # Set y-limits to show the force and torque limits and data
        force_range = max(max(np.abs(force_opt)), max_force)
        torque_range = max(max(np.abs(tau1_opt)), max(np.abs(tau2_opt)), max_torque)
        ax3.set_ylim(-max(force_range, torque_range) * 1.1, max(force_range, torque_range) * 1.1)
        
        # Update limit lines
        update_limit_lines(max_force, max_torque, max_speed, times)

        # Update the robot arm animation
        # In the update_arm function
        def update_arm(frame):
            # Calculate the positions of the links
            x1, y1, x2, y2 = forward_kinematics(slider_pos_opt[frame], theta1_opt[frame], theta2_opt[frame], l1, l2)
    
            # Debugging: Print frame and positions
            #print(f"Frame {frame}: Slider Pos={slider_pos_opt[frame]}, Theta1={theta1_opt[frame]}, Theta2={theta2_opt[frame]}")
            #print(f"Positions: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
            # Update the positions of the links
            link1.set_data([slider_pos_opt[frame], x1], [0, y1])  # Link 1: from slider to end of first link
            link2.set_data([x1, x2], [y1, y2])                   # Link 2: from end of first link to end of second link
    
            # Update the cart position
            cart.set_xy([slider_pos_opt[frame] - cart_width / 2, -cart_height / 2])
    
            # Update the vertical lines in the plots
            vline_ax2.set_xdata([times[frame], times[frame]])
            vline_ax3.set_xdata([times[frame], times[frame]])
            
            return link1, link2, cart, vline_ax2, vline_ax3
        # Stop the previous animation if it exists
        if ani is not None:
            ani.event_source.stop()

        # Create a new animation
        ani = FuncAnimation(fig, update_arm, frames=len(times), interval=5, blit=True, repeat_delay=1000)

        plt.draw()

        # Function to handle mouse clicks
        # Function to handle mouse clicks
    def on_click(event):
        nonlocal target_x, target_y, initial_state
    
        # Check if the click is within the animation axes
        if event.inaxes != ax1:
            return
        
        target_x, target_y = event.xdata, event.ydata
        #print(f"Target position set to: ({target_x:.2f}, {target_y:.2f})")
    
        # Mark the target position with a red dot
        target_point.set_data([target_x], [target_y])

        try:
            # Get current parameter values
            current_l1 = l1_slider.val
            current_l2 = l2_slider.val
            current_m1 = m1_slider.val
            current_m2 = m2_slider.val
            current_time = time_slider.val
            current_max_force = force_slider.val
            current_max_torque = torque_slider.val
            current_max_speed = speed_slider.val
            
            # Calculate inverse kinematics
            theta1, theta2 = inverse_kinematics(initial_state[0], target_x, target_y, current_l1, current_l2)
            target_state = np.array([target_x, theta1, theta2, 0, 0, 0])  # Target state
    
            # Debugging: Print initial and target states
            #print(f"Initial State: {initial_state}")
            #print(f"Target State: {target_state}")
    
            # Optimize trajectory with current parameters
            slider_pos_opt, theta1_opt, theta2_opt, slider_vel_opt, omega1_opt, omega2_opt, force_opt, tau1_opt, tau2_opt, times = optimize_trajectory(
                initial_state, target_state, current_l1, current_l2, current_m1, current_m2, 
                num_steps, current_time, current_max_force, current_max_torque, current_max_speed
            )
    
            # Debugging: Print optimized trajectory
            #print(f"Optimized Slider Position: {slider_pos_opt[-1]}")
            #print(f"Optimized Theta1: {theta1_opt[-1]}, Optimized Theta2: {theta2_opt[-1]}")
    
            # Update the animation and plots
            update_animation_and_plots(slider_pos_opt, theta1_opt, theta2_opt, slider_vel_opt, omega1_opt, omega2_opt, force_opt, tau1_opt, tau2_opt, times, 
                                      current_max_force, current_max_torque, current_max_speed)
    
            # Update the initial state for the next trajectory
            initial_state = np.array([slider_pos_opt[-1], theta1_opt[-1], theta2_opt[-1], slider_vel_opt[-1], omega1_opt[-1], omega2_opt[-1]])
            #print(f"Updated Initial State: {initial_state}")
    
        except ValueError as e:
            print(f"Error: {e}")

# Function to handle slider updates
    # In the update_sliders function
    def update_sliders(val):
        nonlocal l1, l2, m1, m2, total_time, max_force, max_torque, max_speed, initial_state
        l1 = l1_slider.val
        l2 = l2_slider.val
        m1 = m1_slider.val
        m2 = m2_slider.val
        total_time = time_slider.val
        max_force = force_slider.val
        max_torque = torque_slider.val
        max_speed = speed_slider.val
    
        # Reset the initial state when parameters are changed
        initial_state = np.array([0, 0, 0, 0, 0, 0])  # Reset to default initial state
    
        # Re-run the optimization and update the animation
        if target_x is not None and target_y is not None:
            try:
                # Calculate inverse kinematics
                theta1, theta2 = inverse_kinematics(initial_state[0], target_x, target_y, l1, l2)
                target_state = np.array([target_x, theta1, theta2, 0, 0, 0])  # Target state

                # Optimize trajectory
                slider_pos_opt, theta1_opt, theta2_opt, slider_vel_opt, omega1_opt, omega2_opt, force_opt, tau1_opt, tau2_opt, times = optimize_trajectory(
                initial_state, target_state, l1, l2, m1, m2, num_steps, total_time, max_force, max_torque, max_speed
                )

                # Update the animation and plots
                update_animation_and_plots(slider_pos_opt, theta1_opt, theta2_opt, slider_vel_opt, omega1_opt, omega2_opt, force_opt, tau1_opt, tau2_opt, times, 
                                      max_force, max_torque, max_speed)

                # Update the initial state for the next trajectory
                initial_state = np.array([slider_pos_opt[-1], theta1_opt[-1], theta2_opt[-1], slider_vel_opt[-1], omega1_opt[-1], omega2_opt[-1]])
                #print(f"Updated Initial State: {initial_state}")
    
            except ValueError as e:
                print(f"Error: {e}")

    # Connect the sliders to the update function
    l1_slider.on_changed(update_sliders)
    l2_slider.on_changed(update_sliders)
    m1_slider.on_changed(update_sliders)
    m2_slider.on_changed(update_sliders)
    time_slider.on_changed(update_sliders)
    force_slider.on_changed(update_sliders)
    torque_slider.on_changed(update_sliders)
    speed_slider.on_changed(update_sliders)

    # Add a text instruction on the figure
    plt.figtext(0.5, 0.95, "Click in the plot to set a target position for the robot arm", 
                ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()
    

# Entry point
if __name__ == "__main__":
    main()