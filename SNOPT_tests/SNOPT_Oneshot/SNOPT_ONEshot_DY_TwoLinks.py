import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, sqrt, fmax, fmin, if_else, DM
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

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



def point_to_segment_distance(px, py, x1, y1, x2, y2):
    # Vector from start to end of segment
    dx = x2 - x1
    dy = y2 - y1
    # Length of segment squared
    l2 = dx*dx + dy*dy
    
    # Use CasADi's conditional expressions instead of if statements
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    l2_safe = if_else(l2 < epsilon, epsilon, l2)
    
    # Calculate parameter t for projection
    t_raw = ((px - x1) * dx + (py - y1) * dy) / l2_safe
    t = fmin(1, fmax(0, t_raw))
    
    # Find closest point on segment
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # Return distance to closest point
    return sqrt((px - proj_x)**2 + (py - proj_y)**2)

# SNOPT-based optimization to find joint angles and velocities with torque and speed limits and obstacle avoidance
def optimize_trajectory(initial_state, target_state, l1, l2, m1, m2, num_steps, total_time, max_torque, max_speed, obstacles):
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

    # Obstacle avoidance constraints
    obstacle_radius = 0.5  # Radius of the obstacle (safety margin)
    min_distance = 0.1    # Minimum allowable distance between link and obstacle
    
    if obstacles:
        for i in range(num_steps + 1):
            # Calculate the positions of the joints based on the angles
            x1 = l1 * MX.cos(theta1[i])
            y1 = l1 * MX.sin(theta1[i])
            x2 = x1 + l2 * MX.cos(theta1[i] + theta2[i])
            y2 = y1 + l2 * MX.sin(theta1[i] + theta2[i])
            
            for obs_x, obs_y, obs_rad in obstacles:
                # Calculate distance from obstacle to link 1 (origin to first joint)
                dist_link1 = point_to_segment_distance(obs_x, obs_y, 0, 0, x1, y1) - obs_rad - min_distance
                
                # Calculate distance from obstacle to link 2 (first joint to end effector)
                dist_link2 = point_to_segment_distance(obs_x, obs_y, x1, y1, x2, y2) - obs_rad - min_distance
                
                # Add constraints: distances must be >= 0 (no collision)
                g.append(dist_link1)
                lbg.append(0)
                ubg.append(MX.inf)
                
                g.append(dist_link2)
                lbg.append(0)
                ubg.append(MX.inf)

    # Create the NLP problem
    # Inside the optimize_trajectory function, replace the solver call section with this:

    # Create the NLP problem
    nlp = {
        'x': vertcat(*[theta1, theta2, omega1, omega2, tau1, tau2]),
        'f': obj,
        'g': vertcat(*g)
    }

    # Bounds for the variables (adding torque and speed limits)
    lbx = []
    ubx = []

    # No limits on theta1 and theta2 (full rotation allowed)
    lbx += [-np.inf] * (num_steps + 1)  # Lower bound for theta1
    ubx += [np.inf] * (num_steps + 1)   # Upper bound for theta1
    lbx += [-np.inf] * (num_steps + 1)  # Lower bound for theta2
    ubx += [np.inf] * (num_steps + 1)   # Upper bound for theta2

    # Speed limits for omega1 and omega2
    lbx += [-max_speed] * (num_steps + 1)  # Lower bound for omega1
    ubx += [max_speed] * (num_steps + 1)   # Upper bound for omega1
    lbx += [-max_speed] * (num_steps + 1)  # Lower bound for omega2
    ubx += [max_speed] * (num_steps + 1)   # Upper bound for omega2

    # Torque limits for tau1 and tau2
    lbx += [-max_torque] * num_steps  # Lower bound for tau1
    ubx += [max_torque] * num_steps   # Upper bound for tau1
    lbx += [-max_torque] * num_steps  # Lower bound for tau2
    ubx += [max_torque] * num_steps   # Upper bound for tau2

    # Convert the NumPy arrays to CasADi DM types
    lbx_dm = DM(lbx)
    ubx_dm = DM(ubx)
    lbg_dm = DM(lbg)
    ubg_dm = DM(ubg)
    x0_dm = DM.zeros(nlp['x'].shape[0])

    # Solve with SNOPT - using proper CasADi types
    solver = nlpsol('solver', 'snopt', nlp)
    result = solver(x0=x0_dm, lbx=lbx_dm, ubx=ubx_dm, lbg=lbg_dm, ubg=ubg_dm)

    # Extract results
    theta1_opt = np.array(result['x'][:num_steps + 1].full()).flatten()
    theta2_opt = np.array(result['x'][num_steps + 1:2 * (num_steps + 1)].full()).flatten()
    omega1_opt = np.array(result['x'][2 * (num_steps + 1):3 * (num_steps + 1)].full()).flatten()
    omega2_opt = np.array(result['x'][3 * (num_steps + 1):4 * (num_steps + 1)].full()).flatten()
    tau1_opt = np.array(result['x'][4 * (num_steps + 1):4 * (num_steps + 1) + num_steps].full()).flatten()
    tau2_opt = np.array(result['x'][4 * (num_steps + 1) + num_steps:].full()).flatten()

    return theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times

# Main function
def main():
    # Robot parameters
    l1 = 5
    l2 = 5
    m1 = 0.5
    m2 = 0.51
    num_steps = 100
    total_time = 10.0
    max_torque = 10.0  # Default maximum torque (Nm)
    max_speed = 5.0    # Default maximum angular velocity (rad/s)

    # Initial state (fully extended, 0 angle)
    initial_state = np.array([0, 0, 0, 0])  # [theta1, theta2, omega1, omega2]

    # Create a figure with subplots for animation and plots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Robot arm animation
    ax2 = fig.add_subplot(gs[1, 0])  # Angular velocity plot
    ax3 = fig.add_subplot(gs[2, 0])  # Torque plot
    ax_sliders = fig.add_subplot(gs[:, 1])  # Sliders
    ax_sliders.axis('off')  # Hide the axes

    # Initialize the robot arm plot
    ax1.set_xlim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_ylim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Robot Arm Animation")

    link1, = ax1.plot([], [], 'b-', linewidth=2)
    link2, = ax1.plot([], [], 'r-', linewidth=2)
    target_point, = ax1.plot([], [], 'ro', markersize=8)  # Red dot for target position

    # List to store obstacles (x, y, radius)
    obstacles = []
    obstacle_circles = []  # To store the visual representation of obstacles
    obstacle_radius = 0.5  # Default radius for obstacles

    # Initialize the angular velocity plot
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.set_title("Angular Velocities")
    ax2.grid(True)

    omega1_line, = ax2.plot([], [], label="Omega1 (Link 1)")
    omega2_line, = ax2.plot([], [], label="Omega2 (Link 2)")
    
    # Add speed limit lines
    speed_limit_upper, = ax2.plot([], [], 'k--', label="Speed Limit")
    speed_limit_lower, = ax2.plot([], [], 'k--')
    
    ax2.legend()

    # Initialize the torque plot
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Torque (Nm)")
    ax3.set_title("Torques")
    ax3.grid(True)

    tau1_line, = ax3.plot([], [], label="Tau1 (Link 1)")
    tau2_line, = ax3.plot([], [], label="Tau2 (Link 2)")
    
    # Add torque limit lines
    torque_limit_upper, = ax3.plot([], [], 'k--', label="Torque Limit")
    torque_limit_lower, = ax3.plot([], [], 'k--')
    
    ax3.legend()

    # Add vertical lines to track time in plots
    vline_ax2 = ax2.axvline(x=0, color='r', linestyle='--')
    vline_ax3 = ax3.axvline(x=0, color='r', linestyle='--')

    # Add sliders for time, link lengths, weights, torque limit, and speed limit
    slider_y_positions = [0.85, 0.78, 0.71, 0.64, 0.57, 0.50, 0.43, 0.36]
    slider_width = 0.14
    slider_height = 0.03
    slider_x = 0.82
    
    ax_time = plt.axes([slider_x, slider_y_positions[0], slider_width, slider_height])
    ax_l1 = plt.axes([slider_x, slider_y_positions[1], slider_width, slider_height])
    ax_l2 = plt.axes([slider_x, slider_y_positions[2], slider_width, slider_height])
    ax_m1 = plt.axes([slider_x, slider_y_positions[3], slider_width, slider_height])
    ax_m2 = plt.axes([slider_x, slider_y_positions[4], slider_width, slider_height])
    ax_torque = plt.axes([slider_x, slider_y_positions[5], slider_width, slider_height])
    ax_speed = plt.axes([slider_x, slider_y_positions[6], slider_width, slider_height])
    ax_obs_radius = plt.axes([slider_x, slider_y_positions[7], slider_width, slider_height])

    time_slider = Slider(ax_time, 'Time', 1, 20, valinit=total_time)
    l1_slider = Slider(ax_l1, 'L1', 1, 10, valinit=l1)
    l2_slider = Slider(ax_l2, 'L2', 1, 10, valinit=l2)
    m1_slider = Slider(ax_m1, 'M1', 0.1, 2, valinit=m1)
    m2_slider = Slider(ax_m2, 'M2', 0.1, 2, valinit=m2)
    torque_slider = Slider(ax_torque, 'Max Torque', 1, 20, valinit=max_torque)
    speed_slider = Slider(ax_speed, 'Max Speed', 1, 10, valinit=max_speed)
    obs_radius_slider = Slider(ax_obs_radius, 'Obs Radius', 0.1, 2, valinit=obstacle_radius)

    # Store the target position and animation object
    target_x, target_y = None, None
    ani = None

    # Function to update the limit lines on plots
    def update_limit_lines(max_torque, max_speed, times):
        # Update speed limit lines
        speed_limit_upper.set_data([times[0], times[-1]], [max_speed, max_speed])
        speed_limit_lower.set_data([times[0], times[-1]], [-max_speed, -max_speed])
        
        # Update torque limit lines
        torque_limit_upper.set_data([times[0], times[-1]], [max_torque, max_torque])
        torque_limit_lower.set_data([times[0], times[-1]], [-max_torque, -max_torque])

    # Function to update the animation and plots
    def update_animation_and_plots(theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times, max_torque, max_speed):
        nonlocal ani

        # Update the angular velocity plot
        omega1_line.set_data(times, omega1_opt)
        omega2_line.set_data(times, omega2_opt)
        ax2.set_xlim(0, times[-1])
        
        # Set y-limits to show the speed limits and data
        velocity_range = max(max(np.abs(omega1_opt)), max(np.abs(omega2_opt)), max_speed)
        ax2.set_ylim(-velocity_range * 1.1, velocity_range * 1.1)
        
        # Update the torque plot
        tau1_line.set_data(times[:-1], tau1_opt)
        tau2_line.set_data(times[:-1], tau2_opt)
        ax3.set_xlim(0, times[-1])
        
        # Set y-limits to show the torque limits and data
        torque_range = max(max(np.abs(tau1_opt)), max(np.abs(tau2_opt)), max_torque)
        ax3.set_ylim(-torque_range * 1.1, torque_range * 1.1)
        
        # Update limit lines
        update_limit_lines(max_torque, max_speed, times)

        # Update the robot arm animation
        def update_arm(frame):
            x1, y1, x2, y2 = forward_kinematics(theta1_opt[frame], theta2_opt[frame], l1, l2)
            link1.set_data([0, x1], [0, y1])
            link2.set_data([x1, x2], [y1, y2])
            vline_ax2.set_xdata([times[frame], times[frame]])  # Update vertical line
            vline_ax3.set_xdata([times[frame], times[frame]])  # Update vertical line
            return link1, link2, vline_ax2, vline_ax3

        # Stop the previous animation if it exists
        if ani is not None:
            ani.event_source.stop()

        # Create a new animation
        ani = FuncAnimation(fig, update_arm, frames=len(times), interval=50, blit=True, repeat_delay=1000)

        plt.draw()

    # Function to handle mouse clicks
    def on_click(event):
        nonlocal target_x, target_y, initial_state, obstacles, obstacle_circles, obstacle_radius
        
        # Check if the click is within the animation axes
        if event.inaxes != ax1:
            return
            
        # Right-click adds an obstacle
        if event.button == 3:  # Right mouse button
            obstacle_x, obstacle_y = event.xdata, event.ydata
            current_radius = obs_radius_slider.val
            
            # Add obstacle to list
            obstacles.append((obstacle_x, obstacle_y, current_radius))
            print(f"Obstacle added at: ({obstacle_x:.2f}, {obstacle_y:.2f}) with radius {current_radius:.2f}")
            
            # Add visual representation of the obstacle
            obstacle_circle = Circle((obstacle_x, obstacle_y), current_radius, fill=True, color='green', alpha=0.5)
            ax1.add_patch(obstacle_circle)
            obstacle_circles.append(obstacle_circle)
            
            # If we already have a target, recompute the trajectory with the new obstacle
            if target_x is not None and target_y is not None:
                try:
                    recompute_trajectory()
                except ValueError as e:
                    print(f"Error computing trajectory with new obstacle: {e}")
            
            plt.draw()
            return
            
        # Left-click sets the target position
        target_x, target_y = event.xdata, event.ydata
        print(f"Target position set to: ({target_x:.2f}, {target_y:.2f})")

        # Mark the target position with a red dot
        target_point.set_data([target_x], [target_y])
        
        try:
            # Compute trajectory
            recompute_trajectory()
        except ValueError as e:
            print(f"Error: {e}")

    # Function to recompute the trajectory with current parameters
    def recompute_trajectory():
        nonlocal initial_state
        
        # Get current parameter values
        current_l1 = l1_slider.val
        current_l2 = l2_slider.val
        current_m1 = m1_slider.val
        current_m2 = m2_slider.val
        current_time = time_slider.val
        current_max_torque = torque_slider.val
        current_max_speed = speed_slider.val
        
        # Calculate inverse kinematics
        (theta1_up, theta2_up), (theta1_down, theta2_down) = inverse_kinematics(target_x, target_y, current_l1, current_l2)
        target_state = np.array([theta1_up, theta2_up, 0, 0])  # Target state (elbow-up solution)

        # Optimize trajectory with current parameters and obstacles
        theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times = optimize_trajectory(
            initial_state, target_state, current_l1, current_l2, current_m1, current_m2, 
            num_steps, current_time, current_max_torque, current_max_speed, obstacles
        )

        # Update the animation and plots
        update_animation_and_plots(theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times, 
                                   current_max_torque, current_max_speed)

        # Update the initial state for the next trajectory
        initial_state = np.array([theta1_opt[-1], theta2_opt[-1], omega1_opt[-1], omega2_opt[-1]])

    # Function to handle slider updates
    def update_sliders(val):
        nonlocal l1, l2, m1, m2, total_time, max_torque, max_speed, obstacle_radius
        l1 = l1_slider.val
        l2 = l2_slider.val
        m1 = m1_slider.val
        m2 = m2_slider.val
        total_time = time_slider.val
        max_torque = torque_slider.val
        max_speed = speed_slider.val
        obstacle_radius = obs_radius_slider.val
        
        # Update obstacle sizes if the radius slider changed
        if val is obs_radius_slider:
            for i, (obs_x, obs_y, _) in enumerate(obstacles):
                obstacles[i] = (obs_x, obs_y, obstacle_radius)
                obstacle_circles[i].set_radius(obstacle_radius)
            plt.draw()

        # Re-run the optimization and update the animation
        if target_x is not None and target_y is not None:
            try:
                recompute_trajectory()
            except ValueError as e:
                print(f"Error: {e}")

    # Function to clear all obstacles
    def clear_obstacles(event):
        nonlocal obstacles, obstacle_circles
        
        # Clear obstacle list
        obstacles.clear()
        
        # Remove obstacle visualizations
        for circle in obstacle_circles:
            circle.remove()
        obstacle_circles.clear()
        
        plt.draw()
        
        # Recompute trajectory if we have a target
        if target_x is not None and target_y is not None:
            try:
                recompute_trajectory()
            except ValueError as e:
                print(f"Error: {e}")

    # Add a button to clear obstacles
    ax_clear = plt.axes([slider_x, slider_y_positions[7] - 0.07, slider_width, slider_height])
    clear_button = plt.Button(ax_clear, 'Clear Obstacles')
    clear_button.on_clicked(clear_obstacles)

    # Connect the sliders to the update function
    l1_slider.on_changed(update_sliders)
    l2_slider.on_changed(update_sliders)
    m1_slider.on_changed(update_sliders)
    m2_slider.on_changed(update_sliders)
    time_slider.on_changed(update_sliders)
    torque_slider.on_changed(update_sliders)
    speed_slider.on_changed(update_sliders)
    obs_radius_slider.on_changed(update_sliders)

    # Add text instructions on the figure
    plt.figtext(0.5, 0.97, "Left-click to set target position | Right-click to add obstacle", 
                ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

# Entry point
if __name__ == "__main__":
    main()