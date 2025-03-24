import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, sqrt, fmax, fmin, if_else, DM, sum1, horzsplit
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

# Forward Kinematics
def forward_kinematics(theta1, theta2, l1, l2):
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return x1, y1, x2, y2

# Inverse Kinematics
def inverse_kinematics(x, y, l1, l2):
    d = np.sqrt(x**2 + y**2)
    if d > l1 + l2 or d < np.abs(l1 - l2):
        raise ValueError("Target position is out of reach")
    
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = np.arccos(cos_theta2)
    
    theta2_up = theta2
    theta2_down = -theta2
    
    theta1_up = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_up), l1 + l2 * np.cos(theta2_up))
    theta1_down = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2_down), l1 + l2 * np.cos(theta2_down))
    
    return (theta1_up, theta2_up), (theta1_down, theta2_down)

# Robot Dynamics
def robot_dynamics(state, control, l1, l2, m1, m2, g=9.81):
    theta1 = state[0]
    theta2 = state[1]
    omega1 = state[2]
    omega2 = state[3]
    tau1 = control[0]
    tau2 = control[1]

    I1 = (1/12) * m1 * l1**2
    I2 = (1/12) * m2 * l2**2

    alpha1 = (tau1 - tau2 - m2 * l1 * l2 * (omega2**2 * np.sin(theta2) + 2 * omega1 * omega2 * np.sin(theta2))) / (I1 + I2 + m2 * l1**2)
    alpha2 = (tau2 + m2 * l1 * l2 * omega1**2 * np.sin(theta2)) / I2

    return vertcat(omega1, omega2, alpha1, alpha2)

# Distance Calculation
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx*dx + dy*dy
    
    epsilon = 1e-10
    l2_safe = if_else(l2 < epsilon, epsilon, l2)
    
    t_raw = ((px - x1) * dx + (py - y1) * dy) / l2_safe
    t = fmin(1, fmax(0, t_raw))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return sqrt((px - proj_x)**2 + (py - proj_y)**2 + epsilon)

# Optimize Trajectory with Flexible Time
def optimize_trajectory(initial_state, target_state, l1, l2, m1, m2, num_steps, max_time, max_torque, max_speed, obstacles):
    dt = MX.sym('dt')
    total_time = dt * num_steps

    theta1 = MX.sym('theta1', num_steps + 1)
    theta2 = MX.sym('theta2', num_steps + 1)
    omega1 = MX.sym('omega1', num_steps + 1)
    omega2 = MX.sym('omega2', num_steps + 1)
    tau1 = MX.sym('tau1', num_steps)
    tau2 = MX.sym('tau2', num_steps)

    obj = total_time + 0.1 * (sum1(tau1**2) + sum1(tau2**2)) / num_steps

    g = []
    lbg = []
    ubg = []

    # Initial state constraints
    g += [theta1[0], theta2[0], omega1[0], omega2[0]]
    lbg += list(initial_state)
    ubg += list(initial_state)

    # Dynamics constraints - keep as symbolic expressions
    for i in range(num_steps):
        state = vertcat(theta1[i], theta2[i], omega1[i], omega2[i])
        control = vertcat(tau1[i], tau2[i])
        state_next = state + robot_dynamics(state, control, l1, l2, m1, m2) * dt
        
        # Add equality constraints directly
        g += [theta1[i+1] - state_next[0],
              theta2[i+1] - state_next[1],
              omega1[i+1] - state_next[2],
              omega2[i+1] - state_next[3]]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Final state constraints
    g += [theta1[-1], theta2[-1], omega1[-1], omega2[-1]]
    lbg += list(target_state)
    ubg += list(target_state)

    # Obstacle avoidance
    min_distance = 0.2
    for i in range(num_steps + 1):
        x1 = l1 * MX.cos(theta1[i])
        y1 = l1 * MX.sin(theta1[i])
        x2 = x1 + l2 * MX.cos(theta1[i] + theta2[i])
        y2 = y1 + l2 * MX.sin(theta1[i] + theta2[i])
        
        for obs_x, obs_y, obs_rad in obstacles:
            dist_link1 = point_to_segment_distance(obs_x, obs_y, 0, 0, x1, y1) - obs_rad - min_distance
            dist_link2 = point_to_segment_distance(obs_x, obs_y, x1, y1, x2, y2) - obs_rad - min_distance
            
            g += [dist_link1, dist_link2]
            lbg += [0, 0]
            ubg += [np.inf, np.inf]

    # Variable bounds
    lbx = [0.001]  # dt lower bound
    ubx = [max_time/num_steps]  # dt upper bound
    
    # Add bounds for other variables
    lbx += [-np.inf]*(num_steps + 1)  # theta1
    ubx += [np.inf]*(num_steps + 1)
    lbx += [-np.inf]*(num_steps + 1)  # theta2
    ubx += [np.inf]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)  # omega1
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)  # omega2
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_torque]*num_steps  # tau1
    ubx += [max_torque]*num_steps
    lbx += [-max_torque]*num_steps  # tau2
    ubx += [max_torque]*num_steps

    # Initial guess
    x0 = [max_time/num_steps/2]  # dt
    for i in range(num_steps + 1):
        t = i/num_steps
        x0 += [initial_state[0] * (1-t) + target_state[0] * t]  # theta1
        x0 += [initial_state[1] * (1-t) + target_state[1] * t]  # theta2
        x0 += [0, 0]  # omega1, omega2
    x0 += [0] * (2 * num_steps)  # torques

    # NLP problem
    nlp = {
        'x': vertcat(dt, theta1, theta2, omega1, omega2, tau1, tau2),
        'f': obj,
        'g': vertcat(*g)
    }

    # Convert bounds to DM
    lbx_dm = DM(lbx)
    ubx_dm = DM(ubx)
    lbg_dm = DM(lbg)
    ubg_dm = DM(ubg)
    x0_dm = DM(x0)

    solver = nlpsol('solver', 'ipopt', nlp)
    result = solver(
        x0=x0_dm,
        lbx=lbx_dm,
        ubx=ubx_dm,
        lbg=lbg_dm,
        ubg=ubg_dm
    )

    # Extract results
    dt_opt = float(result['x'][0])
    total_time = dt_opt * num_steps
    theta1_opt = np.array(result['x'][1:num_steps+2]).flatten()
    theta2_opt = np.array(result['x'][num_steps+2:2*(num_steps+1)+1]).flatten()
    omega1_opt = np.array(result['x'][2*(num_steps+1)+1:3*(num_steps+1)+1]).flatten()
    omega2_opt = np.array(result['x'][3*(num_steps+1)+1:4*(num_steps+1)+1]).flatten()
    tau1_opt = np.array(result['x'][4*(num_steps+1)+1:4*(num_steps+1)+1+num_steps]).flatten()
    tau2_opt = np.array(result['x'][4*(num_steps+1)+1+num_steps:]).flatten()
    times = np.linspace(0, total_time, num_steps + 1)

    return theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times

def main():
    # Robot parameters
    l1 = 5
    l2 = 5
    m1 = 0.5
    m2 = 0.5
    num_steps = 50
    max_time = 10.0
    max_torque = 10.0
    max_speed = 5.0

    # Initial state
    initial_state = np.array([0, 0, 0, 0])

    # Setup visualization
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax_sliders = fig.add_subplot(gs[:, 1])
    ax_sliders.axis('off')

    ax1.set_xlim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_ylim(-(l1 + l2 + 1), (l1 + l2 + 1))
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title("Robot Arm Animation")

    link1, = ax1.plot([], [], 'b-', linewidth=2)
    link2, = ax1.plot([], [], 'r-', linewidth=2)
    target_point, = ax1.plot([], [], 'ro', markersize=8)

    obstacles = []
    obstacle_circles = []
    obstacle_radius = 0.5

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.set_title("Angular Velocities")
    ax2.grid(True)

    omega1_line, = ax2.plot([], [], label="Omega1 (Link 1)")
    omega2_line, = ax2.plot([], [], label="Omega2 (Link 2)")
    speed_limit_upper, = ax2.plot([], [], 'k--', label="Speed Limit")
    speed_limit_lower, = ax2.plot([], [], 'k--')
    ax2.legend()

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Torque (Nm)")
    ax3.set_title("Torques")
    ax3.grid(True)

    tau1_line, = ax3.plot([], [], label="Tau1 (Link 1)")
    tau2_line, = ax3.plot([], [], label="Tau2 (Link 2)")
    torque_limit_upper, = ax3.plot([], [], 'k--', label="Torque Limit")
    torque_limit_lower, = ax3.plot([], [], 'k--')
    ax3.legend()

    vline_ax2 = ax2.axvline(x=0, color='r', linestyle='--')
    vline_ax3 = ax3.axvline(x=0, color='r', linestyle='--')

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

    time_slider = Slider(ax_time, 'Max Time', 1, 20, valinit=max_time)
    l1_slider = Slider(ax_l1, 'L1', 1, 10, valinit=l1)
    l2_slider = Slider(ax_l2, 'L2', 1, 10, valinit=l2)
    m1_slider = Slider(ax_m1, 'M1', 0.1, 2, valinit=m1)
    m2_slider = Slider(ax_m2, 'M2', 0.1, 2, valinit=m2)
    torque_slider = Slider(ax_torque, 'Max Torque', 1, 20, valinit=max_torque)
    speed_slider = Slider(ax_speed, 'Max Speed', 1, 10, valinit=max_speed)
    obs_radius_slider = Slider(ax_obs_radius, 'Obs Radius', 0.1, 2, valinit=obstacle_radius)

    target_x, target_y = None, None
    ani = None

    def update_limit_lines(max_torque, max_speed, times):
        speed_limit_upper.set_data([times[0], times[-1]], [max_speed, max_speed])
        speed_limit_lower.set_data([times[0], times[-1]], [-max_speed, -max_speed])
        torque_limit_upper.set_data([times[0], times[-1]], [max_torque, max_torque])
        torque_limit_lower.set_data([times[0], times[-1]], [-max_torque, -max_torque])

    def update_animation_and_plots(theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times, max_torque, max_speed):
        nonlocal ani

        omega1_line.set_data(times, omega1_opt)
        omega2_line.set_data(times, omega2_opt)
        ax2.set_xlim(0, times[-1])
        velocity_range = max(max(np.abs(omega1_opt)), max(np.abs(omega2_opt)), max_speed)
        ax2.set_ylim(-velocity_range * 1.1, velocity_range * 1.1)
        
        tau1_line.set_data(times[:-1], tau1_opt)
        tau2_line.set_data(times[:-1], tau2_opt)
        ax3.set_xlim(0, times[-1])
        torque_range = max(max(np.abs(tau1_opt)), max(np.abs(tau2_opt)), max_torque)
        ax3.set_ylim(-torque_range * 1.1, torque_range * 1.1)
        
        update_limit_lines(max_torque, max_speed, times)

        def update_arm(frame):
            x1, y1, x2, y2 = forward_kinematics(theta1_opt[frame], theta2_opt[frame], l1, l2)
            link1.set_data([0, x1], [0, y1])
            link2.set_data([x1, x2], [y1, y2])
            vline_ax2.set_xdata([times[frame], times[frame]])
            vline_ax3.set_xdata([times[frame], times[frame]])
            return link1, link2, vline_ax2, vline_ax3

        if ani is not None:
            ani.event_source.stop()

        ani = FuncAnimation(fig, update_arm, frames=len(times), interval=50, blit=True, repeat_delay=1000)
        plt.draw()

    def on_click(event):
        nonlocal target_x, target_y, obstacle_radius
        
        if event.inaxes != ax1:
            return
            
        if event.button == 3:  # Right mouse button
            obstacle_x, obstacle_y = event.xdata, event.ydata
            current_radius = obs_radius_slider.val
            
            obstacles.append((obstacle_x, obstacle_y, current_radius))
            print(f"Obstacle added at: ({obstacle_x:.2f}, {obstacle_y:.2f}) with radius {current_radius:.2f}")
            
            obstacle_circle = Circle((obstacle_x, obstacle_y), current_radius, fill=True, color='green', alpha=0.5, zorder=10)
            ax1.add_patch(obstacle_circle)
            obstacle_circles.append(obstacle_circle)
            
            if target_x is not None and target_y is not None:
                try:
                    recompute_trajectory()
                except ValueError as e:
                    print(f"Error computing trajectory with new obstacle: {e}")
            
            plt.draw()
            return
            
        target_x, target_y = event.xdata, event.ydata
        print(f"Target position set to: ({target_x:.2f}, {target_y:.2f})")
        target_point.set_data([target_x], [target_y])
        
        try:
            recompute_trajectory()
        except ValueError as e:
            print(f"Error: {e}")

    def recompute_trajectory():
        nonlocal initial_state
        
        current_l1 = l1_slider.val
        current_l2 = l2_slider.val
        current_m1 = m1_slider.val
        current_m2 = m2_slider.val
        current_max_time = time_slider.val
        current_max_torque = torque_slider.val
        current_max_speed = speed_slider.val
        
        (theta1_up, theta2_up), (theta1_down, theta2_down) = inverse_kinematics(target_x, target_y, current_l1, current_l2)
        target_state = np.array([theta1_up, theta2_up, 0, 0])

        theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times = optimize_trajectory(
            initial_state, target_state, current_l1, current_l2, current_m1, current_m2, 
            num_steps, current_max_time, current_max_torque, current_max_speed, obstacles
        )

        update_animation_and_plots(theta1_opt, theta2_opt, omega1_opt, omega2_opt, tau1_opt, tau2_opt, times, 
                                 current_max_torque, current_max_speed)

        initial_state = np.array([theta1_opt[-1], theta2_opt[-1], omega1_opt[-1], omega2_opt[-1]])

    def update_sliders(val):
        if target_x is not None and target_y is not None:
            try:
                recompute_trajectory()
            except ValueError as e:
                print(f"Error: {e}")

    def clear_obstacles(event):
        nonlocal obstacles, obstacle_circles
        
        obstacles.clear()
        for circle in obstacle_circles:
            circle.remove()
        obstacle_circles.clear()
        
        plt.draw()
        if target_x is not None and target_y is not None:
            try:
                recompute_trajectory()
            except ValueError as e:
                print(f"Error: {e}")

    ax_clear = plt.axes([slider_x, slider_y_positions[7] - 0.07, slider_width, slider_height])
    clear_button = plt.Button(ax_clear, 'Clear Obstacles')
    clear_button.on_clicked(clear_obstacles)

    l1_slider.on_changed(update_sliders)
    l2_slider.on_changed(update_sliders)
    m1_slider.on_changed(update_sliders)
    m2_slider.on_changed(update_sliders)
    time_slider.on_changed(update_sliders)
    torque_slider.on_changed(update_sliders)
    speed_slider.on_changed(update_sliders)
    obs_radius_slider.on_changed(update_sliders)

    plt.figtext(0.5, 0.97, "Left-click to set target position | Right-click to add obstacle", 
                ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()