import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, sqrt, fmax, fmin, if_else, DM, sum1, horzsplit
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, TextBox, Button
from mpl_toolkits.mplot3d import Axes3D
import time

# Forward Kinematics for 3DOF robot arm
def forward_kinematics(theta1, theta2, theta3, l1, l2, l3):
    # Base point
    x0, y0, z0 = 0, 0, 0
    
    # First joint (base)
    x1 = x0
    y1 = y0
    z1 = z0
    
    # Second joint (shoulder to elbow)
    x2 = x1 + l2 * np.sin(theta2) * np.cos(theta1)
    y2 = y1 + l2 * np.sin(theta2) * np.sin(theta1)
    z2 = z1 + l2 * np.cos(theta2)
    
    # Third joint (elbow to end effector)
    x3 = x2 + l3 * np.sin(theta2 + theta3) * np.cos(theta1)
    y3 = y2 + l3 * np.sin(theta2 + theta3) * np.sin(theta1)
    z3 = z2 + l3 * np.cos(theta2 + theta3)
    
    return x1, y1, z1, x2, y2, z2, x3, y3, z3

# Inverse Kinematics for 3DOF robot arm
def inverse_kinematics(x, y, z, l1, l2, l3):
    # Calculate distance from base to target
    d = np.sqrt(x**2 + y**2 + z**2)
    
    # Check if target is reachable
    if d > l2 + l3 or d < np.abs(l2 - l3):
        raise ValueError("Target position is out of reach")
    
    # Calculate theta1 (azimuth angle)
    theta1 = np.arctan2(y, x)
    
    # Calculate the distance in the x-y plane
    r = np.sqrt(x**2 + y**2)
    
    # Calculate theta2 and theta3 using law of cosines
    # First, find the angle between the target and the z-axis
    phi = np.arctan2(r, z)
    
    # Use law of cosines to find theta3
    cos_theta3 = (l2**2 + l3**2 - d**2) / (2 * l2 * l3)
    theta3 = np.arccos(np.clip(cos_theta3, -1, 1))
    
    # Calculate theta2 using the law of sines
    sin_alpha = (l3 * np.sin(theta3)) / d
    alpha = np.arcsin(np.clip(sin_alpha, -1, 1))
    theta2 = phi - alpha
    
    # Calculate the actual end effector position for both configurations
    x3_up, y3_up, z3_up = forward_kinematics(theta1, theta2, theta3, l1, l2, l3)[-3:]
    x3_down, y3_down, z3_down = forward_kinematics(theta1, theta2, -theta3, l1, l2, l3)[-3:]
    
    # Calculate distances to target for both configurations
    dist_up = np.sqrt((x3_up - x)**2 + (y3_up - y)**2 + (z3_up - z)**2)
    dist_down = np.sqrt((x3_down - x)**2 + (y3_down - y)**2 + (z3_down - z)**2)
    
    # Choose the configuration that gets closer to the target
    if dist_up < dist_down:
        return (theta1, theta2, theta3)
    else:
        return (theta1, theta2, -theta3)

# Robot Dynamics for 3DOF robot arm
def robot_dynamics(state, control, l1, l2, l3, m1, m2, m3, g=9.81):
    theta1 = state[0]
    theta2 = state[1]
    theta3 = state[2]
    omega1 = state[3]
    omega2 = state[4]
    omega3 = state[5]
    tau1 = control[0]
    tau2 = control[1]
    tau3 = control[2]

    # Moments of inertia
    I1 = (1/12) * m1 * l1**2
    I2 = (1/12) * m2 * l2**2
    I3 = (1/12) * m3 * l3**2

    # Center of mass positions
    c1 = l1/2
    c2 = l2/2
    c3 = l3/2

    # Dynamics equations
    # First joint (base)
    alpha1 = (tau1 - tau2 - m2 * l2 * c2 * (omega2**2 * np.sin(theta2) + 2 * omega1 * omega2 * np.sin(theta2)) -
              m3 * l3 * c3 * (omega3**2 * np.sin(theta3) + 2 * omega1 * omega3 * np.sin(theta3))) / (I1 + I2 + I3)
    
    # Second joint (shoulder)
    alpha2 = (tau2 - tau3 + m2 * l2 * c2 * omega1**2 * np.sin(theta2) -
              m3 * l3 * c3 * (omega3**2 * np.sin(theta3) + 2 * omega2 * omega3 * np.sin(theta3))) / (I2 + I3)
    
    # Third joint (elbow)
    alpha3 = (tau3 + m3 * l3 * c3 * (omega1**2 + omega2**2) * np.sin(theta3)) / I3

    return vertcat(omega1, omega2, omega3, alpha1, alpha2, alpha3)

# Optimize Trajectory with SNOPT
def optimize_trajectory(initial_state, target_state, l1, l2, l3, m1, m2, m3, num_steps, max_time, max_torque, max_speed):
    dt = MX.sym('dt')
    total_time = dt * num_steps

    # State variables
    theta1 = MX.sym('theta1', num_steps + 1)
    theta2 = MX.sym('theta2', num_steps + 1)
    theta3 = MX.sym('theta3', num_steps + 1)
    omega1 = MX.sym('omega1', num_steps + 1)
    omega2 = MX.sym('omega2', num_steps + 1)
    omega3 = MX.sym('omega3', num_steps + 1)
    
    # Control variables
    tau1 = MX.sym('tau1', num_steps)
    tau2 = MX.sym('tau2', num_steps)
    tau3 = MX.sym('tau3', num_steps)

    # Objective: minimize time and control effort
    obj = total_time + 0.1 * (sum1(tau1**2) + sum1(tau2**2) + sum1(tau3**2)) / num_steps

    # Constraints
    g = []
    lbg = []
    ubg = []

    # Initial state constraints
    g += [theta1[0], theta2[0], theta3[0], omega1[0], omega2[0], omega3[0]]
    lbg += list(initial_state)
    ubg += list(initial_state)

    # Dynamics constraints
    for i in range(num_steps):
        state = vertcat(theta1[i], theta2[i], theta3[i], omega1[i], omega2[i], omega3[i])
        control = vertcat(tau1[i], tau2[i], tau3[i])
        state_next = state + robot_dynamics(state, control, l1, l2, l3, m1, m2, m3) * dt
        
        g += [theta1[i+1] - state_next[0],
              theta2[i+1] - state_next[1],
              theta3[i+1] - state_next[2],
              omega1[i+1] - state_next[3],
              omega2[i+1] - state_next[4],
              omega3[i+1] - state_next[5]]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

    # Final state constraints
    g += [theta1[-1], theta2[-1], theta3[-1], omega1[-1], omega2[-1], omega3[-1]]
    lbg += list(target_state)
    ubg += list(target_state)

    # Variable bounds
    lbx = [0.001]  # dt lower bound
    ubx = [max_time/num_steps]  # dt upper bound
    
    # Add bounds for other variables
    lbx += [-2*np.pi]*(num_steps + 1)  # theta1
    ubx += [2*np.pi]*(num_steps + 1)
    lbx += [0]*(num_steps + 1)         # theta2
    ubx += [np.pi]*(num_steps + 1)
    lbx += [-2*np.pi]*(num_steps + 1)  # theta3
    ubx += [2*np.pi]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)  # omega1
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)  # omega2
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)  # omega3
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_torque]*num_steps  # tau1
    ubx += [max_torque]*num_steps
    lbx += [-max_torque]*num_steps  # tau2
    ubx += [max_torque]*num_steps
    lbx += [-max_torque]*num_steps  # tau3
    ubx += [max_torque]*num_steps

    # Initial guess
    x0 = [max_time/num_steps/2]  # dt
    for i in range(num_steps + 1):
        t = i/num_steps
        x0 += [initial_state[0] * (1-t) + target_state[0] * t]  # theta1
        x0 += [initial_state[1] * (1-t) + target_state[1] * t]  # theta2
        x0 += [initial_state[2] * (1-t) + target_state[2] * t]  # theta3
        x0 += [0, 0, 0]  # omega1, omega2, omega3
    x0 += [0] * (3 * num_steps)  # torques

    # NLP problem
    nlp = {
        'x': vertcat(dt, theta1, theta2, theta3, omega1, omega2, omega3, tau1, tau2, tau3),
        'f': obj,
        'g': vertcat(*g)
    }

    # SNOPT options
    opts = {
        'snopt': {
            'Major feasibility tolerance': 1e-6,
            'Major optimality tolerance': 1e-6,
            'Minor feasibility tolerance': 1e-6,
            'Verify level': 2,
            'Print file': 0,  # Disable print file
            'Summary file': 0,  # Disable summary file
            'Print level': 1,  # Minimal console output
            'Major print level': 0,  # Disable major iteration output
            'Minor print level': 0,  # Disable minor iteration output
            'New basis file': 0,  # Disable basis file
            'Backup basis file': 0,  # Disable backup basis file
            'Linesearch tolerance': 0.9,
            'Derivative level': 3,
            'Scale option': 1,
            'Crash option': 0,
            'Hessian full memory': 0,
            'Hessian frequency': 999999,
            'Hessian updates': 10,
            'Expand frequency': 10000,
            'Factorization frequency': 100,
            'Scale tolerance': 0.9,
            'Scale print': 0,
            'Solution file': 0,  # Disable solution file
            'Timing level': 0  # Disable timing output
        },
        'print_time': False
    }

    # Create solver with SNOPT
    solver = nlpsol('solver', 'snopt', nlp, opts)

    # Solve the NLP
    result = solver(
        x0=DM(x0),
        lbx=DM(lbx),
        ubx=DM(ubx),
        lbg=DM(lbg),
        ubg=DM(ubg)
    )

    # Extract results
    dt_opt = float(result['x'][0])
    total_time = dt_opt * num_steps
    theta1_opt = np.array(result['x'][1:num_steps+2]).flatten()
    theta2_opt = np.array(result['x'][num_steps+2:2*(num_steps+1)+1]).flatten()
    theta3_opt = np.array(result['x'][2*(num_steps+1)+1:3*(num_steps+1)+1]).flatten()
    omega1_opt = np.array(result['x'][3*(num_steps+1)+1:4*(num_steps+1)+1]).flatten()
    omega2_opt = np.array(result['x'][4*(num_steps+1)+1:5*(num_steps+1)+1]).flatten()
    omega3_opt = np.array(result['x'][5*(num_steps+1)+1:6*(num_steps+1)+1]).flatten()
    tau1_opt = np.array(result['x'][6*(num_steps+1)+1:6*(num_steps+1)+1+num_steps]).flatten()
    tau2_opt = np.array(result['x'][6*(num_steps+1)+1+num_steps:6*(num_steps+1)+1+2*num_steps]).flatten()
    tau3_opt = np.array(result['x'][6*(num_steps+1)+1+2*num_steps:]).flatten()
    times = np.linspace(0, total_time, num_steps + 1)

    return theta1_opt, theta2_opt, theta3_opt, omega1_opt, omega2_opt, omega3_opt, tau1_opt, tau2_opt, tau3_opt, times

def main():
    # Robot parameters
    l1 = 0.0  # First link length set to 0
    l2 = 5.0
    l3 = 5.0
    m1 = 0.1  # Small mass for first link
    m2 = 0.5
    m3 = 0.5
    num_steps = 300
    max_time = 40.0
    max_torque = 100.0
    max_speed = 20.0

    # Create figure with improved layout
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1, 1])
    
    # 3D View (top)
    ax_3d = fig.add_subplot(gs[0:2, :], projection='3d')
    ax_3d.set_xlim(-(l1+l2+l3), (l1+l2+l3))
    ax_3d.set_ylim(-(l1+l2+l3), (l1+l2+l3))
    ax_3d.set_zlim(-(l1+l2+l3), (l1+l2+l3))
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Robot Arm')
    
    # Add reference planes and axes
    xx, yy = np.meshgrid(np.linspace(-(l1+l2+l3), l1+l2+l3, 10), 
                         np.linspace(-(l1+l2+l3), l1+l2+l3, 10))
    zz = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    ax_3d.quiver(0, 0, 0, l1+l2+l3, 0, 0, color='r', arrow_length_ratio=0.1)
    ax_3d.quiver(0, 0, 0, 0, l1+l2+l3, 0, color='g', arrow_length_ratio=0.1)
    ax_3d.quiver(0, 0, 0, 0, 0, l1+l2+l3, color='b', arrow_length_ratio=0.1)
    
    # Velocity plot (bottom-left)
    ax_vel = fig.add_subplot(gs[2, 0])
    ax_vel.set_title('Joint Velocities')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('rad/s')
    
    # Torque plot (bottom-middle)
    ax_torque = fig.add_subplot(gs[2, 1])
    ax_torque.set_title('Joint Torques')
    ax_torque.set_xlabel('Time (s)')
    ax_torque.set_ylabel('Nm')

    # Sliders area (bottom-right)
    ax_sliders = fig.add_subplot(gs[2, 2])
    ax_sliders.axis('off')

    # Arm visualization (3 links)
    link1, = ax_3d.plot([], [], [], 'b-', linewidth=3)
    link2, = ax_3d.plot([], [], [], 'r-', linewidth=3)
    link3, = ax_3d.plot([], [], [], 'g-', linewidth=3)
    target_point = ax_3d.plot([], [], [], 'ro', markersize=8)[0]

    # Velocity and torque lines
    vel_lines = []
    torque_lines = []
    colors = ['b', 'r', 'g']
    labels = ['Joint 1', 'Joint 2', 'Joint 3']
    for i in range(3):
        vel_lines.append(ax_vel.plot([], [], colors[i], label=labels[i])[0])
        torque_lines.append(ax_torque.plot([], [], colors[i], label=labels[i])[0])
    
    ax_vel.legend()
    ax_torque.legend()
    
    # Vertical lines for animation
    vline_vel = ax_vel.axvline(x=0, color='k', linestyle='--')
    vline_torque = ax_torque.axvline(x=0, color='k', linestyle='--')

    # Create sliders
    slider_y_positions = [0.85, 0.78, 0.71, 0.64, 0.57, 0.50, 0.43, 0.36, 0.29]
    slider_width = 0.14
    slider_height = 0.03
    slider_x = 0.82
    
    ax_time = plt.axes([slider_x, slider_y_positions[0], slider_width, slider_height])
    ax_l1 = plt.axes([slider_x, slider_y_positions[1], slider_width, slider_height])
    ax_l2 = plt.axes([slider_x, slider_y_positions[2], slider_width, slider_height])
    ax_l3 = plt.axes([slider_x, slider_y_positions[3], slider_width, slider_height])
    ax_m1 = plt.axes([slider_x, slider_y_positions[4], slider_width, slider_height])
    ax_m2 = plt.axes([slider_x, slider_y_positions[5], slider_width, slider_height])
    ax_m3 = plt.axes([slider_x, slider_y_positions[6], slider_width, slider_height])
    ax_torque_max = plt.axes([slider_x, slider_y_positions[7], slider_width, slider_height])
    ax_speed_max = plt.axes([slider_x, slider_y_positions[8], slider_width, slider_height])

    time_slider = Slider(ax_time, 'Max Time', 1, 20, valinit=max_time)
    l1_slider = Slider(ax_l1, 'L1', 1, 10, valinit=l1)
    l2_slider = Slider(ax_l2, 'L2', 1, 10, valinit=l2)
    l3_slider = Slider(ax_l3, 'L3', 1, 10, valinit=l3)
    m1_slider = Slider(ax_m1, 'M1', 0.1, 2, valinit=m1)
    m2_slider = Slider(ax_m2, 'M2', 0.1, 2, valinit=m2)
    m3_slider = Slider(ax_m3, 'M3', 0.1, 2, valinit=m3)
    torque_slider = Slider(ax_torque_max, 'Max Torque', 1, 20, valinit=max_torque)
    speed_slider = Slider(ax_speed_max, 'Max Speed', 0.1, 5.0, valinit=max_speed)

    # Add textboxes for target coordinates
    ax_x = plt.axes([0.1, 0.05, 0.1, 0.04])
    ax_y = plt.axes([0.25, 0.05, 0.1, 0.04])
    ax_z = plt.axes([0.4, 0.05, 0.1, 0.04])
    
    x_textbox = TextBox(ax_x, 'X:', initial='0')
    y_textbox = TextBox(ax_y, 'Y:', initial='0')
    z_textbox = TextBox(ax_z, 'Z:', initial='5')

    # Add optimization button
    ax_button = plt.axes([0.55, 0.05, 0.15, 0.04])
    optimize_button = Button(ax_button, 'Start Optimization')

    # State variables
    target_pos = None
    ani = None
    # Start with arm pointing along x-axis (more natural for 3D)
    initial_state = np.array([0, np.pi/2, 0, 0, 0, 0])  # [theta1, theta2, theta3, omega1, omega2, omega3]

    def update_arm_plot(theta1, theta2, theta3, l1, l2, l3):
        # Base point
        x0, y0, z0 = 0, 0, 0
        
        # First joint (base)
        x1 = x0
        y1 = y0
        z1 = z0
        
        # Second joint (shoulder to elbow)
        x2 = x1 + l2 * np.sin(theta2) * np.cos(theta1)
        y2 = y1 + l2 * np.sin(theta2) * np.sin(theta1)
        z2 = z1 + l2 * np.cos(theta2)
        
        # Third joint (elbow to end effector)
        x3 = x2 + l3 * np.sin(theta2 + theta3) * np.cos(theta1)
        y3 = y2 + l3 * np.sin(theta2 + theta3) * np.sin(theta1)
        z3 = z2 + l3 * np.cos(theta2 + theta3)
        
        # Update the arm visualization
        link1.set_data([x0, x1], [y0, y1])
        link1.set_3d_properties([z0, z1])
        link2.set_data([x1, x2], [y1, y2])
        link2.set_3d_properties([z1, z2])
        link3.set_data([x2, x3], [y2, y3])
        link3.set_3d_properties([z2, z3])
        
        # Update the target point visualization
        if target_pos is not None:
            target_point.set_data([target_pos[0]], [target_pos[1]])
            target_point.set_3d_properties([target_pos[2]])
        
        return link1, link2, link3, target_point

    def update_target(text):
        nonlocal target_pos
        
        try:
            x = float(x_textbox.text)
            y = float(y_textbox.text)
            z = float(z_textbox.text)
            
            target_pos = (x, y, z)
            print(f"Target position set to: ({x:.2f}, {y:.2f}, {z:.2f})")
            target_point.set_data([x], [y])
            target_point.set_3d_properties([z])
            plt.draw()
        except ValueError:
            print("Please enter valid numbers for X, Y, and Z coordinates")

    def start_optimization(event):
        nonlocal ani, initial_state
        
        if target_pos is None:
            print("Please set a target position first")
            return
            
        try:
            compute_trajectory()
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            import traceback
            traceback.print_exc()

    def compute_trajectory():
        nonlocal ani, initial_state
        
        if target_pos is None:
            return
            
        try:
            current_l1 = l1_slider.val
            current_l2 = l2_slider.val
            current_l3 = l3_slider.val
            current_m1 = m1_slider.val
            current_m2 = m2_slider.val
            current_m3 = m3_slider.val
            current_max_time = time_slider.val
            current_max_torque = torque_slider.val
            current_max_speed = speed_slider.val
            
            # Get IK solution
            theta1, theta2, theta3 = inverse_kinematics(*target_pos, current_l1, current_l2, current_l3)
            target_state = np.array([theta1, theta2, theta3, 0, 0, 0])
            
            # Debug output
            print(f"Initial state: {initial_state}")
            print(f"Target position: {target_pos}")
            print(f"Target joint angles: theta1={theta1:.2f}, theta2={theta2:.2f}, theta3={theta3:.2f}")
            
            # Verify forward kinematics matches target
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = forward_kinematics(theta1, theta2, theta3, current_l1, current_l2, current_l3)
            print(f"Forward kinematics result: ({x3:.2f}, {y3:.2f}, {z3:.2f})")
            
            # Try different configurations and initial conditions
            start_time = time.time()
            max_computation_time = 10.0  # Maximum time to try finding a solution
            best_solution = None
            best_error = float('inf')
            
            # Different initial conditions to try
            initial_conditions = [
                initial_state,
                np.array([0, np.pi/2, 0, 0, 0, 0]),  # Vertical up
                np.array([np.pi/2, np.pi/2, 0, 0, 0, 0]),  # Side
                np.array([np.pi, np.pi/2, 0, 0, 0, 0]),  # Back
                np.array([-np.pi/2, np.pi/2, 0, 0, 0, 0]),  # Other side
            ]
            
            for init_state in initial_conditions:
                if time.time() - start_time > max_computation_time:
                    break
                    
                try:
                    # Try with current configuration
                    theta1_opt, theta2_opt, theta3_opt, omega1_opt, omega2_opt, omega3_opt, \
                    tau1_opt, tau2_opt, tau3_opt, times = optimize_trajectory(
                        init_state, target_state, current_l1, current_l2, current_l3,
                        current_m1, current_m2, current_m3, num_steps, current_max_time, 
                        current_max_torque, current_max_speed
                    )
                    
                    # Calculate final position error
                    final_x, final_y, final_z = forward_kinematics(
                        theta1_opt[-1], theta2_opt[-1], theta3_opt[-1], 
                        current_l1, current_l2, current_l3
                    )[-3:]
                    
                    error = np.sqrt((final_x - target_pos[0])**2 + 
                                  (final_y - target_pos[1])**2 + 
                                  (final_z - target_pos[2])**2)
                    
                    if error < best_error:
                        best_error = error
                        best_solution = (theta1_opt, theta2_opt, theta3_opt, 
                                       omega1_opt, omega2_opt, omega3_opt,
                                       tau1_opt, tau2_opt, tau3_opt, times)
                        
                        # If we found a good solution, break early
                        if error < 0.01:  # 1cm accuracy
                            break
                            
                except Exception as e:
                    print(f"Attempt failed: {str(e)}")
                    continue
            
            if best_solution is None:
                raise ValueError("Could not find a valid solution within time limit")
                
            theta1_opt, theta2_opt, theta3_opt, omega1_opt, omega2_opt, omega3_opt, \
            tau1_opt, tau2_opt, tau3_opt, times = best_solution
            
            print(f"Best solution found with error: {best_error:.4f}")
            
            # Update plots
            vel_lines[0].set_data(times, omega1_opt)
            vel_lines[1].set_data(times, omega2_opt)
            vel_lines[2].set_data(times, omega3_opt)
            
            torque_lines[0].set_data(times[:-1], tau1_opt)
            torque_lines[1].set_data(times[:-1], tau2_opt)
            torque_lines[2].set_data(times[:-1], tau3_opt)
            
            ax_vel.relim()
            ax_vel.autoscale_view()
            ax_torque.relim()
            ax_torque.autoscale_view()
            
            # Animation
            def animate(i):
                update_arm_plot(theta1_opt[i], theta2_opt[i], theta3_opt[i], current_l1, current_l2, current_l3)
                vline_vel.set_xdata([times[i], times[i]])
                vline_torque.set_xdata([times[i], times[i]])
                return link1, link2, link3, target_point, vline_vel, vline_torque
            
            if ani is not None:
                ani.event_source.stop()
            
            ani = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True)
            plt.draw()
            
            # Update initial state for next trajectory
            initial_state = np.array([
                theta1_opt[-1], theta2_opt[-1], theta3_opt[-1],
                omega1_opt[-1], omega2_opt[-1], omega3_opt[-1]
            ])
            
        except Exception as e:
            print(f"Error computing trajectory: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_sliders(val):
        if target_pos is not None:
            try:
                compute_trajectory()
            except ValueError as e:
                print(f"Error: {e}")

    # Connect events
    x_textbox.on_submit(update_target)
    y_textbox.on_submit(update_target)
    z_textbox.on_submit(update_target)
    optimize_button.on_clicked(start_optimization)
    
    for slider in [time_slider, l1_slider, l2_slider, l3_slider, m1_slider, m2_slider, m3_slider, torque_slider, speed_slider]:
        slider.on_changed(update_sliders)

    # Initial plot
    update_arm_plot(initial_state[0], initial_state[1], initial_state[2], l1, l2, l3)
    
    plt.figtext(0.5, 0.97, "Enter target coordinates and click 'Start Optimization' to begin", 
                ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 