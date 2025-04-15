import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from matplotlib.animation import FuncAnimation
from casadi import MX, vertcat, nlpsol, sqrt, fmax, fmin, if_else, DM, sum1

# Forward Kinematics for 3D two-link arm (3 DOF)
def forward_kinematics(theta1, theta2, theta3, l1, l2):
    # theta1: base rotation (z-axis)
    # theta2: shoulder elevation (y-axis)
    # theta3: elbow rotation (x-axis)
    
    # First link (from base to shoulder)
    x1 = l1 * np.cos(theta1) * np.sin(theta2)
    y1 = l1 * np.sin(theta1) * np.sin(theta2)
    z1 = l1 * np.cos(theta2)
    
    # Second link (from shoulder to end effector)
    # Note: theta3 is relative to the first link's orientation
    x2 = x1 + l2 * np.cos(theta1 + theta3) * np.sin(theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta3) * np.sin(theta2)
    z2 = z1 + l2 * np.cos(theta2)
    
    return x1, y1, z1, x2, y2, z2

# Inverse Kinematics for 3D
def inverse_kinematics(x, y, z, l1, l2):
    # Distance from origin to target
    d = np.sqrt(x**2 + y**2 + z**2)
    
    # Check if target is reachable
    if d > l1 + l2 or d < np.abs(l1 - l2):
        raise ValueError("Target position is out of reach")
    
    # Base rotation (theta1)
    theta1 = np.arctan2(y, x)
    
    # Distance in xy plane
    r = np.sqrt(x**2 + y**2)
    
    # Elbow angle (theta3) using law of cosines
    cos_theta3 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta3 = np.arccos(np.clip(cos_theta3, -1, 1))
    
    # Shoulder angle (theta2) - elevation angle
    # First calculate the angle to the target point
    phi = np.arctan2(z, r)
    # Then calculate the angle between the first link and the line to the target
    alpha = np.arctan2(l2 * np.sin(theta3), l1 + l2 * np.cos(theta3))
    theta2 = phi - alpha
    
    return (theta1, theta2, theta3), (theta1, theta2, -theta3)  # Two solutions

# Simplified Robot Dynamics in 3D (single integrator model)
def robot_dynamics(state, control, l1, l2, m1, m2, g=9.81):
    theta1, theta2, theta3 = state[0], state[1], state[2]
    omega1, omega2, omega3 = state[3], state[4], state[5]
    tau1, tau2, tau3 = control[0], control[1], control[2]

    # Simplified dynamics (control directly affects velocities)
    alpha1 = tau1
    alpha2 = tau2
    alpha3 = tau3
    
    return vertcat(omega1, omega2, omega3, alpha1, alpha2, alpha3)

# Optimize Trajectory using SNOPT
def optimize_trajectory(initial_state, target_state, l1, l2, m1, m2, num_steps, max_time, max_torque, max_speed):
    # Time step variable
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
        state_next = state + robot_dynamics(state, control, l1, l2, m1, m2) * dt
        
        g += [theta1[i+1] - state_next[0],
              theta2[i+1] - state_next[1],
              theta3[i+1] - state_next[2],
              omega1[i+1] - state_next[3],
              omega2[i+1] - state_next[4],
              omega3[i+1] - state_next[5]]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

    # Final state constraints (position only)
    g += [theta1[-1], theta2[-1], theta3[-1]]
    lbg += list(target_state[:3])
    ubg += list(target_state[:3])

    # Add end effector position constraints at final state
    x1_final = l1 * MX.cos(theta1[-1]) * MX.sin(theta2[-1])
    y1_final = l1 * MX.sin(theta1[-1]) * MX.sin(theta2[-1])
    z1_final = l1 * MX.cos(theta2[-1])
    
    x2_final = x1_final + l2 * MX.cos(theta1[-1] + theta3[-1]) * MX.sin(theta2[-1])
    y2_final = y1_final + l2 * MX.sin(theta1[-1] + theta3[-1]) * MX.sin(theta2[-1])
    z2_final = z1_final + l2 * MX.cos(theta2[-1])
    
    # Get target position from forward kinematics of target state
    x_target, y_target, z_target = forward_kinematics(target_state[0], target_state[1], target_state[2], l1, l2)[3:]
    
    # Add position constraints with small tolerance
    tolerance = 1e-3
    g += [x2_final - x_target, y2_final - y_target, z2_final - z_target]
    lbg += [-tolerance, -tolerance, -tolerance]
    ubg += [tolerance, tolerance, tolerance]

    # Add intermediate position constraints to guide the trajectory
    for i in range(1, num_steps, num_steps//4):  # Add constraints at 25%, 50%, and 75% of the trajectory
        x1 = l1 * MX.cos(theta1[i]) * MX.sin(theta2[i])
        y1 = l1 * MX.sin(theta1[i]) * MX.sin(theta2[i])
        z1 = l1 * MX.cos(theta2[i])
        
        x2 = x1 + l2 * MX.cos(theta1[i] + theta3[i]) * MX.sin(theta2[i])
        y2 = y1 + l2 * MX.sin(theta1[i] + theta3[i]) * MX.sin(theta2[i])
        z2 = z1 + l2 * MX.cos(theta2[i])
        
        # Add constraints to keep the end effector moving towards the target
        g += [x2 - x_target, y2 - y_target, z2 - z_target]
        lbg += [-np.inf, -np.inf, -np.inf]  # Allow intermediate positions to be anywhere
        ubg += [np.inf, np.inf, np.inf]

    # Variable bounds
    lbx = [0.001]  # dt
    ubx = [max_time/num_steps]
    
    # Joint angle bounds
    lbx += [-2*np.pi]*(num_steps + 1)  # theta1
    ubx += [2*np.pi]*(num_steps + 1)
    lbx += [0]*(num_steps + 1)         # theta2
    ubx += [np.pi]*(num_steps + 1)
    lbx += [-2*np.pi]*(num_steps + 1)  # theta3
    ubx += [2*np.pi]*(num_steps + 1)
    
    # Angular velocity bounds
    lbx += [-max_speed]*(num_steps + 1)
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)
    ubx += [max_speed]*(num_steps + 1)
    lbx += [-max_speed]*(num_steps + 1)
    ubx += [max_speed]*(num_steps + 1)
    
    # Torque bounds
    lbx += [-max_torque]*num_steps
    ubx += [max_torque]*num_steps
    lbx += [-max_torque]*num_steps
    ubx += [max_torque]*num_steps
    lbx += [-max_torque]*num_steps
    ubx += [max_torque]*num_steps

    # Initial guess - use cubic interpolation for smoother trajectory
    x0 = [max_time/num_steps/2]  # dt
    
    # Create time points for interpolation
    t_points = np.linspace(0, 1, num_steps + 1)
    
    # Cubic interpolation for each joint angle
    for i in range(num_steps + 1):
        t = t_points[i]
        # Cubic interpolation coefficients
        a = 2 * (initial_state[0] - target_state[0])
        b = 2 * (initial_state[1] - target_state[1])
        c = 2 * (initial_state[2] - target_state[2])
        
        # Interpolate joint angles
        theta1_interp = initial_state[0] + (target_state[0] - initial_state[0]) * t + a * t * (1 - t)
        theta2_interp = initial_state[1] + (target_state[1] - initial_state[1]) * t + b * t * (1 - t)
        theta3_interp = initial_state[2] + (target_state[2] - initial_state[2]) * t + c * t * (1 - t)
        
        x0 += [theta1_interp]
        x0 += [theta2_interp]
        x0 += [theta3_interp]
        x0 += [0, 0, 0]  # omega1, omega2, omega3
    
    x0 += [0] * (3 * num_steps)  # tau1, tau2, tau3

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
            'Print file': 'snopt_print.out',
            'Summary file': 'snopt_summary.out'
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
    l1 = 5.0
    l2 = 5.0
    m1 = 0.5
    m2 = 0.5
    num_steps = 100  # Reduced for faster optimization
    max_time = 5.0
    max_torque = 10.0
    max_speed = 2.0

    # Create figure with improved layout
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1, 1])
    
    # 3D View (top)
    ax_3d = fig.add_subplot(gs[0:2, :], projection='3d')
    ax_3d.set_xlim(-(l1+l2), (l1+l2))
    ax_3d.set_ylim(-(l1+l2), (l1+l2))
    ax_3d.set_zlim(-(l1+l2), (l1+l2))
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Robot Arm')
    
    # Add reference planes and axes
    xx, yy = np.meshgrid(np.linspace(-(l1+l2), l1+l2, 10), 
                         np.linspace(-(l1+l2), l1+l2, 10))
    zz = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    ax_3d.quiver(0, 0, 0, l1+l2, 0, 0, color='r', arrow_length_ratio=0.1)
    ax_3d.quiver(0, 0, 0, 0, l1+l2, 0, color='g', arrow_length_ratio=0.1)
    ax_3d.quiver(0, 0, 0, 0, 0, l1+l2, color='b', arrow_length_ratio=0.1)
    
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
    target_point = ax_3d.plot([], [], [], 'ro', markersize=8)[0]

    # Velocity and torque plots
    vel_lines = [
        ax_vel.plot([], [], label='Omega1')[0],
        ax_vel.plot([], [], label='Omega2')[0],
        ax_vel.plot([], [], label='Omega3')[0]
    ]
    ax_vel.legend()
    
    torque_lines = [
        ax_torque.plot([], [], label='Tau1')[0],
        ax_torque.plot([], [], label='Tau2')[0],
        ax_torque.plot([], [], label='Tau3')[0]
    ]
    ax_torque.legend()

    # Time indicators
    vline_vel = ax_vel.axvline(x=0, color='r', linestyle='--')
    vline_torque = ax_torque.axvline(x=0, color='r', linestyle='--')

    # Input controls
    ax_x = plt.axes([0.15, 0.05, 0.1, 0.04])
    ax_y = plt.axes([0.3, 0.05, 0.1, 0.04])
    ax_z = plt.axes([0.45, 0.05, 0.1, 0.04])
    ax_submit = plt.axes([0.6, 0.05, 0.1, 0.04])

    x_text = TextBox(ax_x, 'X', initial="0.5")
    y_text = TextBox(ax_y, 'Y', initial="0.5")
    z_text = TextBox(ax_z, 'Z', initial="0.5")
    submit_btn = Button(ax_submit, 'Submit')

    # Sliders
    slider_y = 0.85
    slider_dy = 0.07
    slider_width = 0.15
    slider_height = 0.03
    
    ax_time = plt.axes([0.8, slider_y, slider_width, slider_height])
    ax_l1 = plt.axes([0.8, slider_y-slider_dy, slider_width, slider_height])
    ax_l2 = plt.axes([0.8, slider_y-2*slider_dy, slider_width, slider_height])
    ax_m1 = plt.axes([0.8, slider_y-3*slider_dy, slider_width, slider_height])
    ax_m2 = plt.axes([0.8, slider_y-4*slider_dy, slider_width, slider_height])
    ax_torque_max = plt.axes([0.8, slider_y-5*slider_dy, slider_width, slider_height])
    ax_speed_max = plt.axes([0.8, slider_y-6*slider_dy, slider_width, slider_height])

    time_slider = Slider(ax_time, 'Max Time', 0.1, 10.0, valinit=max_time)
    l1_slider = Slider(ax_l1, 'L1', 0.1, 10.0, valinit=l1)
    l2_slider = Slider(ax_l2, 'L2', 0.1, 10.0, valinit=l2)
    m1_slider = Slider(ax_m1, 'M1', 0.1, 20.0, valinit=m1)
    m2_slider = Slider(ax_m2, 'M2', 0.1, 20.0, valinit=m2)
    torque_slider = Slider(ax_torque_max, 'Max Torque', 1, 20, valinit=max_torque)
    speed_slider = Slider(ax_speed_max, 'Max Speed', 0.1, 5.0, valinit=max_speed)

    # State variables
    target_pos = None
    ani = None
    # Start with arm pointing along x-axis (more natural for 3D)
    initial_state = np.array([0, np.pi/2, 0, 0, 0, 0])  # [theta1, theta2, theta3, omega1, omega2, omega3]

    def update_arm_plot(theta1, theta2, theta3, l1, l2):
        # Base to shoulder
        x0, y0, z0 = 0, 0, 0
        # Shoulder to elbow
        x1 = l1 * np.cos(theta1) * np.sin(theta2)
        y1 = l1 * np.sin(theta1) * np.sin(theta2)
        z1 = l1 * np.cos(theta2)
        # Elbow to wrist
        x2 = x1 + l2 * np.cos(theta1 + theta3) * np.sin(theta2)
        y2 = y1 + l2 * np.sin(theta1 + theta3) * np.sin(theta2)
        z2 = z1 + l2 * np.cos(theta2)
        
        # Update the arm visualization
        link1.set_data([x0, x1], [y0, y1])
        link1.set_3d_properties([z0, z1])
        link2.set_data([x1, x2], [y1, y2])
        link2.set_3d_properties([z1, z2])
        
        # Update the target point visualization
        if target_pos is not None:
            target_point.set_data([target_pos[0]], [target_pos[1]])
            target_point.set_3d_properties([target_pos[2]])
        
        return link1, link2, target_point

    def submit_coords(event):
        nonlocal target_pos, ani, initial_state
        
        try:
            x = float(x_text.text)
            y = float(y_text.text)
            z = float(z_text.text)
            target_pos = (x, y, z)
            target_point.set_data([x], [y])
            target_point.set_3d_properties([z])
            print(f"Target set to ({x:.2f}, {y:.2f}, {z:.2f})")
            compute_trajectory()
        except ValueError:
            print("Invalid coordinates")

    def compute_trajectory():
        nonlocal ani, initial_state
        
        if target_pos is None:
            return
            
        try:
            current_l1 = l1_slider.val
            current_l2 = l2_slider.val
            current_max_time = time_slider.val
            current_max_torque = torque_slider.val
            current_max_speed = speed_slider.val
            
            # Get IK solution for end of second link
            (theta1, theta2, theta3), _ = inverse_kinematics(*target_pos, current_l1, current_l2)
            target_state = np.array([theta1, theta2, theta3, 0, 0, 0])
            
            # Debug output
            print(f"Initial state: {initial_state}")
            print(f"Target position: {target_pos}")
            print(f"Target joint angles: theta1={theta1:.2f}, theta2={theta2:.2f}, theta3={theta3:.2f}")
            
            # Verify forward kinematics matches target
            x1, y1, z1, x2, y2, z2 = forward_kinematics(theta1, theta2, theta3, current_l1, current_l2)
            print(f"Forward kinematics result: ({x2:.2f}, {y2:.2f}, {z2:.2f})")
            
            # Optimize trajectory
            theta1_opt, theta2_opt, theta3_opt, omega1_opt, omega2_opt, omega3_opt, \
            tau1_opt, tau2_opt, tau3_opt, times = optimize_trajectory(
                initial_state, target_state, current_l1, current_l2, 
                m1_slider.val, m2_slider.val, num_steps, current_max_time, 
                current_max_torque, current_max_speed
            )
            
            # Ensure all arrays have proper shape
            times = np.asarray(times).flatten()
            omega1_opt = np.asarray(omega1_opt).flatten()
            omega2_opt = np.asarray(omega2_opt).flatten()
            omega3_opt = np.asarray(omega3_opt).flatten()
            tau1_opt = np.asarray(tau1_opt).flatten()
            tau2_opt = np.asarray(tau2_opt).flatten()
            tau3_opt = np.asarray(tau3_opt).flatten()
            
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
                update_arm_plot(theta1_opt[i], theta2_opt[i], theta3_opt[i], current_l1, current_l2)
                vline_vel.set_xdata([times[i], times[i]])
                vline_torque.set_xdata([times[i], times[i]])
                return link1, link2, target_point, vline_vel, vline_torque
            
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

    # Connect events
    submit_btn.on_clicked(submit_coords)
    
    for slider in [time_slider, l1_slider, l2_slider, m1_slider, m2_slider, torque_slider, speed_slider]:
        slider.on_changed(lambda val: compute_trajectory() if target_pos else None)

    # Initial plot
    update_arm_plot(initial_state[0], initial_state[1], initial_state[2], l1, l2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()