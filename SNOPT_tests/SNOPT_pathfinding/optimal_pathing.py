import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, sqrt, fmax, fmin, if_else, DM, sum1, horzsplit, fabs
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.patches import Circle, Rectangle
import time

class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.patch = Rectangle((x - width/2, y - height/2), width, height, color='red', alpha=0.5)
    
    def contains(self, x, y):
        return (abs(x - self.x) < self.width/2) and (abs(y - self.y) < self.height/2)

def optimize_trajectory(initial_state, target_state, obstacles, radius, num_steps, max_time, max_accel, max_speed, boundary):
    dt = MX.sym('dt')
    total_time = dt * num_steps

    # State variables: [x, y, vx, vy]
    x = MX.sym('x', num_steps + 1)
    y = MX.sym('y', num_steps + 1)
    vx = MX.sym('vx', num_steps + 1)
    vy = MX.sym('vy', num_steps + 1)
    
    # Control variables: [ax, ay]
    ax = MX.sym('ax', num_steps)
    ay = MX.sym('ay', num_steps)

    # Objective: minimize time and control effort
    obj = total_time + 0.1 * (sum1(ax**2) + 0.1 * (sum1(ay**2))) / num_steps

    # Constraints
    g = []
    lbg = []
    ubg = []

    # Initial state constraints
    g += [x[0], y[0], vx[0], vy[0]]
    lbg += list(initial_state)
    ubg += list(initial_state)

    # Dynamics constraints (simple double integrator)
    for i in range(num_steps):
        state = vertcat(x[i], y[i], vx[i], vy[i])
        control = vertcat(ax[i], ay[i])
        state_next = state + vertcat(vx[i], vy[i], ax[i], ay[i]) * dt
        
        g += [x[i+1] - state_next[0],
              y[i+1] - state_next[1],
              vx[i+1] - state_next[2],
              vy[i+1] - state_next[3]]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Final state constraints
    g += [x[-1], y[-1], vx[-1], vy[-1]]
    lbg += list(target_state)
    ubg += list(target_state)

    # Obstacle avoidance constraints
    for i in range(num_steps + 1):
        for obs in obstacles:
            # Calculate signed distance to obstacle
            dx = x[i] - obs.x
            dy = y[i] - obs.y
            
            # Use smooth approximations for better numerical stability
            eps = 1e-3  # Small positive number to prevent division by zero
            
            # Smooth absolute value approximation
            abs_dx = sqrt(dx*dx + eps)
            abs_dy = sqrt(dy*dy + eps)
            
            # Smooth maximum approximation
            dist_x = fmax(abs_dx - obs.width/2, eps)
            dist_y = fmax(abs_dy - obs.height/2, eps)
            
            # Smooth distance calculation
            min_dist = sqrt(dist_x*dist_x + dist_y*dist_y + eps)
            
            # Add constraint to stay outside obstacle with safety margin
            g += [min_dist - radius]
            lbg += [0]
            ubg += [np.inf]

    # Boundary constraints with smooth approximations
    for i in range(num_steps + 1):
        # Add small eps to prevent exactly hitting the boundary
        eps_bound = 1e-2
        g += [x[i] - (boundary[0] + radius + eps_bound),  # Left boundary
              boundary[1] - radius - eps_bound - x[i],    # Right boundary
              y[i] - (boundary[2] + radius + eps_bound),  # Bottom boundary
              boundary[3] - radius - eps_bound - y[i]]    # Top boundary
        lbg += [0, 0, 0, 0]
        ubg += [np.inf, np.inf, np.inf, np.inf]

    # Variable bounds
    lbx = [0.001]  # dt lower bound
    ubx = [max_time/num_steps]  # dt upper bound
    
    # Add bounds for other variables with small margins
    margin = 1e-6
    lbx += [boundary[0] + radius + margin] * (num_steps + 1)  # x
    ubx += [boundary[1] - radius - margin] * (num_steps + 1)
    lbx += [boundary[2] + radius + margin] * (num_steps + 1)  # y
    ubx += [boundary[3] - radius - margin] * (num_steps + 1)
    lbx += [-max_speed] * (num_steps + 1)  # vx
    ubx += [max_speed] * (num_steps + 1)
    lbx += [-max_speed] * (num_steps + 1)  # vy
    ubx += [max_speed] * (num_steps + 1)
    lbx += [-max_accel] * num_steps  # ax
    ubx += [max_accel] * num_steps
    lbx += [-max_accel] * num_steps  # ay
    ubx += [max_accel] * num_steps

    # Initial guess - linear interpolation with smooth velocity profile
    x0 = [max_time/num_steps/2]  # dt
    for i in range(num_steps + 1):
        t = i/num_steps
        # Smooth acceleration and deceleration
        s = 0.5 * (1 - np.cos(np.pi * t))  # Smooth interpolation parameter
        x0 += [initial_state[0] * (1-s) + target_state[0] * s]  # x
        x0 += [initial_state[1] * (1-s) + target_state[1] * s]  # y
        
        # Velocity profile that smoothly accelerates and decelerates
        vel_scale = 4 * t * (1-t)  # Peak at t=0.5
        dx = target_state[0] - initial_state[0]
        dy = target_state[1] - initial_state[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            x0 += [vel_scale * max_speed * dx/dist]  # vx
            x0 += [vel_scale * max_speed * dy/dist]  # vy
        else:
            x0 += [0, 0]  # vx, vy
    
    # Initial accelerations
    x0 += [0] * (2 * num_steps)  # ax, ay

    # NLP problem
    nlp = {
        'x': vertcat(dt, x, y, vx, vy, ax, ay),
        'f': obj,
        'g': vertcat(*g)
    }

    # SNOPT options with more robust settings
    opts = {
        'snopt': {
            'Major feasibility tolerance': 1e-6,
            'Major optimality tolerance': 1e-6,
            'Minor feasibility tolerance': 1e-6,
            'Verify level': 3,
            'Scale option': 2,
            'Scale tolerance': 0.9,
            'Linesearch tolerance': 0.95,
            'Major iterations limit': 1000,
            'Minor iterations limit': 500,
            'Iterations limit': 10000,
            'Major step limit': 2.0,
            'Penalty parameter': 1.0,
            'Print file': 0,
            'Summary file': 0,
            'System information': 'No',
            'Print level': 1,
            'Major print level': 0,
            'Minor print level': 0
        },
        'print_time': False,
        'error_on_fail': False
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
    x_opt = np.array(result['x'][1:num_steps+2]).flatten()
    y_opt = np.array(result['x'][num_steps+2:2*(num_steps+1)+1]).flatten()
    vx_opt = np.array(result['x'][2*(num_steps+1)+1:3*(num_steps+1)+1]).flatten()
    vy_opt = np.array(result['x'][3*(num_steps+1)+1:4*(num_steps+1)+1]).flatten()
    ax_opt = np.array(result['x'][4*(num_steps+1)+1:4*(num_steps+1)+1+num_steps]).flatten()
    ay_opt = np.array(result['x'][4*(num_steps+1)+1+num_steps:]).flatten()
    times = np.linspace(0, total_time, num_steps + 1)

    return x_opt, y_opt, vx_opt, vy_opt, ax_opt, ay_opt, times

def main():
    # Simulation parameters
    radius = 0.5
    num_steps = 100
    max_time = 10.0
    max_accel = 2.0
    max_speed = 3.0
    boundary = [-10, 10, -10, 10]  # xmin, xmax, ymin, ymax
    
    # Create obstacles
    obstacles = [
        Obstacle(-5, -5, 3, 3),
        Obstacle(5, 5, 4, 2),
        Obstacle(0, 0, 2, 6),
        Obstacle(3, -4, 5, 2)
    ]

    # Create figure with improved layout
    fig = plt.figure(figsize=(14, 12))
    
    # Create GridSpec with proper spacing
    gs = plt.GridSpec(4, 3, height_ratios=[2, 2, 1, 0.3], figure=fig)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.35, wspace=0.3)
    
    # Main view
    ax_main = fig.add_subplot(gs[0:2, :])
    ax_main.set_xlim(boundary[0], boundary[1])
    ax_main.set_ylim(boundary[2], boundary[3])
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')
    ax_main.set_title('Optimal Path Planning with Obstacle Avoidance')
    ax_main.grid(True)
    ax_main.set_aspect('equal')
    
    # Draw boundary
    boundary_rect = Rectangle((boundary[0], boundary[2]), 
                          boundary[1]-boundary[0], boundary[3]-boundary[2],
                          linewidth=2, edgecolor='black', facecolor='none')
    ax_main.add_patch(boundary_rect)
    
    # Draw obstacles
    for obs in obstacles:
        ax_main.add_patch(obs.patch)
    
    # Velocity plot (bottom-left)
    ax_vel = fig.add_subplot(gs[2, 0:2])
    ax_vel.set_title('Velocity')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('m/s')
    ax_vel.grid(True)
    
    # Acceleration plot (bottom-right)
    ax_accel = fig.add_subplot(gs[2, 2])
    ax_accel.set_title('Acceleration')
    ax_accel.set_xlabel('Time (s)')
    ax_accel.set_ylabel('m/s²')
    ax_accel.grid(True)

    # Create moving circle
    circle = Circle((0, 0), radius, color='blue', alpha=0.7)
    ax_main.add_patch(circle)
    target_circle = Circle((0, 0), radius, color='green', alpha=0.3)
    ax_main.add_patch(target_circle)
    
    # Path line
    path_line, = ax_main.plot([], [], 'b--', alpha=0.5)
    
    # Velocity and acceleration lines
    vel_lines = []
    accel_lines = []
    colors = ['b', 'r']
    labels = ['X', 'Y']
    for i in range(2):
        vel_lines.append(ax_vel.plot([], [], colors[i], label=labels[i])[0])
        accel_lines.append(ax_accel.plot([], [], colors[i], label=labels[i])[0])
    
    ax_vel.legend()
    ax_accel.legend()
    
    # Vertical lines for animation
    vline_vel = ax_vel.axvline(x=0, color='k', linestyle='--')
    vline_accel = ax_accel.axvline(x=0, color='k', linestyle='--')

    # Create control panel at the bottom
    control_ax = fig.add_subplot(gs[3, :])
    control_ax.axis('off')

    # Create sliders with better positioning
    slider_positions = [
        ('Time', max_time, 1, 20),
        ('Radius', radius, 0.1, 2),
        ('Accel', max_accel, 0.1, 5),
        ('Speed', max_speed, 0.1, 5),
        ('Steps', num_steps, 20, 200)
    ]
    
    sliders = []
    for i, (name, val, min_val, max_val) in enumerate(slider_positions):
        ax = fig.add_axes([0.2 + i*0.15, 0.02, 0.1, 0.02])
        slider = Slider(ax, name, min_val, max_val, valinit=val, 
                       valstep=1 if name == 'Steps' else None)
        sliders.append(slider)
    
    time_slider, radius_slider, accel_slider, speed_slider, steps_slider = sliders

    # Add textboxes for coordinates with better positioning
    coord_box_width = 0.08
    coord_box_height = 0.03
    coord_box_y = 0.15
    
    ax_start_x = fig.add_axes([0.1, coord_box_y, coord_box_width, coord_box_height])
    ax_start_y = fig.add_axes([0.2, coord_box_y, coord_box_width, coord_box_height])
    ax_target_x = fig.add_axes([0.35, coord_box_y, coord_box_width, coord_box_height])
    ax_target_y = fig.add_axes([0.45, coord_box_y, coord_box_width, coord_box_height])
    
    start_x_textbox = TextBox(ax_start_x, 'Start X:', initial='-5')
    start_y_textbox = TextBox(ax_start_y, 'Y:', initial='-5')
    target_x_textbox = TextBox(ax_target_x, 'Target X:', initial='5')
    target_y_textbox = TextBox(ax_target_y, 'Y:', initial='5')

    # Add optimization button with better positioning
    ax_button = fig.add_axes([0.7, coord_box_y, 0.2, coord_box_height])
    optimize_button = Button(ax_button, 'Start Optimization')

    # Add instruction text at the top
    fig.text(0.5, 0.95, "Right click to set start position, left click to set target position", 
             ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # State variables
    target_pos = None
    start_pos = None
    ani = None
    initial_state = np.array([-5, -5, 0, 0])  # [x, y, vx, vy]

    def update_circle_position(x, y):
        circle.center = (x, y)
        return circle

    def validate_position(x, y, is_start=True):
        pos_type = "Starting" if is_start else "Target"
        
        # Check if position is inside boundaries
        if (x < boundary[0] + radius or x > boundary[1] - radius or 
            y < boundary[2] + radius or y > boundary[3] - radius):
            print(f"{pos_type} position is outside boundaries")
            return False
            
        # Check if position collides with obstacles
        for obs in obstacles:
            if obs.contains(x, y):
                print(f"{pos_type} position is inside an obstacle")
                return False
        
        return True

    def update_start(text):
        nonlocal start_pos, initial_state
        
        try:
            x = float(start_x_textbox.text)
            y = float(start_y_textbox.text)
            
            if not validate_position(x, y, True):
                return
            
            start_pos = (x, y)
            initial_state = np.array([x, y, 0, 0])
            print(f"Starting position set to: ({x:.2f}, {y:.2f})")
            circle.center = (x, y)
            plt.draw()
        except ValueError:
            print("Please enter valid numbers for starting X and Y coordinates")

    def update_target(text):
        nonlocal target_pos
        
        try:
            x = float(target_x_textbox.text)
            y = float(target_y_textbox.text)
            
            if not validate_position(x, y, False):
                return
            
            target_pos = (x, y)
            print(f"Target position set to: ({x:.2f}, {y:.2f})")
            target_circle.center = (x, y)
            plt.draw()
        except ValueError:
            print("Please enter valid numbers for target X and Y coordinates")

    def start_optimization(event):
        nonlocal ani, initial_state
        
        if target_pos is None:
            print("Please set a target position first")
            return
            
        if start_pos is None:
            print("Please set a starting position first")
            return
            
        try:
            compute_trajectory()
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            import traceback
            traceback.print_exc()

    def compute_trajectory():
        nonlocal ani, initial_state
        
        if target_pos is None or start_pos is None:
            return
            
        try:
            current_radius = radius_slider.val
            current_max_accel = accel_slider.val
            current_max_speed = speed_slider.val
            current_max_time = time_slider.val
            current_num_steps = int(steps_slider.val)
            
            target_state = np.array([target_pos[0], target_pos[1], 0, 0])
            
            # Debug output
            print(f"Initial state: {initial_state}")
            print(f"Target position: {target_pos}")
            
            start_time = time.time()
            
            # Try with current configuration
            x_opt, y_opt, vx_opt, vy_opt, ax_opt, ay_opt, times = optimize_trajectory(
                initial_state, target_state, obstacles, current_radius, 
                current_num_steps, current_max_time, current_max_accel, 
                current_max_speed, boundary
            )
            
            print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
            
            # Update plots
            vel_lines[0].set_data(times, vx_opt)
            vel_lines[1].set_data(times, vy_opt)
            
            accel_lines[0].set_data(times[:-1], ax_opt)
            accel_lines[1].set_data(times[:-1], ay_opt)
            
            path_line.set_data(x_opt, y_opt)
            
            ax_vel.relim()
            ax_vel.autoscale_view()
            ax_accel.relim()
            ax_accel.autoscale_view()
            
            # Animation
            def animate(i):
                update_circle_position(x_opt[i], y_opt[i])
                vline_vel.set_xdata([times[i], times[i]])
                vline_accel.set_xdata([times[i], times[i]])
                return circle, path_line, vline_vel, vline_accel
            
            if ani is not None:
                ani.event_source.stop()
            
            ani = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True)
            plt.draw()
            
            # Update initial state for next trajectory
            initial_state = np.array([
                x_opt[-1], y_opt[-1], vx_opt[-1], vy_opt[-1]
            ])
            
        except Exception as e:
            print(f"Error computing trajectory: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_sliders(val):
        # Update circle radius when slider changes
        circle.set_radius(radius_slider.val)
        if target_pos is not None:
            target_circle.set_radius(radius_slider.val)
        plt.draw()

    # Connect events
    start_x_textbox.on_submit(update_start)
    start_y_textbox.on_submit(update_start)
    target_x_textbox.on_submit(update_target)
    target_y_textbox.on_submit(update_target)
    optimize_button.on_clicked(start_optimization)
    
    for slider in [time_slider, radius_slider, accel_slider, speed_slider, steps_slider]:
        slider.on_changed(update_sliders)

    def on_click(event):
        if event.inaxes != ax_main:
            return

        x, y = event.xdata, event.ydata
        
        if x is None or y is None:
            return
            
        # Right click for start position
        if event.button == 3:  # Right click
            if validate_position(x, y, True):
                nonlocal start_pos, initial_state
                start_pos = (x, y)
                initial_state = np.array([x, y, 0, 0])
                circle.center = (x, y)
                start_x_textbox.set_val(f"{x:.2f}")
                start_y_textbox.set_val(f"{y:.2f}")
                print(f"Starting position set to: ({x:.2f}, {y:.2f})")
                plt.draw()
        
        # Left click for target position
        elif event.button == 1:  # Left click
            if validate_position(x, y, False):
                nonlocal target_pos
                target_pos = (x, y)
                target_circle.center = (x, y)
                target_x_textbox.set_val(f"{x:.2f}")
                target_y_textbox.set_val(f"{y:.2f}")
                print(f"Target position set to: ({x:.2f}, {y:.2f})")
                plt.draw()

    # Connect mouse click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Initial plot
    update_circle_position(initial_state[0], initial_state[1])
    target_circle.center = (5, 5)  # Default target
    target_pos = (5, 5)
    start_pos = (-5, -5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()