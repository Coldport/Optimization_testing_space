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

    # State and control variables (same as before)
    x = MX.sym('x', num_steps + 1)
    y = MX.sym('y', num_steps + 1)
    vx = MX.sym('vx', num_steps + 1)
    vy = MX.sym('vy', num_steps + 1)
    ax = MX.sym('ax', num_steps)
    ay = MX.sym('ay', num_steps)

    # Base cost: time + control effort
    obj = total_time + 0.1 * (sum1(ax**2) + 0.1 * (sum1(ay**2)) )/ num_steps

    # 1. GOAL ATTRACTION (soft guidance)
    lambda_goal = 0.5
    for k in range(num_steps + 1):
        dist_to_goal = sqrt((x[k] - target_state[0])**2 + (y[k] - target_state[1])**2 + 1e-6)
        obj += lambda_goal * dist_to_goal**2 / num_steps

    # 2. OBSTACLE PENALTY (NEW: Hard penalty for collision)
    penalty_scale = 3e6  # Very large penalty for touching obstacles
    for k in range(num_steps + 1):
        for obs in obstacles:
            # Calculate penetration depth
            dx = x[k] - obs.x
            dy = y[k] - obs.y
            penetration_x = fmax(obs.width/2 - fabs(dx), 0)
            penetration_y = fmax(obs.height/2 - fabs(dy), 0)
            
            # Apply penalty only if penetration occurs
            if_penetrating = if_else(penetration_x > 0, 1, 0) * if_else(penetration_y > 0, 1, 0)
            obj += penalty_scale * if_penetrating * (penetration_x**2 + penetration_y**2)

    # 3. OBSTACLE REPULSION (soft avoidance)
    lambda_obs = 1.0
    for k in range(num_steps + 1):
        for obs in obstacles:
            dx = x[k] - obs.x
            dy = y[k] - obs.y
            dist_x = fmax(fabs(dx) - obs.width/2, 1e-3)
            dist_y = fmax(fabs(dy) - obs.height/2, 1e-3)
            dist_to_obs = sqrt(dist_x**2 + dist_y**2 + 1e-6)
            obj += lambda_obs / (dist_to_obs**2 + 1e-3)

    # CONSTRAINTS (same as before)
    g = []
    lbg = []
    ubg = []

    # Initial state
    g += [x[0], y[0], vx[0], vy[0]]
    lbg += list(initial_state)
    ubg += list(initial_state)

    # Dynamics
    for i in range(num_steps):
        state_next = vertcat(
            x[i] + vx[i] * dt,
            y[i] + vy[i] * dt,
            vx[i] + ax[i] * dt,
            vy[i] + ay[i] * dt
        )
        g += [x[i+1] - state_next[0],
              y[i+1] - state_next[1],
              vx[i+1] - state_next[2],
              vy[i+1] - state_next[3]]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    # Final state
    g += [x[-1], y[-1], vx[-1], vy[-1]]
    lbg += list(target_state)
    ubg += list(target_state)

    # HARD OBSTACLE CONSTRAINTS (NEW: Strict no-penetration)
    safety_margin = radius * 1.2  # Extra buffer
    for i in range(num_steps + 1):
        for obs in obstacles:
            dx = x[i] - obs.x
            dy = y[i] - obs.y
            dist_x = fmax(fabs(dx) - obs.width/2, 0)
            dist_y = fmax(fabs(dy) - obs.height/2, 0)
            min_dist = sqrt(dist_x**2 + dist_y**2 + 1e-6)
            g += [min_dist - safety_margin]
            lbg += [0]
            ubg += [np.inf]

    # Boundary constraints
    for i in range(num_steps + 1):
        eps_bound = 1e-2
        g += [x[i] - (boundary[0] + radius + eps_bound),
              boundary[1] - radius - eps_bound - x[i],
              y[i] - (boundary[2] + radius + eps_bound),
              boundary[3] - radius - eps_bound - y[i]]
        lbg += [0, 0, 0, 0]
        ubg += [np.inf, np.inf, np.inf, np.inf]

    # Variable bounds (same as before)
    lbx = [0.001] + [boundary[0]+radius]*(num_steps+1) + [boundary[2]+radius]*(num_steps+1) + [-max_speed]*(2*(num_steps+1)) + [-max_accel]*(2*num_steps)
    ubx = [max_time/num_steps] + [boundary[1]-radius]*(num_steps+1) + [boundary[3]-radius]*(num_steps+1) + [max_speed]*(2*(num_steps+1)) + [max_accel]*(2*num_steps)

    # Initial guess (same as before)
    x0 = [max_time/num_steps/2]
    for i in range(num_steps + 1):
        t = i/num_steps
        s = 0.5 * (1 - np.cos(np.pi * t))
        x0 += [initial_state[0]*(1-s) + target_state[0]*s, 
               initial_state[1]*(1-s) + target_state[1]*s]
        vel_scale = 4 * t * (1-t)
        dx = target_state[0] - initial_state[0]
        dy = target_state[1] - initial_state[1]
        dist = sqrt(dx**2 + dy**2 + 1e-6)
        x0 += [vel_scale * max_speed * dx/dist, 
               vel_scale * max_speed * dy/dist]
    x0 += [0]*(2*num_steps)

    # Solve NLP (same as before)
    nlp = {'x': vertcat(dt, x, y, vx, vy, ax, ay), 'f': obj, 'g': vertcat(*g)}
    solver = nlpsol('solver', 'snopt', nlp)
    result = solver(x0=DM(x0), lbx=DM(lbx), ubx=DM(ubx), lbg=DM(lbg), ubg=DM(ubg))

    # Extract and return results (same as before)
    return (
        np.array(result['x'][1:num_steps+2]).flatten(),
        np.array(result['x'][num_steps+2:2*num_steps+3]).flatten(),
        np.array(result['x'][2*num_steps+3:3*num_steps+4]).flatten(),
        np.array(result['x'][3*num_steps+4:4*num_steps+5]).flatten(),
        np.array(result['x'][4*num_steps+5:4*num_steps+5+num_steps]).flatten(),
        np.array(result['x'][4*num_steps+5+num_steps:]).flatten(),
        np.linspace(0, float(result['x'][0])*num_steps, num_steps + 1)
    )

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