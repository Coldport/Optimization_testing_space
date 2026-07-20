import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, vertcat, nlpsol, sqrt, fmax, if_else, DM, sum1, fabs
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.patches import Circle, Rectangle
import time

from optimal_pathing_sqp import Obstacle, _route_around_obstacles, _point_along_path

# Colors used throughout to tell the two objective variants apart.
SOFT_COLOR = '#e07b39'   # orange - soft-shaped objective (control effort + goal
                         # attraction + obstacle repulsion), same hard obstacle
                         # constraint as everything else in this repo.
HARD_COLOR = '#6a3d9a'   # purple - control effort only, purely hard-constrained.


def _hermite_simpson_trajectory(initial_state, target_state, obstacles, radius, num_steps,
                                 max_time, max_accel, max_speed, boundary, soft):
    """Direct collocation (Hermite-Simpson, per Kelly 2017) trajectory optimization.

    Unlike the Euler-defect scheme in optimal_pathing.py / optimal_pathing_sqp.py, the
    dynamics here are enforced with a cubic Hermite interpolant per interval plus a
    Simpson's-rule integral defect, using an explicit free midpoint state AND control
    per interval (the "separated" Hermite-Simpson form). This is 4th-order accurate
    instead of 1st-order, at the cost of roughly 2x the decision variables.

    `soft` selects which of the two demo objectives to use:
      - soft=True:  control effort + goal attraction + obstacle repulsion (same shaping
        terms as optimal_pathing.py), on top of the same hard obstacle constraint.
      - soft=False: control effort only, purely hard-constrained (same as
        optimal_pathing_sqp.py).
    Both variants share identical dynamics, obstacle constraints, and bounds - the
    objective is the only thing that changes between them.
    """
    eps = 1e-6
    N = num_steps

    dt = MX.sym('dt')
    total_time = dt * N

    # Node variables (N+1 points)
    x = MX.sym('x', N + 1)
    y = MX.sym('y', N + 1)
    vx = MX.sym('vx', N + 1)
    vy = MX.sym('vy', N + 1)
    ax = MX.sym('ax', N + 1)
    ay = MX.sym('ay', N + 1)

    # Free midpoint state + control per interval (N points) - "separated" Hermite-Simpson
    xc = MX.sym('xc', N)
    yc = MX.sym('yc', N)
    vxc = MX.sym('vxc', N)
    vyc = MX.sym('vyc', N)
    axc = MX.sym('axc', N)
    ayc = MX.sym('ayc', N)

    g, lbg, ubg = [], [], []

    # Time budget
    g += [total_time]
    lbg += [0]
    ubg += [max_time]

    # Initial / final state
    g += [x[0], y[0], vx[0], vy[0]]
    lbg += list(initial_state); ubg += list(initial_state)
    g += [x[N], y[N], vx[N], vy[N]]
    lbg += list(target_state); ubg += list(target_state)

    # Hermite-Simpson interpolation + collocation defects, per interval
    h = dt
    for k in range(N):
        # Interpolation: cubic Hermite fit must pass through the midpoint
        g += [xc[k] - (0.5*(x[k]+x[k+1]) + (h/8)*(vx[k]-vx[k+1]))]
        g += [yc[k] - (0.5*(y[k]+y[k+1]) + (h/8)*(vy[k]-vy[k+1]))]
        g += [vxc[k] - (0.5*(vx[k]+vx[k+1]) + (h/8)*(ax[k]-ax[k+1]))]
        g += [vyc[k] - (0.5*(vy[k]+vy[k+1]) + (h/8)*(ay[k]-ay[k+1]))]
        lbg += [0, 0, 0, 0]; ubg += [0, 0, 0, 0]

        # Collocation: Simpson's rule integral of the derivative matches the actual change
        g += [(x[k+1]-x[k]) - (h/6)*(vx[k] + 4*vxc[k] + vx[k+1])]
        g += [(y[k+1]-y[k]) - (h/6)*(vy[k] + 4*vyc[k] + vy[k+1])]
        g += [(vx[k+1]-vx[k]) - (h/6)*(ax[k] + 4*axc[k] + ax[k+1])]
        g += [(vy[k+1]-vy[k]) - (h/6)*(ay[k] + 4*ayc[k] + ay[k+1])]
        lbg += [0, 0, 0, 0]; ubg += [0, 0, 0, 0]

    # Every sample point (nodes + midpoints) gets the same hard obstacle/boundary
    # treatment - collocation gives us the midpoints "for free", and skipping them
    # would leave a gap where the Hermite curve could bulge into an obstacle
    # unnoticed between two consecutive nodes.
    checkpoints = [(x[i], y[i]) for i in range(N + 1)] + [(xc[k], yc[k]) for k in range(N)]

    safety_margin = radius * 1.2
    for px, py in checkpoints:
        for obs in obstacles:
            dx = px - obs.x
            dy = py - obs.y
            dist_x = fmax(fabs(dx) - obs.width/2, 0)
            dist_y = fmax(fabs(dy) - obs.height/2, 0)
            min_dist = sqrt(dist_x**2 + dist_y**2 + 1e-6)
            g += [min_dist - safety_margin]
            lbg += [0]; ubg += [np.inf]

    eps_bound = 1e-2
    for px, py in checkpoints:
        g += [px - (boundary[0] + radius + eps_bound),
              boundary[1] - radius - eps_bound - px,
              py - (boundary[2] + radius + eps_bound),
              boundary[3] - radius - eps_bound - py]
        lbg += [0, 0, 0, 0]; ubg += [np.inf, np.inf, np.inf, np.inf]

    # Objective
    all_ax = vertcat(ax, axc)
    all_ay = vertcat(ay, ayc)
    n_ctrl = 2*N + 1
    control_effort = sum1(all_ax**2 + all_ay**2) / n_ctrl
    obj = control_effort

    if soft:
        lambda_goal = 0.5
        obstacle_weight = 5.0
        lambda_obs = 1.0
        penalty_scale = 3e6
        for px, py in checkpoints:
            obj += lambda_goal * ((px - target_state[0])**2 + (py - target_state[1])**2) / len(checkpoints)
            for obs in obstacles:
                dx = px - obs.x
                dy = py - obs.y
                dx_dist = fmax(fabs(dx) - obs.width/2, 0)
                dy_dist = fmax(fabs(dy) - obs.height/2, 0)
                dist = sqrt(dx_dist**2 + dy_dist**2 + eps)
                obj += obstacle_weight / (dist + eps)

                dist_x = fmax(fabs(dx) - obs.width/2, 1e-3)
                dist_y = fmax(fabs(dy) - obs.height/2, 1e-3)
                dist_soft = sqrt(dist_x**2 + dist_y**2 + 1e-6)
                obj += lambda_obs / (dist_soft**2 + 1e-3)

                penetration_x = fmax(obs.width/2 - fabs(dx), 0)
                penetration_y = fmax(obs.height/2 - fabs(dy), 0)
                penetrating = if_else(penetration_x > 0, 1, 0) * if_else(penetration_y > 0, 1, 0)
                obj += penalty_scale * penetrating * (penetration_x**2 + penetration_y**2)

    # Variable bounds
    lbx = ([0.001] + [boundary[0]+radius]*(N+1) + [boundary[2]+radius]*(N+1)
           + [-max_speed]*(2*(N+1)) + [-max_accel]*(2*(N+1))
           + [boundary[0]+radius]*N + [boundary[2]+radius]*N
           + [-max_speed]*(2*N) + [-max_accel]*(2*N))
    ubx = ([max_time/N] + [boundary[1]-radius]*(N+1) + [boundary[3]-radius]*(N+1)
           + [max_speed]*(2*(N+1)) + [max_accel]*(2*(N+1))
           + [boundary[1]-radius]*N + [boundary[3]-radius]*N
           + [max_speed]*(2*N) + [max_accel]*(2*N))

    # Initial guess: obstacle-aware routed path (reused from optimal_pathing_sqp),
    # sampled at both nodes and midpoints. Built per-variable and concatenated in
    # the same block order as the decision vector below.
    safety_clearance = safety_margin + 0.3
    waypoints = _route_around_obstacles(initial_state[:2], target_state[:2], obstacles, boundary, safety_clearance)
    cum_len = [0.0]
    for i in range(1, len(waypoints)):
        cum_len.append(cum_len[-1] + np.linalg.norm(waypoints[i] - waypoints[i-1]))
    total_len = cum_len[-1] if cum_len[-1] > 1e-9 else 1.0

    def guess_at(s):
        point, direction = _point_along_path(waypoints, cum_len, total_len, s)
        vel_scale = 4 * s * (1-s)
        return point[0], point[1], vel_scale*max_speed*direction[0], vel_scale*max_speed*direction[1]

    x0_x, x0_y, x0_vx, x0_vy, x0_ax, x0_ay = [], [], [], [], [], []
    for i in range(N + 1):
        px, py, pvx, pvy = guess_at(i/N)
        x0_x.append(px); x0_y.append(py); x0_vx.append(pvx); x0_vy.append(pvy)
        x0_ax.append(0.0); x0_ay.append(0.0)

    x0_xc, x0_yc, x0_vxc, x0_vyc, x0_axc, x0_ayc = [], [], [], [], [], []
    for k in range(N):
        px, py, pvx, pvy = guess_at((k+0.5)/N)
        x0_xc.append(px); x0_yc.append(py); x0_vxc.append(pvx); x0_vyc.append(pvy)
        x0_axc.append(0.0); x0_ayc.append(0.0)

    x0 = ([max_time/N/2] + x0_x + x0_y + x0_vx + x0_vy + x0_ax + x0_ay
          + x0_xc + x0_yc + x0_vxc + x0_vyc + x0_axc + x0_ayc)

    z = vertcat(dt, x, y, vx, vy, ax, ay, xc, yc, vxc, vyc, axc, ayc)
    nlp = {'x': z, 'f': obj, 'g': vertcat(*g)}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = nlpsol('solver', 'ipopt', nlp, opts)
    result = solver(x0=DM(x0), lbx=DM(lbx), ubx=DM(ubx), lbg=DM(lbg), ubg=DM(ubg))

    z_opt = np.array(result['x']).flatten()
    o = 1
    x_opt = z_opt[o:o+N+1]; o += N+1
    y_opt = z_opt[o:o+N+1]; o += N+1
    vx_opt = z_opt[o:o+N+1]; o += N+1
    vy_opt = z_opt[o:o+N+1]; o += N+1
    ax_opt = z_opt[o:o+N+1]; o += N+1
    ay_opt = z_opt[o:o+N+1]; o += N+1

    dt_opt = float(z_opt[0])
    times = np.linspace(0, dt_opt*N, N+1)

    return x_opt, y_opt, vx_opt, vy_opt, ax_opt, ay_opt, times


def optimize_both(initial_state, target_state, obstacles, radius, num_steps,
                   max_time, max_accel, max_speed, boundary):
    """Solve both the soft and hard objective variants and return both trajectories."""
    soft = _hermite_simpson_trajectory(initial_state, target_state, obstacles, radius,
                                        num_steps, max_time, max_accel, max_speed, boundary, soft=True)
    hard = _hermite_simpson_trajectory(initial_state, target_state, obstacles, radius,
                                        num_steps, max_time, max_accel, max_speed, boundary, soft=False)
    return soft, hard


def main():
    # Simulation parameters
    radius = 0.5
    num_steps = 20
    max_time = 10.0
    max_accel = 2.0
    max_speed = 3.0
    boundary = [-10, 10, -10, 10]  # xmin, xmax, ymin, ymax

    # Same demo layout as optimal_pathing_sqp.py, for a like-for-like comparison.
    obstacles = [
        Obstacle(-5, -5, 3, 3),
        Obstacle(5, 5, 4, 2),
        Obstacle(0, 0, 2, 6),
        Obstacle(3, -4, 5, 2)
    ]

    fig = plt.figure(figsize=(14, 12))
    gs = plt.GridSpec(4, 3, height_ratios=[2, 2, 1, 0.3], figure=fig)
    gs.update(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.35, wspace=0.3)

    ax_main = fig.add_subplot(gs[0:2, :])
    ax_main.set_xlim(boundary[0], boundary[1])
    ax_main.set_ylim(boundary[2], boundary[3])
    ax_main.set_xlabel('X')
    ax_main.set_ylabel('Y')
    ax_main.set_title('Hermite-Simpson Direct Collocation: Soft-Shaped vs Hard-Only Objective')
    ax_main.grid(True)
    ax_main.set_aspect('equal')

    boundary_rect = Rectangle((boundary[0], boundary[2]),
                          boundary[1]-boundary[0], boundary[3]-boundary[2],
                          linewidth=2, edgecolor='black', facecolor='none')
    ax_main.add_patch(boundary_rect)

    for obs in obstacles:
        ax_main.add_patch(obs.patch)

    # Speed/accel magnitude plots (one line per variant, rather than per-axis,
    # to keep the soft-vs-hard comparison the visual focus).
    ax_vel = fig.add_subplot(gs[2, 0:2])
    ax_vel.set_title('Speed')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('m/s')
    ax_vel.grid(True)

    ax_accel = fig.add_subplot(gs[2, 2])
    ax_accel.set_title('|Acceleration|')
    ax_accel.set_xlabel('Time (s)')
    ax_accel.set_ylabel('m/s²')
    ax_accel.grid(True)

    # Two moving circles - one per objective variant
    circle_soft = Circle((0, 0), radius, color=SOFT_COLOR, alpha=0.8)
    circle_hard = Circle((0, 0), radius, color=HARD_COLOR, alpha=0.8)
    ax_main.add_patch(circle_soft)
    ax_main.add_patch(circle_hard)
    target_circle = Circle((0, 0), radius, color='green', alpha=0.3)
    ax_main.add_patch(target_circle)

    path_line_soft, = ax_main.plot([], [], color=SOFT_COLOR, linewidth=2, label='Soft (shaped objective)')
    path_line_hard, = ax_main.plot([], [], color=HARD_COLOR, linewidth=2, linestyle='--', label='Hard (control effort only)')
    ax_main.legend(loc='upper right')

    speed_line_soft, = ax_vel.plot([], [], color=SOFT_COLOR, label='Soft')
    speed_line_hard, = ax_vel.plot([], [], color=HARD_COLOR, label='Hard')
    ax_vel.legend()

    accel_line_soft, = ax_accel.plot([], [], color=SOFT_COLOR, label='Soft')
    accel_line_hard, = ax_accel.plot([], [], color=HARD_COLOR, label='Hard')
    ax_accel.legend()

    vline_vel = ax_vel.axvline(x=0, color='k', linestyle=':')
    vline_accel = ax_accel.axvline(x=0, color='k', linestyle=':')

    control_ax = fig.add_subplot(gs[3, :])
    control_ax.axis('off')

    slider_positions = [
        ('Time', max_time, 1, 20),
        ('Radius', radius, 0.1, 2),
        ('Accel', max_accel, 0.1, 5),
        ('Speed', max_speed, 0.1, 5),
        ('Steps', num_steps, 10, 40)  # collocation roughly doubles variable count per step vs Euler
    ]

    sliders = []
    for i, (name, val, min_val, max_val) in enumerate(slider_positions):
        ax = fig.add_axes([0.2 + i*0.15, 0.02, 0.1, 0.02])
        slider = Slider(ax, name, min_val, max_val, valinit=val,
                       valstep=1 if name == 'Steps' else None)
        sliders.append(slider)

    time_slider, radius_slider, accel_slider, speed_slider, steps_slider = sliders

    coord_box_width = 0.08
    coord_box_height = 0.03
    coord_box_y = 0.15

    ax_start_x = fig.add_axes([0.1, coord_box_y, coord_box_width, coord_box_height])
    ax_start_y = fig.add_axes([0.2, coord_box_y, coord_box_width, coord_box_height])
    ax_target_x = fig.add_axes([0.35, coord_box_y, coord_box_width, coord_box_height])
    ax_target_y = fig.add_axes([0.45, coord_box_y, coord_box_width, coord_box_height])

    start_x_textbox = TextBox(ax_start_x, 'Start X:', initial='-8')
    start_y_textbox = TextBox(ax_start_y, 'Y:', initial='8')
    target_x_textbox = TextBox(ax_target_x, 'Target X:', initial='8')
    target_y_textbox = TextBox(ax_target_y, 'Y:', initial='0')

    ax_button = fig.add_axes([0.7, coord_box_y, 0.2, coord_box_height])
    optimize_button = Button(ax_button, 'Start Optimization')

    fig.text(0.5, 0.95, "Right click to set start position, left click to set target position",
             ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    target_pos = None
    start_pos = None
    ani = None
    initial_state = np.array([-8, 8, 0, 0])  # [x, y, vx, vy]

    def update_circle_positions(sx, sy, hx, hy):
        circle_soft.center = (sx, sy)
        circle_hard.center = (hx, hy)

    def validate_position(x, y, is_start=True):
        pos_type = "Starting" if is_start else "Target"
        if (x < boundary[0] + radius or x > boundary[1] - radius or
            y < boundary[2] + radius or y > boundary[3] - radius):
            print(f"{pos_type} position is outside boundaries")
            return False
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
            circle_soft.center = (x, y)
            circle_hard.center = (x, y)
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

            print(f"Initial state: {initial_state}")
            print(f"Target position: {target_pos}")

            start_time = time.time()

            optimize_button.label.set_text('Optimizing...')
            fig.canvas.draw()
            fig.canvas.flush_events()

            soft, hard = optimize_both(
                initial_state, target_state, obstacles, current_radius,
                current_num_steps, current_max_time, current_max_accel,
                current_max_speed, boundary
            )
            x_s, y_s, vx_s, vy_s, ax_s, ay_s, t_s = soft
            x_h, y_h, vx_h, vy_h, ax_h, ay_h, t_h = hard

            optimize_button.label.set_text('Start Optimization')
            print(f"Optimization completed in {time.time() - start_time:.2f} seconds")

            speed_s = np.hypot(vx_s, vy_s)
            speed_h = np.hypot(vx_h, vy_h)
            accel_mag_s = np.hypot(ax_s, ay_s)
            accel_mag_h = np.hypot(ax_h, ay_h)

            speed_line_soft.set_data(t_s, speed_s)
            speed_line_hard.set_data(t_h, speed_h)
            accel_line_soft.set_data(t_s, accel_mag_s)
            accel_line_hard.set_data(t_h, accel_mag_h)

            path_line_soft.set_data(x_s, y_s)
            path_line_hard.set_data(x_h, y_h)

            ax_vel.relim(); ax_vel.autoscale_view()
            ax_accel.relim(); ax_accel.autoscale_view()

            def animate(i):
                update_circle_positions(x_s[i], y_s[i], x_h[i], y_h[i])
                vline_vel.set_xdata([t_s[i], t_s[i]])
                vline_accel.set_xdata([t_s[i], t_s[i]])
                return circle_soft, circle_hard, path_line_soft, path_line_hard, vline_vel, vline_accel

            if ani is not None:
                ani.event_source.stop()

            ani = FuncAnimation(fig, animate, frames=len(t_s), interval=50, blit=True)
            plt.draw()

            initial_state = np.array([x_s[-1], y_s[-1], vx_s[-1], vy_s[-1]])

        except Exception as e:
            optimize_button.label.set_text('Start Optimization')
            print(f"Error computing trajectory: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_sliders(val):
        circle_soft.set_radius(radius_slider.val)
        circle_hard.set_radius(radius_slider.val)
        if target_pos is not None:
            target_circle.set_radius(radius_slider.val)
        plt.draw()

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

        if event.button == 3:  # Right click - start
            if validate_position(x, y, True):
                nonlocal start_pos, initial_state
                start_pos = (x, y)
                initial_state = np.array([x, y, 0, 0])
                circle_soft.center = (x, y)
                circle_hard.center = (x, y)
                start_x_textbox.set_val(f"{x:.2f}")
                start_y_textbox.set_val(f"{y:.2f}")
                print(f"Starting position set to: ({x:.2f}, {y:.2f})")
                plt.draw()

        elif event.button == 1:  # Left click - target
            if validate_position(x, y, False):
                nonlocal target_pos
                target_pos = (x, y)
                target_circle.center = (x, y)
                target_x_textbox.set_val(f"{x:.2f}")
                target_y_textbox.set_val(f"{y:.2f}")
                print(f"Target position set to: ({x:.2f}, {y:.2f})")
                plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    update_circle_positions(initial_state[0], initial_state[1], initial_state[0], initial_state[1])
    target_circle.center = (8, 0)
    target_pos = (8, 0)
    start_pos = (-8, 8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
