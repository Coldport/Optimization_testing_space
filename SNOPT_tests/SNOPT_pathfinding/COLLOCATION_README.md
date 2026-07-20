# Direct Collocation (Hermite–Simpson): How `optimal_pathing_collocation.py` Works

This note explains the collocation scheme used in
[`optimal_pathing_collocation.py`](optimal_pathing_collocation.py) — general theory first, then
the exact specialization used in this file — and both the **hard-constraint** and
**soft-constraint** objective variants it solves side by side. It assumes the background from
[`SQP_vs_IPOPT.md`](SQP_vs_IPOPT.md) (shared decision-variable/direct-transcription setup); this
note goes one level deeper into *how the dynamics themselves* are discretized, which is the one
thing that changed relative to both of those files (they use first-order Euler defects; this file
does not).

Reference: Matthew Kelly, *"An Introduction to Trajectory Optimization: How to Do Your Own Direct
Collocation"*, SIAM Review, 2017 — §4, Hermite-Simpson collocation.

---

## 1. General direct collocation (Hermite–Simpson)

A continuous-time optimal control problem has the form

$$
\min_{z(\cdot),\,u(\cdot),\,T} \; \int_0^T L\big(z(t),u(t)\big)\,dt
\quad\text{s.t.}\quad \dot z(t) = f\big(z(t), u(t)\big),\quad z(0)=z_0,\ z(T)=z_T
$$

where $z(t)$ is the state, $u(t)$ the control, and $f$ the system dynamics. Direct collocation
discretizes this into $N$ intervals over mesh points $t_0 < t_1 < \dots < t_N$ (step $h = t_{k+1}-t_k$)
and turns the whole thing into a finite NLP by making **the state and control at every mesh point
decision variables**, and replacing the ODE with algebraic constraints that a polynomial
approximation of $z(t)$ must satisfy.

**Hermite–Simpson** (the scheme Kelly recommends as the best default trade-off of accuracy and
simplicity) additionally introduces an explicit variable at the **midpoint** of every interval,
$z_c^{(k)}, u_c^{(k)}$, and imposes two constraints per interval:

**(a) Interpolation constraint** — the cubic Hermite polynomial built from
$(z_k, f_k)$ and $(z_{k+1}, f_{k+1})$ must pass through the midpoint variable:

$$
z_c^{(k)} = \tfrac12\big(z_k + z_{k+1}\big) + \tfrac{h}{8}\big(f_k - f_{k+1}\big),
\qquad f_k := f(z_k, u_k)
$$

**(b) Collocation (defect) constraint** — Simpson's-rule quadrature of the derivative across the
interval must equal the actual change in state:

$$
z_{k+1} - z_k = \frac{h}{6}\Big(f_k + 4 f_c^{(k)} + f_{k+1}\Big), \qquad f_c^{(k)} := f\big(z_c^{(k)}, u_c^{(k)}\big)
$$

This combination is **4th-order accurate** ($O(h^4)$ global error), versus $O(h)$ for the
explicit-Euler defects used in `optimal_pathing.py` / `optimal_pathing_sqp.py`. The running cost
is likewise integrated with Simpson's rule:

$$
\int_{t_k}^{t_{k+1}} L\,dt \;\approx\; \frac{h}{6}\Big(L_k + 4 L_c^{(k)} + L_{k+1}\Big)
$$

Everything above is generic — it holds for any state/control dimension and any (possibly
nonlinear) $f$. What follows specializes it to this specific 2D path-planning problem.

---

## 2. Specific application: double-integrator point mass

**State and control:**

$$
z = (x,\,y,\,v^x,\,v^y) \in \mathbb{R}^4, \qquad u = (a^x,\,a^y) \in \mathbb{R}^2
$$

**Dynamics** (point mass, acceleration-controlled — linear in $z,u$):

$$
f(z,u) = \big(v^x,\; v^y,\; a^x,\; a^y\big)
$$

Because $f$ is linear, the general interpolation/collocation formulas above expand into a simple
closed-form linear equation **per state component**, exactly as implemented
([`optimal_pathing_collocation.py:75-86`](optimal_pathing_collocation.py#L75-L86)):

**Interpolation** (per interval $k = 0,\dots,N-1$, step $h$):

$$
x_c^{(k)} = \tfrac12(x_k+x_{k+1}) + \tfrac{h}{8}(v^x_k - v^x_{k+1}), \qquad
v^x_c{}^{(k)} = \tfrac12(v^x_k+v^x_{k+1}) + \tfrac{h}{8}(a^x_k - a^x_{k+1})
$$

(and the same pair of equations for $y, v^y$).

**Collocation / defect:**

$$
x_{k+1} - x_k = \frac{h}{6}\big(v^x_k + 4\,v^x_c{}^{(k)} + v^x_{k+1}\big), \qquad
v^x_{k+1} - v^x_k = \frac{h}{6}\big(a^x_k + 4\,a^x_c{}^{(k)} + a^x_{k+1}\big)
$$

(and the same pair for $y, v^y$). This was verified numerically against the analytic
constant-acceleration solution $x(t) = x_0 + v^x_0 t + \tfrac12 a t^2$ — with $a$ held constant, this
is an exact match for a cubic-degree interpolant, and the residual came out at machine precision
($\sim 10^{-16}$).

**Free time**: as in the other two files, $dt$ is itself a decision variable, so $h = dt$ is
uniform across all $N$ intervals and the total trajectory time $T = N \cdot dt$ is optimized subject
to $0 \le T \le T_{max}$.

### Decision vector

$$
z_{\text{full}} = \Big(\,dt,\;\; \underbrace{x,\,y,\,v^x,\,v^y,\,a^x,\,a^y}_{\text{nodes},\ N+1\text{ each}},\;\; \underbrace{x_c,\,y_c,\,v^x_c,\,v^y_c,\,a^x_c,\,a^y_c}_{\text{midpoints},\ N\text{ each}}\,\Big)
$$

Total size $= 1 + 6(N+1) + 6N = 12N + 7$ — roughly double the $6N+5$ of the Euler-based files, since
control is now defined at every node (not just per-interval) *and* there's a full extra
state+control per interval at the midpoint. This is the "separated" Hermite-Simpson form — the
midpoint control $u_c^{(k)}$ is a free variable, not forced to equal $\tfrac12(u_k+u_{k+1})$, which
gives the optimizer more freedom in shaping the control profile within each interval.

### Path constraints at collocation points

Collocation gives the midpoints "for free" as extra, already-computed points along the curve — so
obstacle and boundary constraints are applied at **every node and every midpoint**
($2N{+}1$ points total), not just the nodes:

$$
\text{checkpoints} = \{(x_k, y_k)\}_{k=0}^{N} \;\cup\; \{(x_c^{(k)}, y_c^{(k)})\}_{k=0}^{N-1}
$$

Skipping the midpoints would leave a gap where the Hermite curve could bulge into an obstacle
between two consecutive nodes without any constraint noticing.

---

## 3. Obstacle avoidance: hard vs. soft

Both variants share the exact same dynamics, decision vector, and rectangle-distance formula.
**The only thing that differs is the objective.** This mirrors the distinction already documented
in `SQP_vs_IPOPT.md`.

### Rectangle distance (shared by both variants)

For an axis-aligned obstacle centered at $(o_x,o_y)$ with half-width/half-height $(w,h)$:

$$
d(p; o) = \sqrt{\ \max(|p_x-o_x|-w,\,0)^2 \;+\; \max(|p_y-o_y|-h,\,0)^2\ }
$$

This is $0$ anywhere on or inside the rectangle.

### 3.1 Hard constraints — `soft=False`

**General form**: an inequality constraint the solver is never allowed to violate,
$g(z) \ge 0$.

**Specific**: for every checkpoint $p_k$ and every obstacle $o$, with safety margin
$r_{margin} = 1.2 \cdot r_{radius}$:

$$
d(p_k;\, o) - r_{margin} \;\ge\; 0
$$

**Objective** — nothing but control effort, averaged over every control sample (node **and**
midpoint controls, $n = 2N{+}1$ of them):

$$
f_{\text{hard}}(z) = \frac{1}{2N+1} \sum_{k} \Big(a^x_k{}^2 + a^y_k{}^2\Big)
$$

There is no incentive anywhere in the objective to keep extra distance from an obstacle — the
optimal hard-constrained path will glide along the safety margin exactly, only pulling away when
the effort cost makes that cheaper.

### 3.2 Soft constraints — `soft=True`

**General idea**: rather than only forbidding a violation, add extra terms to the objective that
grow as the trajectory approaches or crosses a constraint boundary, so the optimizer is
*discouraged* from getting close well before it's *forbidden* from crossing. This does **not**
remove the hard constraint above — both are active simultaneously; the soft terms just bias which
of the (still fully obstacle-free) feasible trajectories the solver prefers.

**Specific soft terms**, summed over every checkpoint $p_k$ (nodes + midpoints):

**Goal attraction** — pulls every sample point toward the target, not just the final one:

$$
J_{\text{goal}} = \frac{\lambda_{goal}}{|\text{checkpoints}|} \sum_{k} \big\| p_k - p_{\text{target}} \big\|^2, \qquad \lambda_{goal}=0.5
$$

**Inverse-distance repulsion** — a mild push away from obstacles at range:

$$
J_{\text{rep1}} = \sum_{k}\sum_{o} \frac{w_{obs}}{d(p_k;o) + \epsilon}, \qquad w_{obs}=5.0
$$

**Inverse-square repulsion** — a second, steeper-falloff repulsion term:

$$
J_{\text{rep2}} = \sum_{k}\sum_{o} \frac{\lambda_{obs}}{d_{soft}(p_k;o)^2 + \delta}, \qquad \lambda_{obs}=1.0,\ \delta=10^{-3}
$$

(computed with a slightly floored distance $d_{soft}$ to avoid a divide-by-zero right at the
rectangle boundary).

**Penetration penalty** — a large penalty that only activates if a point is strictly *inside* an
obstacle (manual soft-penalty backstop on top of the hard constraint):

$$
J_{\text{pen}} = \sum_{k}\sum_{o} c \cdot \mathbb{1}[\text{inside}(p_k,o)] \cdot \big(\text{pen}_x^2 + \text{pen}_y^2\big), \qquad c = 3\times10^{6}
$$

**Full soft objective:**

$$
f_{\text{soft}}(z) = \underbrace{\frac{1}{2N+1}\sum_k \big(a^x_k{}^2+a^y_k{}^2\big)}_{\text{control effort}} + J_{\text{goal}} + J_{\text{rep1}} + J_{\text{rep2}} + J_{\text{pen}}
$$

---

## 4. The two full NLPs, side by side

$$
\begin{aligned}
\textbf{Hard:} \quad & \min_z\; f_{\text{hard}}(z) \\
\textbf{Soft:} \quad & \min_z\; f_{\text{soft}}(z) \\[4pt]
\text{s.t. (both, identical):}\quad
& \text{Hermite–Simpson interpolation + collocation defects (§2)} \\
& \text{initial/final state equalities} \\
& 0 \le N\cdot dt \le T_{max} \\
& d(p_k;o) - r_{margin} \ge 0 \quad \forall\, p_k \in \text{checkpoints},\ \forall\, o \\
& \text{boundary box constraints} \quad \forall\, p_k \in \text{checkpoints} \\
& \text{variable bounds: speed} \le v_{max},\ |\text{accel}| \le a_{max},\ \text{position within boundary}
\end{aligned}
$$

Both are solved with IPOPT (`optimal_pathing_collocation.py:188`) — collocation's dynamics
constraints are smooth and linear here (the only non-smoothness anywhere is the $\max(\cdot,0)$
kink in the rectangle-distance formula, present identically in both variants), so there's no
solver-robustness distinction to make in this file the way there is between IPOPT and SQP;
this file isolates the **hard-vs-soft objective** comparison on its own.

---

## 5. Code map

| Concept | Where |
|---|---|
| Node/midpoint decision variables | `optimal_pathing_collocation.py:43-56` |
| Interpolation + collocation defects | `:71-86` |
| Obstacle/boundary constraints at all checkpoints | `:92-111` |
| Control-effort objective (shared) | `:113-118` |
| Soft shaping terms | `:120-143` |
| Variable bounds | `:145-153` |
| Obstacle-aware warm start (reused from the SQP file) | `:155-183` |
| Solve + result extraction | `:185-203` |
| Solve both variants for the demo | `optimize_both()`, `:206-213` |

## 6. Observed behavior

On the repo's four-obstacle demo scene (start `(-8,8)`, target `(8,0)`): both variants converge in
well under a second each. As expected from the objective difference in §3: the **hard** path hugs
the obstacle safety margin almost exactly (measured clearance $\approx 0.6$, matching
$r_{margin}$), while the **soft** path keeps noticeably more clearance ($\approx 1.13$) thanks to
the repulsion terms — visible directly in the GUI as the purple (hard) path cutting corners tighter
than the orange (soft) one.
