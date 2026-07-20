# Trajectory Optimization: IPOPT vs. SQP

This note explains how [`optimal_pathing.py`](optimal_pathing.py) (interior-point, via IPOPT) and
[`optimal_pathing_sqp.py`](optimal_pathing_sqp.py) (sequential quadratic programming, via CasADi's
`sqpmethod`) each solve the same 2D point-to-point path-planning problem, and why they behave so
differently in practice despite solving mathematically similar problems.

Both files pose the same underlying question: *find a minimum-effort trajectory from a start state
to a target state, in bounded time, that never enters any rectangular obstacle.* They differ in
**what exactly is handed to the solver** (objective and constraints) and **which algorithm solves
it**, and both of those choices matter a lot for reliability.

---

## 1. Shared problem structure

Both scripts use **direct transcription**: the continuous trajectory is discretized into `N`
(`num_steps`) segments of equal duration `dt`, and the solver decides the state and control at
every grid point simultaneously (as opposed to simulating forward from a policy).

**Decision variables** (`N+1` state points, `N` control points, and the timestep itself):

$$
z = \big(\, dt,\; x_{0:N},\; y_{0:N},\; v^x_{0:N},\; v^y_{0:N},\; a^x_{0:N-1},\; a^y_{0:N-1} \,\big)
$$

In code this is exactly `vertcat(dt, x, y, vx, vy, ax, ay)` — CasADi concatenates it into one flat
vector, **blocked by variable** (all of `x`, then all of `y`, then all of `vx`, ...), not
interleaved by timestep. (This ordering matters — see [§5](#5-a-real-bug-the-initial-guess-ordering).)

**Dynamics** are forward-Euler integration of a double-integrator (point mass with acceleration
control), enforced as equality constraints between consecutive steps:

$$
x_{k+1} = x_k + v^x_k\,dt, \qquad
y_{k+1} = y_k + v^y_k\,dt, \qquad
v^x_{k+1} = v^x_k + a^x_k\,dt, \qquad
v^y_{k+1} = v^y_k + a^y_k\,dt
$$

**Free final time**: `dt` is itself a decision variable (not fixed), so the total trajectory time
$T = N \cdot dt$ is also optimized, subject to $0 \le T \le T_{max}$. This is a standard trick for
letting the solver pick how fast to go, without needing a separate time-scaling reformulation.

**Obstacle geometry**: every obstacle is an axis-aligned rectangle. The distance from a point
$(x,y)$ to the *outside* of a rectangle centered at $(o_x,o_y)$ with half-width/half-height
$(w,h)$ is:

$$
d_{\text{rect}}(x,y) = \sqrt{\ \max(|x-o_x|-w,\,0)^2 \;+\; \max(|y-o_y|-h,\,0)^2\ }
$$

This is $0$ for *any* point on or inside the rectangle (it doesn't measure penetration depth), and
it has a kink (non-smooth corner) exactly on the rectangle's boundary and along its edge
extensions — both solvers differentiate straight through this kink via CasADi's automatic
differentiation (subgradient at the kink), so neither one gets a "smoother" version of this
function. What differs is how robust each algorithm is to that non-smoothness (see §4).

Both files build the NLP in the general form:

$$
\min_{z} \; f(z) \quad \text{s.t.} \quad g_{lb} \le g(z) \le g_{ub}, \qquad z_{lb} \le z \le z_{ub}
$$

and hand it to `casadi.nlpsol(...)` — only the objective $f$, the solver plugin, and the solver
options differ.

---

## 2. `optimal_pathing.py` — IPOPT

### Objective

The IPOPT version's cost function is a *sum of soft shaping terms* — nothing here is strictly
required for a valid trajectory; each term biases the solver toward more natural-looking paths and
away from constraint boundaries:

$$
f(z) \;=\; \underbrace{\frac{1}{N}\sum_{k} \big(a^x_k{}^2 + a^y_k{}^2\big)}_{\text{control effort}}
\;+\; \underbrace{\sum_{k} \frac{w_{obs}}{d_{\text{rect},k} + \epsilon}}_{\text{inverse-distance repulsion}}
\;+\; \underbrace{\frac{\lambda_{goal}}{N}\sum_{k} \big\|p_k - p_{target}\big\|^2}_{\text{goal attraction}}
\;+\; \underbrace{\sum_{k} c \cdot \mathbb{1}[\text{penetrating}] \cdot \|\text{penetration}_k\|^2}_{\text{penalty for touching an obstacle}}
\;+\; \underbrace{\sum_{k} \frac{\lambda_{obs}}{d_{\text{rect},k}^2 + \epsilon}}_{\text{inverse-square repulsion}}
$$

- **Control effort**: standard "smooth, low-energy motion" term.
- **Goal attraction**: pulls *every* point on the trajectory toward the target, not just the last
  one — a soft-guidance term to help the solver converge toward a sensible shape.
- **Two repulsion terms** (inverse-distance and inverse-square): push the path away from obstacles
  *before* it gets close, giving smoother, better-clearance paths than a bare hard constraint would.
- **Penetration penalty**: an extra large penalty specifically when a point is *inside* an
  obstacle's box — effectively a manual soft-penalty backstop.

### Constraints

Despite all that soft shaping, **the same hard, non-negotiable constraints as the SQP file** are
also present:

$$
d_{\text{rect},k} - r_{margin} \ge 0 \quad \forall k,\ \forall\text{obstacle}
$$

(plus dynamics equalities, initial/final state equalities, boundary inequalities — identical
structure to the SQP file, [§3](#3-optimal_pathing_sqppy--sequential-quadratic-programming-sqp)).

So this file isn't "soft instead of hard" obstacle avoidance — it's **hard constraints, plus soft
cost terms layered on top** to shape *which* feasible trajectory the solver prefers and to make the
optimization landscape friendlier to a barrier method. The hard constraint is what guarantees no
penetration; the soft terms are there for path quality and numerical conditioning.

### Solver

```python
solver = nlpsol('solver', 'ipopt', nlp, opts)
```

IPOPT is a **primal-dual interior-point method**. For each inequality constraint $g_i(z) \ge 0$ it
adds a logarithmic barrier term to the objective, controlled by a barrier parameter $\mu > 0$:

$$
\min_z\; f(z) \;-\; \mu \sum_i \ln\big(g_i(z)\big)
$$

and solves a sequence of these barrier subproblems (via Newton's method on the KKT system) while
driving $\mu \to 0$. Because $\ln(g_i(z))$ is $-\infty$ at the constraint boundary, **every
iterate is kept strictly inside the feasible region** — the solver can approach an obstacle's
safety margin but never step across it, even transiently. If a step would be infeasible, IPOPT's
line-search/filter and restoration-phase machinery pull it back automatically.

This is the key reason IPOPT tolerates a bad initial guess: it never needs to jump cleanly across a
constraint kink, it just gradually shrinks the barrier while staying inside the feasible set the
whole time.

---

## 3. `optimal_pathing_sqp.py` — Sequential Quadratic Programming (SQP)

### Objective

Deliberately minimal — no soft shaping at all:

$$
f(z) \;=\; \frac{1}{N}\sum_{k} \big(a^x_k{}^2 + a^y_k{}^2\big)
$$

Just control effort. Obstacle avoidance, the goal, and everything else are **only** enforced as
hard constraints — there's nothing pulling the path away from an obstacle early or biasing it
toward a "nicer" route. In practice this means the optimal path will hug obstacle corners exactly
at the safety margin, since there's no cost penalty for cutting it close.

### Constraints

Identical in kind to IPOPT's hard constraints — dynamics, initial/final state, boundary, and the
same $d_{\text{rect},k} - r_{margin} \ge 0$ obstacle constraint — just with **no accompanying soft
terms** in the objective.

### Solver

```python
solver = nlpsol('solver', 'sqpmethod', nlp, opts)   # qpsol='qpoases'
```

`sqpmethod` is a classic **line-search SQP** (no trust region, no interior-point barrier, no
filter/restoration phase). At each iterate $z_k$ it builds a **quadratic model** of the Lagrangian
and a **linearization** of the constraints, and solves a QP subproblem for the step $d$:

$$
\min_{d}\; \nabla f(z_k)^T d + \tfrac{1}{2} d^T H_k d
\quad \text{s.t.} \quad
g_{lb} - g(z_k) \le \nabla g(z_k)\, d \le g_{ub} - g(z_k),
\quad z_{lb}-z_k \le d \le z_{ub}-z_k
$$

then takes $z_{k+1} = z_k + \alpha_k d_k$ with $\alpha_k$ from an Armijo line search on a merit
function. $H_k$ is the (exact or BFGS-approximate) Hessian of the Lagrangian.

Three extra mechanisms in the solver options exist specifically to compensate for this simpler
algorithm's fragility:

- **`convexify_strategy: 'regularize'`** — $H_k$ isn't guaranteed positive semi-definite away from
  a true local minimum, which would make the QP subproblem ill-posed. This shifts $H_k$'s
  eigenvalues up ($H_k' = H_k + \sigma I$, $\sigma$ chosen so $\lambda_{min}(H_k') \ge$ some floor)
  before it's handed to the QP solver.
- **`elastic_mode: True`** — because linearizing a non-smooth, tightly-packed set of obstacle
  constraints can produce a QP subproblem with *no feasible point at all* (even though the
  original NLP is feasible nearby), elastic mode adds slack variables $s \ge 0$ to the linearized
  constraints and penalizes them in the QP objective by a growing weight $\gamma$:
  $\min_d\; \nabla f^T d + \tfrac12 d^T H_k d + \gamma \sum s$. This guarantees the QP always has a
  solution, at the cost of a temporarily "cheating" (infeasible w.r.t. the linearization) step. If
  $\gamma$ has to grow past `gamma_max` to stay feasible, the solve is abandoned rather than
  grinding indefinitely.
- **`second_order_corrections: True`** — after a step, take an extra corrective QP solve to reduce
  constraint violation the plain linear step introduced (helps convergence near non-smooth
  constraint boundaries specifically).

Unlike IPOPT, **there is no guarantee iterates stay feasible**, and there is no restoration phase —
if the QP subproblems keep failing even with elastic-mode slack, the solve just fails. This file
also checks the *actual* constraint violation of the returned point directly, rather than trusting
`solver.stats()['success']`, because this non-smooth formulation frequently can't reach the tight
KKT tolerance sqpmethod's own convergence check wants, even when the returned trajectory is a
perfectly good, constraint-satisfying answer.

### Why the initial guess is a whole subsystem here

A straight-line initial guess through an obstacle leaves the very first QP subproblem infeasible.
IPOPT shrugs this off (it just walks toward feasibility inside the barrier); SQP has no equivalent
recovery. So `optimal_pathing_sqp.py` includes an obstacle-aware warm-start pipeline that IPOPT's
file doesn't need at all:

1. `_route_around_obstacles` — an A* search on a coarse grid over the boundary (with
   diagonal-corner-cutting explicitly disallowed) finds *some* path from start to target that
   clears every obstacle by the safety margin.
2. `_simplify_path` — greedily shortcuts that grid path down to a handful of waypoints using
   line-of-sight checks (`_segment_intersects_box`), so the guess isn't a jagged staircase.
3. `_point_along_path` samples that waypoint polyline with the same cosine ease-in/out timing
   profile IPOPT's file uses for its straight-line guess, producing per-timestep position and
   velocity-direction guesses.

---

## 4. Key differences

| | `optimal_pathing.py` (IPOPT) | `optimal_pathing_sqp.py` (SQP) |
|---|---|---|
| **Algorithm class** | Primal-dual interior point | Line-search SQP (no trust region/filter/restoration) |
| **Objective** | Control effort + 4 soft shaping terms (goal attraction, 2 repulsion terms, penetration penalty) | Control effort only |
| **Obstacle constraint** | Hard inequality (same formula) **+** soft cost shaping | Hard inequality only, no shaping |
| **Feasibility of iterates** | Stays strictly inside feasible region throughout (barrier) | Not guaranteed; can step outside, relies on elastic-mode slack to stay solvable |
| **Recovery from a bad step** | Built-in filter/restoration phase | Elastic mode (bounded by `gamma_max`) + regularized Hessian; no restoration phase |
| **Sensitivity to warm start** | Low — converges fine from a naive straight-line guess | High — needed a dedicated A*-based obstacle-avoiding warm start to be reliable |
| **Resulting path shape** | Smoother, keeps extra clearance from obstacles (soft repulsion) | Hugs the safety margin exactly — no incentive to keep extra distance |
| **Failure mode** | Rare; converges even on messy problems | Can genuinely fail to find a feasible trajectory and raises `RuntimeError` rather than hang |
| **Typical solve time (this repo's demo scene)** | ~1–2s | ~1–4s (with a correct warm start) |

The most important distinction isn't "hard vs. soft constraints" — both files enforce the exact
same hard no-penetration constraint. It's that **IPOPT's barrier method is structurally robust to
a bad starting point and to constraint non-smoothness**, while **bare SQP is not**, so the SQP file
has to spend real code (`_route_around_obstacles` and friends) doing by hand what IPOPT gets "for
free" from its algorithm.

---

## 5. A real bug: the initial-guess ordering

While preparing this document, a genuine bug was found (and fixed, in both files) in how the
initial guess `x0` was assembled. The decision vector `z` is laid out **blocked by variable**:

```
[ dt,  x_0..x_N,  y_0..y_N,  vx_0..vx_N,  vy_0..vy_N,  ax_0..ax_{N-1},  ay_0..ay_{N-1} ]
```

but the old code built `x0` **interleaved per timestep**:

```
[ dt,  x_0, y_0, vx_0, vy_0,  x_1, y_1, vx_1, vy_1,  ...,  0...0 ]
```

Both lists have the same *length*, so nothing crashed — but every value ended up in the wrong slot
(e.g. `y_0`'s value landed where `x_1` was supposed to go). Decoding the old `x0` the way the
solver actually reads it produced a scrambled, non-physical "path" rather than the intended
straight-line (IPOPT) or obstacle-routed (SQP) guess.

For IPOPT this was largely harmless — interior-point methods are quite tolerant of an arbitrary
(but bounded) starting point. For SQP it silently defeated the entire point of the A*-based
warm-start routing: the QP subproblems were starting from a scrambled guess despite all that extra
machinery. After fixing the ordering (build four separate per-variable lists, then concatenate them
in the correct block order), every scene that previously failed to converge with the SQP solver —
including ones that still failed after generously loosening the iteration and relaxation limits —
now converges in a few seconds. This is strong evidence that most of the SQP file's earlier
fragility traced back to this bug rather than to a fundamental limitation of the algorithm.
