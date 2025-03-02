# Multi-Shot Optimization with Collocation

This repository contains a Python implementation of a **multi-shot optimization method** with **collocation** for solving an optimal control problem. The goal is to move a system from an initial state to a desired final state while minimizing the control effort.

---

## Problem Description

We consider a simple dynamical system with the following states:
- **Position (`x`)**: The position of the system.
- **Velocity (`v`)**: The velocity of the system.

The system dynamics are given by:
\[
\frac{dx}{dt} = v, \quad \frac{dv}{dt} = u
\]
where:
- \( u \) is the control input (force).

### Objective
Minimize the control effort, defined as the sum of squared control inputs:
\[
\text{Minimize } J = \sum_{i=1}^{N \cdot \text{num\_shots}} u_i^2
\]

### Constraints
- The system must start at the initial state: \( x(0) = 0 \), \( v(0) = 0 \).
- The system must reach the final state: \( x(T) = 2 \), \( v(T) = 0 \).
- The dynamics must be satisfied at collocation points.

---

## Multi-Shot Optimization with Collocation

The multi-shot method divides the time horizon into multiple segments (or "shots") and optimizes the control inputs and states simultaneously while ensuring continuity between segments. Collocation is used to enforce the system dynamics at specific points within each segment.

---

## Python Implementation

### Key Functions

1. **`dynamics(z, u)`**:
   - Computes the system dynamics:
     \[
     \frac{dx}{dt} = v, \quad \frac{dv}{dt} = u
     \]

   ```python
   def dynamics(z, u):
       x, v = z
       dxdt = v
       dvdt = u
       return np.array([dxdt, dvdt])
    ```

2. **`dynamics(z, u)`**:
 - Computes the objective function (sum of squared control inputs).

   ```python
   def objective(u):
      return np.sum(u**2)  # Minimize the sum of squared control inputs 
   ```

3. **`constraint(u, z0, z1, dt, N, num_shots)`**:

 - Enforces the system dynamics at collocation points and ensures continuity between segments.

   ```python
   def constraint(u, z0, z1, dt, N, num_shots):
    z = np.zeros((2, N * num_shots + 1))  # State vector (position and velocity)
    z[:, 0] = z0  # Initial state

    # Simulate the system using the control inputs
    for shot in range(num_shots):
        for i in range(N):
            idx = shot * N + i
            z[:, idx+1] = z[:, idx] + dynamics(z[:, idx], u[idx]) * dt

    # Collocation constraints: Ensure dynamics are satisfied at segment boundaries
    collocation_constraints = []
    for shot in range(1, num_shots):
        idx = shot * N
        # Dynamics at the boundary
        dz = dynamics(z[:, idx], u[idx])
        # Collocation constraint: Match the dynamics
        collocation_constraints.extend(z[:, idx] - z[:, idx-1] - dz * dt)

    # Final state constraint: position = target_position, velocity = 0
    final_state_constraints = [z[0, -1] - z1[0], z[1, -1] - z1[1]]

    return np.array(collocation_constraints + final_state_constraints)
   ```

4. **`csolve_block_move_multishot(z0, z1, N, num_shots, T)`**:

 - Sets up and solves the optimization problem.

   ```python
   def solve_block_move_multishot(z0, z1, N, num_shots, T):
    dt = T / (N * num_shots)  # Time step size
    times = np.linspace(0, T, N * num_shots + 1)  # Time vector

    # Initial guess for control inputs (force)
    u_guess = np.ones(N * num_shots) * 0.1  # Small initial force

    # Define the constraint dictionary
    cons = {
        'type': 'eq',
        'fun': constraint,
        'args': (z0, z1, dt, N, num_shots)
    }

    # Run the optimization to minimize control effort, subject to constraints
    result = minimize(objective, u_guess, constraints=cons, method='SLSQP')

    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed:", result.message)

    return result.x, times
   ```