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

## Collocation

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

3. **`enforce_constraints(control_inputs, initial_state, target_state, dt, num_steps_per_shot, num_shots)`**:

 - Enforces the system dynamics at collocation points and ensures continuity between segments.

   ```python
   def enforce_constraints(control_inputs, initial_state, target_state, dt, num_steps_per_shot, num_shots):
    state = np.zeros((2, num_steps_per_shot * num_shots + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the control inputs
    for shot in range(num_shots):
        for i in range(num_steps_per_shot):
            idx = shot * num_steps_per_shot + i
            state[:, idx+1] = state[:, idx] + system_dynamics(state[:, idx], control_inputs[idx]) * dt

    # Collocation constraints: Ensure dynamics are satisfied at segment boundaries
    collocation_constraints = []
    for shot in range(1, num_shots):
        idx = shot * num_steps_per_shot
        # Dynamics at the boundary
        dz = system_dynamics(state[:, idx], control_inputs[idx])
        # Collocation constraint: Match the dynamics
        collocation_constraints.extend(state[:, idx] - state[:, idx-1] - dz * dt)

    # Final state constraint: position = target_position, velocity = 0
    final_state_constraints = [state[0, -1] - target_state[0], state[1, -1] - target_state[1]]

    return np.array(collocation_constraints + final_state_constraints)

   ```



4. **`simulate_system(u_optimized, z0, dt, N, num_shots)`** :

 - Simulates the system using the optimized control inputs.

   ```python
   def simulate_trajectory(optimized_controls, initial_state, dt, num_steps_per_shot, num_shots):
    state = np.zeros((2, num_steps_per_shot * num_shots + 1))  # State vector (position and velocity)
    state[:, 0] = initial_state  # Initial state

    # Simulate the system using the optimized control inputs
    for shot in range(num_shots):
        for i in range(num_steps_per_shot):
            idx = shot * num_steps_per_shot + i
            state[:, idx+1] = state[:, idx] + system_dynamics(state[:, idx], optimized_controls[idx]) * dt

    return state


   ```