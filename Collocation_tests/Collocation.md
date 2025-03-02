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

3. **`t`**:

   ```python
   
   ```
