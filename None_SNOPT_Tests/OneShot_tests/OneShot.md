# One-Shot Optimization for Optimal Control

This repository contains a Python implementation of a **one-shot optimization method** for solving an optimal control problem. The goal is to move a system from an initial state to a desired final state while minimizing the control effort.

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
\text{Minimize } J = \sum_{i=1}^{N} u_i^2
\]

### Constraints
- The system must start at the initial state: \( x(0) = 0 \), \( v(0) = 0 \).
- The system must reach the final state: \( x(1) = 2 \), \( v(1) = 0 \).

## One-Shot Optimization Method

The one-shot method optimizes the control inputs and states simultaneously over the entire time horizon. This approach avoids the complexity of enforcing continuity constraints between segments, making it easier to implement and debug.

### Key Steps
1. **Discretize the Time Horizon**:
   - Divide the time horizon \( t \in [0, 1] \) into \( N \) steps.
   - Define the time vector: \( t = [t_0, t_1, \dots, t_N] \), where \( t_0 = 0 \) and \( t_N = 1 \).

2. **Define the Control Inputs**:
   - The control inputs \( u = [u_1, u_2, \dots, u_N] \) are optimized at each time step.

3. **Simulate the System**:
   - Use a forward Euler method to simulate the system dynamics:
     \[
     x_{i+1} = x_i + v_i \cdot \Delta t
     \]
     \[
     v_{i+1} = v_i + u_i \cdot \Delta t
     \]
   - Here, \( \Delta t = \frac{1}{N} \) is the time step size.

4. **Define the Objective Function**:
   - Minimize the sum of squared control inputs:
     \[
     J = \sum_{i=1}^{N} u_i^2
     \]

5. **Enforce the Final State Constraint**:
   - Ensure the system reaches the desired final state:
     \[
     x_N = 2, \quad v_N = 0
     \]

6. **Run the Optimization**:
   - Use the `scipy.optimize.minimize` function with the `SLSQP` method to solve the optimization problem.

## Python Implementation

The Python code implements the one-shot optimization method as described above. Here are the key components:

### Key Functions
1. **`dynamics(z, u)`**:
   - Computes the system dynamics:
     \[
     \frac{dx}{dt} = v, \quad \frac{dv}{dt} = u
     \]

2. **`objective(u)`**:
   - Computes the objective function (sum of squared control inputs).

3. **`constraint(u, z0, z1, dt, N)`**:
   - Enforces the final state constraint.

4. **`solve_block_move_oneshot()`**:
   - Sets up and solves the optimization problem.

5. **`simulate_system(u_optimized, z0, dt, N)`**:
   - Simulates the system using the optimized control inputs.

### Usage
1. Run the `solve_block_move_oneshot()` function to optimize the control inputs.
2. Use the `simulate_system()` function to simulate the system with the optimized control inputs.
3. Plot the results to visualize the position, velocity, and control inputs over time.

### Example Output
- **Position vs. Time**: The system moves from \( x = 0 \) to \( x = 2 \).
- **Velocity vs. Time**: The velocity starts and ends at \( v = 0 \).
- **Control Input vs. Time**: The control inputs are smooth and minimize the control effort.

## Dependencies
- Python 3.x
- NumPy
- SciPy
- Matplotlib

## How to Run
1. Install the required dependencies:
   ```bash
   pip install numpy scipy matplotlib