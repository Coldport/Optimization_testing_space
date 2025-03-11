from casadi import *

# Define variables
x = MX.sym('x')
y = MX.sym('y')

# Define objective
obj = x**2 + y**2

# Define constraints
g = vertcat(x + y - 1)

# Create NLP problem
nlp = {'x': vertcat(x, y), 'f': obj, 'g': g}

# Create solver
solver = nlpsol('solver', 'snopt', nlp)

# Solve the problem
result = solver(x0=[1.0, 1.0], lbg=[0], ubg=[0])

# Print results
print("Optimal solution:")
print(result['x'])

git config --global user.email "you@example.com"
  git config --global user.name "Your Name"