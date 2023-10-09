# Particle Swarm Optimization with Teleporting Particles

## Introduction

This script provides an implementation of the PSO algorithm with a teleportation mechanism. It offers flexibility to set various parameters and visualizes the optimization process in real-time.

The script utilizes the PSO algorithm to minimize a function called `five_minima`, which is represented in a two-dimensional space. This guide outlines the features, classes, methods, and parameters used in the script.

## Dependencies

- numpy
- pygame

## Objective Function

The function to be minimized, `five_minima`, is a 2D function defined as follows:

$$
\text{five\_minima}(x, y) = (x^2 - 10x + y^2 - 10y + 20) \times \sin(x) \times \sin(y) - 2 \times \exp(-((x-2)^2 + (y-2)^2))
$$

## Classes

### Particle Class

`Particle` represents a single particle in the swarm. The particle has several attributes like `position`, `velocity`, `best_position`, `best_score`, and coefficients (`w`, `C1`, `C2`) that influence its behavior.

#### Methods

- `__init__`: Initializes particle with given attributes.
- `teleport`: Teleports the particle to a random position within the search space, biased by a `teleport_bias`.
- `print_parameters`: Prints important parameters of the particle.

### Swarm Class

`Swarm` represents a collection of particles. The swarm class manages the particles and contains methods to solve the optimization problem.

#### Methods

- `__init__`: Initializes the swarm with particles and parameters like the objective function, number of particles, iteration limit, and search space boundaries.
- `solve`: Executes the PSO algorithm, updating particles and finding the global best position.

## Parameters

- `N`: Number of particles.
- `max_iter`: Maximum number of iterations.
- `minx` and `maxx`: Lower and upper boundaries of the search space.
- `damping_factor`: Controls the momentum of the particles.
- `teleport_probability`: Probability of teleportation for each particle.
- `teleport_bias`: Bias towards the lower boundary during teleportation.
- `C1` and `C2`: Coefficients that influence individual and social behavior of particles.

## Execution Flow

1. Initialize the Pygame library and set up the visualization window.
2. Iterate through the maximum number of iterations (`max_iter`).
    - In each iteration:
        - Update each particle's position and velocity.
        - Evaluate the objective function for each particle.
        - Update the global and individual best scores and positions.
        - Possibly teleport the particle.
        - Visualize the swarm using Pygame.

## Visualization

The script includes real-time visualization using Pygame:
- Each particle is represented as a white circle.
- Teleported particles are shown in green.
- The center of mass of the swarm is shown as a red circle.

## Usage

Simply run the script. Modify the parameters as needed.

```python
func =  five_minima
N = 50
max_iter = 3000
minx = np.array([0,0])
maxx = np.array([10, 10])
reinforcing_factor = 0.7
teleport_probability = 0.
teleport_bias = 0.

pso = Swarm(func, N, max_iter, minx, maxx, reinforcing_factor, teleport_probability, teleport_bias, 1.458, 1.458)
pso.solve()
```
or simply do
```bash
python swarm_1.py
```
Tested with python 3.11.x