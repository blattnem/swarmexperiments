# Collective Motion Simulation

## Overview

This script simulates the collective motion of agents based on specific rules for repulsion, flocking, self-propulsion, and random forces. The simulation can be visualized using Pygame or can be run in a headless mode, storing the data for further analysis.

## Dependencies

- Python
- Numpy
- Matplotlib
- Pygame (optional)

## Parameters

- `N`: Number of agents
- `n_iter`: Number of temporal iterations
- `dt`: Time step
- `rO`: Range of repulsion pressure
- `eps`: Amplitude of repulsion pressure
- `rf`: Range of flocking pressure
- `alpha`: Amplitude of flocking pressure
- `vO`: Target speed
- `mu`: Amplitude of self-propulsion pressure
- `ramp`: Amplitude of random pressure
- `PYGAME`: Toggle for Pygame visualization

## Functions

### `buffering`

Creates a periodic boundary buffer for agents.

**Parameters:**
- `rb`: Range beyond which agents need buffering
- `x, y`: Coordinates of agents
- `x_velocity, y_velocity`: Velocities of agents

**Returns:**
- `nb`: Number of buffered agents
- `x_buffer, y_buffer`: Coordinates of buffered agents
- `x_velocity_buffer, y_velocity_buffer`: Velocities of buffered agents

### `pressure`

Calculates the pressures that agents experience.

**Parameters:**
- `nb`: Number of buffered agents
- `x_buffer, y_buffer`: Coordinates of buffered agents
- `x_velocity_buffer, y_velocity_buffer`: Velocities of buffered agents
- `x, y`: Coordinates of agents
- `x_velocity, y_velocity`: Velocities of agents

**Returns:**
- `fx, fy`: Forces in x and y direction experienced by each agent

## Execution Modes

### Pygame Visualization

If `PYGAME=True`, the script will run with a Pygame window showing the movement of agents.

### Headless Mode

If `PYGAME=False`, the script runs without visualization and prints out iteration numbers. The final state can be plotted using Matplotlib.

## Output

- Pygame mode: Visualization of agents in real-time.
- Headless mode: Plot showing the final state of the agents, along with vectors indicating the velocity.

## Usage

Run the script as is. Modify parameters as needed.
```bash
python swarm_2.py
```


## Future Work

- Extend to 3D space
- Add obstacle interaction
- Optimize for performance

## Authors
Marcel Blattner

## License
This project is licensed under the MIT License.

## Code adapted and from
Charbonneau, Paul, Natural Complexity: A Modeling Handbook (Princeton, NJ, 2017; online edn, Princeton Scholarship Online, 24 May 2018)
