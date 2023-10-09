# Gene Regulatory Network-based Navigation

## Overview

The script demonstrates a simple 2D grid navigation problem solved through a Gene Regulatory Network (GRN) trained using reinforcement learning (RL). The GRN simulates genes whose states (Q-values) change based on RL principles to guide an agent through a grid while avoiding obstacles. The environment is visualized using the Pygame library.

---

## Dependencies

- numpy
- pygame
- time (from Python's standard library)

To install dependencies, run:
```
pip install numpy pygame
```

---

## Classes and Functions

### Gene

Represents a gene with attributes:
- `name`: Identifies the gene.
- `gene_type`: Type of the gene.
- `Q_values`: Stores Q-values for different actions.

#### Methods

- `__init__(self, name, gene_type)`: Constructor.
- `get_type()`: Returns the gene type.
- `get_Q_value(action)`: Returns the Q-value for a given action.
- `set_Q_value(action, value)`: Sets the Q-value for a given action.

### GeneRegulatoryNetwork

Models a gene regulatory network for RL.

#### Attributes

- `genes`: Dictionary of genes.
- `actions`: List of possible actions.
- `alpha`: Learning rate.
- `gamma`: Discount factor.
- `epsilon`: Exploration rate.

#### Methods

- `__init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.1)`: Constructor.
- `add_gene(name, gene_type)`: Adds a new gene.
- `get_gene(name)`: Returns a gene by its name.
- `choose_action(state)`: Chooses an action based on the current state.
- `learn(s, a, r, s_)`: Updates Q-values.

### Functions

- `draw_grid(grid, position)`: Draws the grid using Pygame.
- `move(position, action)`: Returns a new position based on the current position and action.


### Run the script
```
python swarm_3.py
```

---

## Notes

- The Pygame window will pop up displaying the grid.
- The agent (in red) will navigate to the end (bottom-right corner), avoiding obstacles (in black).
- The program uses an ε-greedy policy for exploration and exploitation.

---

## Future Extensions

1. Dynamic obstacle placement.
2. Real-time training visualization.

---

## Known Issues

- The agent might get stuck if ε is too low during exploration.

For additional details, consult the inline comments in the script.

