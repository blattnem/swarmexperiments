import numpy as np
import pygame
import time

class Gene:
    """
    Class to represent a gene with a name, type, and associated Q-values for different actions.
    """

    def __init__(self, name, gene_type):
        """
        Initialize a new gene with a name, type, and an empty dictionary for Q-values.

        Args:
            name: A string representing the name of the gene.
            gene_type: A string representing the type of the gene.
        """

        self.name = name
        self.gene_type = gene_type
        self.Q_values = {}  # Q-values for actions, in form of {'action1': Q1, 'action2': Q2}

    def get_type(self):
        """
        Return the type of the gene.

        Returns:
            A string representing the type of the gene.
        """

        return self.gene_type

    def get_Q_value(self, action):
        """
        Return the Q-value for the given action.

        Args:
            action: A string representing the action.

        Returns:
            A float representing the Q-value of the action.
        """

        return self.Q_values.get(action, 0.0)

    def set_Q_value(self, action, value):
        """
        Set the Q-value for the given action to the given value.

        Args:
            action: A string representing the action.
            value: A float representing the new Q-value for the action.
        """

        self.Q_values[action] = value

class GeneRegulatoryNetwork:
    """
    Class to model a gene regulatory network as a reinforcement learning problem.
    """

    def __init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.1):
        """
        Initialize a new Gene Regulatory Network with given parameters.

        Args:
            actions: A list of strings representing the possible actions.
            alpha: A float representing the learning rate. Default is 0.5.
            gamma: A float representing the discount factor. Default is 0.9.
            epsilon: A float representing the exploration rate. Default is 0.1.
        """

        self.genes = {}
        self.actions = actions  # possible actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

    def add_gene(self, name, gene_type):
        """
        Add a new gene to the network with the given name and type.

        Args:
            name: A string representing the name of the gene.
            gene_type: A string representing the type of the gene.
        """

        self.genes[name] = Gene(name, gene_type)

    def get_gene(self, name):
        """
        Return the gene with the given name.

        Args:
            name: A string representing the name of the gene.

        Returns:
            An instance of the Gene class.
        """

        return self.genes.get(name)

    def choose_action(self, state):
        """
        Choose an action based on the current state: either randomly (for exploration) or based on Q-values (for exploitation).

        Args:
            state: A tuple representing the current state.

        Returns:
            A string representing the chosen action.
        """

        if np.random.uniform(0, 1) < self.epsilon:
            # exploration: choose a random action
            action = np.random.choice(self.actions)
        else:
            # exploitation: choose the action with highest Q-value
            state_gene = self.get_gene(state)
            if state_gene.Q_values:
                action = max(state_gene.Q_values, key=state_gene.Q_values.get)
            else:
                action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        Update the Q-value of the given action based on the reward and the maximum future Q-value.

        Args:
            s: A tuple representing the current state.
            a: A string representing the current action.
            r: A float representing the current reward.
            s_: A tuple representing the new state.
        """

        old_q = self.get_gene(s).get_Q_value(a)
        if s_:
            if self.get_gene(s_).Q_values:
                max_future_q = max(self.get_gene(s_).Q_values.values())
            else:
                max_future_q = 0
        else:
            max_future_q = 0
        new_q = (1 - self.alpha) * old_q + self.alpha * (r + self.gamma * max_future_q)
        self.get_gene(s).set_Q_value(a, new_q)

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define grid size and scale
GRID_SIZE = 5
SCALE = 100

# Create Pygame display
win = pygame.display.set_mode((GRID_SIZE * SCALE, GRID_SIZE * SCALE))

def draw_grid(grid, position):
    """
    Draw the grid on the Pygame window.

    Args:
        grid: A 2D numpy array representing the grid.
        position: A tuple representing the current position.
    """

    win.fill(WHITE)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == 1:
                pygame.draw.rect(win, BLACK, pygame.Rect(j * SCALE, i * SCALE, SCALE, SCALE))
            elif (i, j) == position:
                pygame.draw.rect(win, RED, pygame.Rect(j * SCALE, i * SCALE, SCALE, SCALE))
            else:
                pygame.draw.rect(win, GREEN, pygame.Rect(j * SCALE, i * SCALE, SCALE, SCALE), 1)
    pygame.display.update()

# Initialize the grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Place some obstacles
grid[1, 1] = 1
grid[1, 3] = 1
grid[2, 2] = 1
grid[3, 1] = 1
grid[4, 1] = 1

# Set start and end points
start = (0, 0)
end = (4, 4)

# Create a new Gene Regulatory Network
actions = ['up', 'down', 'left', 'right']
grn = GeneRegulatoryNetwork(actions,epsilon=0.3)

# Add genes for each position on the grid
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        grn.add_gene((i, j), "Regulatory")

def move(position, action):
    """
    Get a new position based on the current position and the action.

    Args:
        position: A tuple representing the current position.
        action: A string representing the action.

    Returns:
        A tuple representing the new position.
    """

    if action == 'up':
        return (max(0, position[0] - 1), position[1])
    elif action == 'down':
        return (min(4, position[0] + 1), position[1])
    elif action == 'left':
        return (position[0], max(0, position[1] - 1))
    else:  # right
        return (position[0], min(4, position[1] + 1))

# Training
for episode in range(20):
    print("Episode: ",episode)
    position = start
    while position != end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        
        action = grn.choose_action(position)
        new_position = move(position, action)
        if grid[new_position] == 1 or new_position == position:  # moved into obstacle or off the grid
            reward = -1
        elif new_position == end:
            reward = 1
        else:
            reward = -0.3
        grn.learn(position, action, reward, new_position)
        position = new_position
        draw_grid(grid, position)
        pygame.display.update()
        time.sleep(0.05)

# Use
position = start
while position != end:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)

    action = grn.choose_action(position)
    print(f"Position: {position}, Action: {action}")
    position = move(position, action)
    draw_grid(grid, position)
    pygame.display.update()
    time.sleep(0.1)
    if position == end:
        break

# Freeze the scene when agent reaches the end
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)

pygame.quit()
