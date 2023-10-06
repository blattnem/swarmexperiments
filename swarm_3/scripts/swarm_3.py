import numpy as np
import matplotlib.pyplot as plt
import pygame

class Path:
    def __init__(self, start, end, max_length, moves, grid_shape):
        self.start = start
        self.end = end
        self.max_length = max_length
        self.moves = moves
        self.grid_shape = grid_shape
        self.path = self.initialize_path()

    def initialize_path(self):
        path = [self.start]
        for _ in range(self.max_length):
            move = self.moves[np.random.choice(len(self.moves))]
            next_position = self.bound_position(path[-1] + move)
            path.append(next_position)
        return path

    def bound_position(self, pos):
        # Ensure the position does not exceed the grid boundaries
        return np.clip(pos, [0, 0], np.array(self.grid_shape) - 1)

    def compute_fitness(self, obstacles):
        fitness = 0
        for position in self.path:
            if any(self.is_within_obstacle(position, obstacle) for obstacle in obstacles):  # collision with obstacle
                fitness += 10000  # large penalty for collision
            else:
                fitness += np.linalg.norm(position - self.end)  # distance to end
        return fitness

    def is_within_obstacle(self, position, obstacle):
        # Check if the position is within the rectangle formed by the obstacle
        top_left, bottom_right = obstacle
        return top_left[0] <= position[0] <= bottom_right[0] and top_left[1] <= position[1] <= bottom_right[1]


class PSO:
    def __init__(self, num_particles, max_iter, start, end, obstacles, max_path_length, grid_shape):
        self.moves = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]  # up, down, right, left
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.start = start
        self.end = end
        self.obstacles = obstacles
        self.max_path_length = max_path_length
        self.grid_shape = grid_shape

        self.particles = [Path(start, end, max_path_length, self.moves, grid_shape) for _ in range(num_particles)]
        self.best_global_path = None
        self.best_global_fitness = np.inf


    def solve(self):
        tolerance = 1  # Adjust this based on your requirements
        for _ in range(self.max_iter):
            for particle in self.particles:
                fitness = particle.compute_fitness(self.obstacles)
                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_path = particle.path
                


                    # Check if any particle has reached the end position
                    if np.linalg.norm(particle.path[-1] - self.end) <= tolerance:
                        print("Target reached!")
                        return self.best_global_path, self.best_global_fitness

    

                for particle in self.particles:  # update paths (particles)
                    if np.random.rand() < 0.1:  # random chance to update a move in the path
                        index = np.random.choice(len(particle.path))
                        move = particle.moves[np.random.choice(len(particle.moves))]
                        particle.path[index] = particle.bound_position(particle.path[index] + move)

        return self.best_global_path, self.best_global_fitness



def main():
    # Define some color codes
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    
    # Pygame initialization
    pygame.init()
    screen = pygame.display.set_mode([800, 800])  # Set screen size
    pygame.display.set_caption('Real-Time PSO Path Finding')

    # Define starting point, ending point, obstacles, and parameters for PSO
    start = np.array([0, 0])
    end = np.array([8, 8])
    #obstacles = [np.array([5, 5]), np.array([6, 6])]
    obstacles = [
        [np.array([4, 4]), np.array([6, 6])],
        [np.array([7, 7]), np.array([9, 9])]
    ]
    grid_shape = (15, 15)
    num_particles = 10
    max_iter = 50
    max_path_length = 40

    # Initialize PSO
    pso = PSO(num_particles, max_iter, start, end, obstacles, max_path_length, grid_shape)

    clock = pygame.time.Clock()

    for i in range(max_iter):
        screen.fill(BLACK)  # Clear the screen

        for event in pygame.event.get():  # Check for user interaction
            if event.type == pygame.QUIT:  # If user clicked close
                pygame.quit()  # Flag that we are done so we exit this loop
                
        # Draw start and end positions
        pygame.draw.circle(screen, GREEN, start*20, 5)
        pygame.draw.circle(screen, YELLOW, end*60, 10)

        # Draw obstacles
        #for obs in obstacles:
        #    pygame.draw.rect(screen, RED, pygame.Rect(obs[0]*50, obs[1]*50, 50, 50))
        # Draw obstacles
        
        # Draw obstacles
        for obs in obstacles:
            top_left, bottom_right = obs
            width = (bottom_right[0] - top_left[0]) * 50
            height = (bottom_right[1] - top_left[1]) * 50
            pygame.draw.rect(screen, RED, pygame.Rect(top_left[0]*50, top_left[1]*50, width, height))

        best_path, best_fitness = pso.solve()

        # Draw best path
        for j in range(len(best_path)-1):
            pygame.draw.line(screen, BLUE, best_path[j]*50, best_path[j+1]*50, 2)

        # Check if best path has reached the target
        if np.array_equal(best_path[-1], end):
            print("Target reached!")
            break

        pygame.display.flip()  # Update screen
        clock.tick(24)  # Limit to 60 frames per second

    pygame.quit()  # Exit Pygame

main()
