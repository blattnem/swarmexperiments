import numpy as np
import pygame

def five_minima(X):

    x, y = X
    return (x**2 - 10*x + y**2 - 10*y + 20) * np.sin(x) * np.sin(y) -2 * np.exp(-((x-2)**2 + (y-2)**2))

class Particle:
    """
    A Particle represents a candidate solution in the Particle Swarm Optimization (PSO) algorithm.

    Attributes:
        position (np.array): Current position of the particle.
        velocity (np.array): Current velocity of the particle.
        best_position (np.array): The best position visited by the particle so far.
        best_score (float): The best score (i.e., value of the objective function) found by the particle so far.
        w (float): Inertia weight factor.
        C1 (float): Ego influence factor.
        C2 (float): Social influence factor.
        teleport_probability (float): Probability of a teleportation event occurring.
        teleport_bias (float): Bias towards the lower boundary during teleportation.
    """

    def __init__(self, x0, teleport_probability, teleport_bias,c1,c2):
        """
        Initializes a particle with given initial position, teleport probability, and teleport bias.

        Args:
            x0 (np.array): Initial position of the particle.
            teleport_probability (float): Probability of a teleportation event occurring.
            teleport_bias (float): Bias towards the lower boundary during teleportation.
        """

        self.position = x0
        self.velocity = np.zeros_like(x0)
        self.best_position = x0
        self.best_score = float('inf')
        self.w = np.random.normal(loc=0.73, scale=0.0001)
        self.C1 = c1
        self.C2 = c2
        self.teleport_probability = teleport_probability
        self.teleport_bias = teleport_bias

    def teleport(self, minx, maxx):
        """
        Teleports the particle to a random position, biased towards the lower boundary of the search space.

        Args:
            minx (np.array): Lower boundary of the search space.
            maxx (np.array): Upper boundary of the search space.

        Returns:
            bool: Always returns True, indicating that a teleportation event occurred.
        """

        self.position = self.teleport_bias * np.array(minx) + (1 - self.teleport_bias) * np.random.uniform(minx, maxx, len(minx))
        self.velocity = np.zeros_like(self.position)
        self.best_score = float('inf')
        return True

    def print_parameters(self):
        """
        Prints the inertia weight and the personal and social influence factors of the particle.
        """

        print(self.w, self.C1, self.C2)

class Swarm:
    """
    A Swarm represents a group of particles in the Particle Swarm Optimization (PSO) algorithm.

    Attributes:
        function (callable): The objective function to be minimized.
        particles (list of Particle): The particles in the swarm.
        global_best_score (float): The best score (i.e., value of the objective function) found by any particle in the swarm.
        global_best_position (np.array): The position corresponding to the best score found by any particle in the swarm.
        N (int): The number of particles in the swarm.
        max_iter (int): The maximum number of iterations for the PSO algorithm.
        minx (np.array): Lower boundary of the search space.
        maxx (np.array): Upper boundary of the search space.
        damping_factor (float): Factor to control velocity (momentum) of the particles.
        positions (list of list of np.array): List to store positions of each particle at each time step.
        com (list of np.array): List to store center of mass of the swarm at each time step.
    """

    def __init__(self, function, N, max_iter, minx, maxx, damping_factor, teleport_probability, teleport_bias,c1,c2):
        """
        Initializes a swarm with given objective function, number of particles, maximum number of iterations,
        search space boundaries, damping factor, teleport probability, and teleport bias.

        Args:
            function (callable): The objective function to be minimized.
            N (int): The number of particles in the swarm.
            max_iter (int): The maximum number of iterations for the PSO algorithm.
            minx (np.array): Lower boundary of the search space.
            maxx (np.array): Upper boundary of the search space.
            damping_factor (float): Factor to control velocity (momentum) of the particles.
            teleport_probability (float): Probability of a teleportation event occurring.
            teleport_bias (float): Bias towards the lower boundary during teleportation.
        """

        self.function = function
        self.particles = [Particle(np.random.uniform(minx, maxx, len(minx)), teleport_probability, teleport_bias,c1,c2) for _ in range(N)]
        self.global_best_score = float('inf')
        self.global_best_position = np.random.uniform(minx, maxx, len(minx))
        self.N = N
        self.max_iter = max_iter
        self.minx = minx
        self.maxx = maxx
        self.damping_factor = damping_factor
        self.positions = [[] for _ in range(N)]
        self.com = []

    def solve(self):
        """
        Runs the Particle Swarm Optimization (PSO) algorithm and visualizes the swarm movement.
        """

        # Initialize the Pygame library
        pygame.init()

        # Define the size of the window
        WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 1000
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        # Load the background image
        background = pygame.image.load('../figures/energyland.png')

        # Scale the image to fit the window
        background = pygame.transform.scale(background, (WINDOW_WIDTH, WINDOW_HEIGHT))

        for i in range(self.max_iter):
            # Event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit the solve method

            # Draw the background
            window.blit(background, (0, 0))

            for j, particle in enumerate(self.particles):
                # Allow particle to teleport with the given probability
                teleport_occurred = False
                if np.random.rand() < particle.teleport_probability:
                    teleport_occurred = particle.teleport(self.minx, self.maxx)

                fitness_candidate = self.function(particle.position)

                if fitness_candidate < particle.best_score:
                    particle.best_score = fitness_candidate
                    particle.best_position = particle.position

                if fitness_candidate < self.global_best_score:
                    self.global_best_score = fitness_candidate
                    self.global_best_position = particle.position

            for j, particle in enumerate(self.particles):
                particle.velocity = (particle.w * particle.velocity 
                                     + particle.C1 * np.random.rand() * (particle.best_position - particle.position) 
                                     + particle.C2 * np.random.rand() * (self.global_best_position - particle.position))

                particle.velocity *= self.damping_factor

                particle.position += particle.velocity

                # Ensure we are within the bounds
                particle.position = np.clip(particle.position, self.minx, self.maxx)

                self.positions[j].append(particle.position.copy())  # Using copy()

                # Draw each particle
                x = (particle.position[0] - self.minx[0]) / (self.maxx[0] - self.minx[0]) * WINDOW_WIDTH
                y = (particle.position[1] - self.minx[1]) / (self.maxx[1] - self.minx[1]) * WINDOW_HEIGHT
                color = (0, 255, 0) if teleport_occurred else (255, 255, 255)
                pygame.draw.circle(window, color, (int(x), int(y)), 5)

            # Compute the center of mass and draw it
            com = np.mean([p.position for p in self.particles], axis=0)
            x_com = (com[0] - self.minx[0]) / (self.maxx[0] - self.minx[0]) * WINDOW_WIDTH
            y_com = (com[1] - self.minx[1]) / (self.maxx[1] - self.minx[1]) * WINDOW_HEIGHT
            pygame.draw.circle(window, (255, 0, 0), (int(x_com), int(y_com)), 7)  # Draw center of mass as red

            pygame.display.flip()
            delay = 100
            if i == 0: 
                delay = 3000
            pygame.time.delay(delay)

        # Clean up  
        pygame.quit()

# The objective function and other parameters can be set here
func =  five_minima # Set this to the actual function to be minimized
N = 50  # Set this to the desired number of particles
max_iter = 3000 # Set this to the desired maximum number of iterations
minx = np.array([0,0])  # Set this to the lower bound of the search space
maxx = np.array([10, 10])  # Set this to the upper bound of the search space
reinforcing_factor = 0.7  # Set this to the desired damping factor
teleport_probability = 0. # Set this to the desired teleport probability
teleport_bias = 0. # Set this to the desired teleport bias

pso = Swarm(func, N, max_iter, minx, maxx, reinforcing_factor, teleport_probability, teleport_bias,1.458,1.458)
pso.solve()


