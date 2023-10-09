import numpy as np
import matplotlib.pyplot as plt
import pygame

N = 342  # Number of agents
n_iter = 150  # Number of temporal iterations
dt = 0.1  # Time step
rO = 0.025  # Range of repulsion pressure
eps = 0.2  # Amplitude of repulsion pressure
rf = 0.03  # Range of flocking pressure
alpha = 1.  # Amplitude of flocking pressure
vO = 0.2 # Target speed
mu = 12.  # Amplitude of self-propulsion pressure
ramp = 0.1  # Amplitude of random pressure

PYGAME = True 

def buffering(rb, x, y, x_velocity, y_velocity):
    """
    Buffering function that creates a periodic boundary buffer for agents.

    Parameters
    ----------
    rb : float
        The range beyond which the agents need buffering.
    x, y : np.ndarray
        The arrays containing the x and y coordinates of the agents.
    x_velocity, y_velocity : np.ndarray
        The arrays containing the x and y velocities of the agents.

    Returns
    -------
    nb : int
        The number of buffered agents.
    x_buffer, y_buffer : np.ndarray
        The arrays containing the x and y coordinates of the buffered agents.
    x_velocity_buffer, y_velocity_buffer : np.ndarray
        The arrays containing the x and y velocities of the buffered agents.
    """

    N = len(x)
    x_buffer = np.zeros(8*N)
    y_buffer = np.zeros(8*N)
    x_velocity_buffer = np.zeros(8*N)
    y_velocity_buffer = np.zeros(8*N)
    
    x_buffer[0:N] = x[0:N]
    y_buffer[0:N] = y[0:N]
    x_velocity_buffer[0:N] = x_velocity[0:N]
    y_velocity_buffer[0:N] = y_velocity[0:N]
    nb = N - 1
    
    for k in range(0, N):
        if x[k] <= rb:
            nb += 1
            x_buffer[nb] = x[k] + 1.
            y_buffer[nb] = y[k]
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if x[k] >= 1.-rb:
            nb += 1
            x_buffer[nb] = x[k] - 1.
            y_buffer[nb] = y[k]
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if y[k] <= rb:
            nb += 1
            x_buffer[nb] = x[k]
            y_buffer[nb] = y[k] + 1.
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if y[k] >= 1.-rb:
            nb += 1
            x_buffer[nb] = x[k]
            y_buffer[nb] = y[k] - 1.
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if x[k] <= rb and y[k] <= rb:
            nb += 1
            x_buffer[nb] = x[k] + 1.
            y_buffer[nb] = y[k] + 1.
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if x[k] >= 1.-rb and y[k] <= rb:
            nb += 1
            x_buffer[nb] = x[k] - 1.
            y_buffer[nb] = y[k] + 1.
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if x[k] <= rb and y[k] >= 1.-rb:
            nb += 1
            x_buffer[nb] = x[k] + 1.
            y_buffer[nb] = y[k] - 1.
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
        if x[k] >= 1.-rb and y[k] >= 1.-rb:
            nb += 1
            x_buffer[nb] = x[k] - 1.
            y_buffer[nb] = y[k] - 1.
            x_velocity_buffer[nb] = x_velocity[k]
            y_velocity_buffer[nb] = y_velocity[k]
    return nb, x_buffer, y_buffer, x_velocity_buffer, y_velocity_buffer


def pressure(nb, x_buffer, y_buffer, x_velocity_buffer, y_velocity_buffer, x, y, x_velocity, y_velocity):
    """
    Calculate the pressure each agent experiences due to the flock, repulsion, and self-propulsion.

    Parameters
    ----------
    nb : int
        The number of buffered agents.
    x_buffer, y_buffer : np.ndarray
        The arrays containing the x and y coordinates of the buffered agents.
    x_velocity_buffer, y_velocity_buffer : np.ndarray
        The arrays containing the x and y velocities of the buffered agents.
    x, y : np.ndarray
        The arrays containing the x and y coordinates of the agents.
    x_velocity, y_velocity : np.ndarray
        The arrays containing the x and y velocities of the agents.

    Returns
    -------
    fx, fy : np.ndarray
        The arrays containing the forces in x and y direction experienced by each agent.
    """

    fx, fy = np.zeros_like(x), np.zeros_like(y)

    for j in range(0, N):
        repx, repy, flockx, flocky, nflock = 0., 0., 0., 0., 0

        for k in range(0, nb):
            d2 = (x_buffer[k] - x[j])**2 + (y_buffer[k] - y[j])**2

            if (d2 <= rf**2) and (j != k):
                flockx += x_velocity_buffer[k]
                flocky += y_velocity_buffer[k]
                nflock += 1
            
            if (d2 <= 4.*rO**2):
                d = np.sqrt(d2) + 1e-9  # added small constant to prevent division by zero
                repx += eps * (1. - d/(2.*rO))**1.5 * (x[j] - x_buffer[k])/d
                repy += eps * (1. - d/(2.*rO))**1.5 * (y[j] - y_buffer[k])/d

        normflock = np.sqrt(flockx**2 + flocky**2)
        if nflock == 0: 
            normflock = 1.
        flockx = alpha * flockx/normflock
        flocky = alpha * flocky/normflock

        vnorm = np.sqrt(x_velocity[j]**2 + y_velocity[j]**2)
        fpropx = mu * (vO - vnorm) * (x_velocity[j]/vnorm)
        fpropy = mu * (vO - vnorm) * (y_velocity[j]/vnorm)

        frandx = ramp * np.random.uniform(-1., 1.)
        frandy = ramp * np.random.uniform(-1., 1.)

        fx[j] = (flockx + frandx + fpropx + repx)
        fy[j] = (flocky + frandy + fpropy + repy)

    return fx, fy

if PYGAME:

    # initialize pygame
    pygame.init()

    # define the screen dimensions
    WIDTH, HEIGHT = 800, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # convert coordinates to pixel values
    scale_x = lambda x: int(x * WIDTH)
    scale_y = lambda y: int(y * HEIGHT)

    # set the color of the particles
    color = (255, 255, 255)
    color_average_velocity = (255, 0, 0)

    # Initial conditions
    x, y = np.zeros(N), np.zeros(N)
    x_velocity, y_velocity = np.zeros(N), np.zeros(N)

    for j in range(N):
        x[j] = np.random.uniform()
        y[j] = np.random.uniform()
        x_velocity[j] = np.random.uniform(-1., 1.)
        y_velocity[j] = np.random.uniform(-1., 1.)

    # run the simulation and draw the particles
    running = True
    while running:
        # check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # clear the screen
        screen.fill((0, 0, 0))

        # update the simulation
        nb, x_buffer, y_buffer, x_velocity_buffer, y_velocity_buffer = buffering(max(rO, rf), x, y, x_velocity, y_velocity)
        fx, fy = pressure(nb, x_buffer, y_buffer, x_velocity_buffer, y_velocity_buffer, x, y, x_velocity, y_velocity)
        x_velocity += fx * dt
        y_velocity += fy * dt
        x += x_velocity * dt
        y += y_velocity * dt
        x = (1. + x) % 1
        y = (1. + y) % 1

        # Calculate average position and velocity
        avg_pos_x, avg_pos_y = np.mean(x), np.mean(y)
        avg_vel_x, avg_vel_y = np.mean(x_velocity), np.mean(y_velocity)

        scaling_factor = 100
        avg_vel_x *= scaling_factor
        avg_vel_y *= scaling_factor

        # draw the particles
        for i in range(N):
            pygame.draw.circle(screen, color, (scale_x(x[i]), scale_y(y[i])), 3)
        
        pygame.draw.line(screen, color_average_velocity, (scale_x(avg_pos_x), scale_y(avg_pos_y)), (scale_x(avg_pos_x+avg_vel_x), scale_y(avg_pos_y+avg_vel_y)), 3)
    
        # update the display
        pygame.display.flip()

    # quit pygame
    pygame.quit()

else:

    # Initial conditions
    x, y = np.zeros(N), np.zeros(N)
    x_velocity, y_velocity = np.zeros(N), np.zeros(N)

    for j in range(N):
        x[j] = np.random.uniform()
        y[j] = np.random.uniform()
        x_velocity[j] = np.random.uniform(-1., 1.)
        y_velocity[j] = np.random.uniform(-1., 1.)

    for iterate in range(n_iter):
        print("iteration: ", iterate)
        nb, x_buffer, y_buffer, x_velocity_buffer, y_velocity_buffer = buffering(max(rO, rf), x, y, x_velocity, y_velocity)
        fx, fy = pressure(nb, x_buffer, y_buffer, x_velocity_buffer, y_velocity_buffer, x, y, x_velocity, y_velocity)
        x_velocity += fx * dt
        y_velocity += fy * dt
        x += x_velocity * dt
        y += y_velocity * dt
        x = (1. + x) % 1
        y = (1. + y) % 1

    average_x = np.mean(x)
    average_y = np.mean(y)
    average_x_velocity = np.mean(x_velocity)
    average_y_velocity = np.mean(y_velocity)
    scaling_factor = 10

    scaled_velocity_x = average_x_velocity * scaling_factor
    scaled_velocity_y = average_y_velocity * scaling_factor
    
    plt.scatter(x, y)
    plt.quiver(x, y, x_velocity, y_velocity, headlength=5)


    plt.scatter(np.mean(x),np.mean(y))
    plt.quiver(average_x,average_y,scaled_velocity_x,scaled_velocity_y,color='r', scale=1, width=0.005, headwidth=5, headlength=7, zorder=5)
    

    plt.axis([0., 1., 0., 1.])
    plt.show()
