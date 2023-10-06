import streamlit as st
import numpy as np
from swarm_2 import Particle, Swarm, five_minima  # Assuming your existing classes and functions are in 'your_module'

def run_simulation():
    func = five_minima
    N = st.sidebar.slider("Number of Particles", 10, 100, 50)
    max_iter = st.sidebar.slider("Max Iterations", 100, 5000, 3000)
    minx = np.array([0, 0])
    maxx = np.array([10, 10])
    reinforcing_factor = st.sidebar.slider("Reinforcing Factor", 0.1, 1.0, 0.7)
    teleport_probability = st.sidebar.slider("Teleport Probability", 0.0, 1.0, 0.0)
    teleport_bias = st.sidebar.slider("Teleport Bias", 0.0, 1.0, 0.0)
    C1 = st.sidebar.slider("C1 (Ego Influence Factor)", 0.0, 2.0, 1.49618)
    C2 = st.sidebar.slider("C2 (Social Influence Factor)", 0.0, 2.0, 1.52)

    pso = Swarm(func, N, max_iter, minx, maxx, reinforcing_factor, teleport_probability, teleport_bias, C1, C2)
    pso.solve()

# Sidebar title
st.sidebar.title("Parameters")

# Main title
st.title("Particle Swarm Optimization")

# Add a button to run the simulation
if st.button("Run Simulation"):
    run_simulation()
