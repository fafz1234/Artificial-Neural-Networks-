import numpy as np

def random_weights (L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, L_in + 1) * (2 * epsilon_init) - epsilon_init