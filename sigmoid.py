import math
import numpy as np
def sgm(z):
    return(1 / (1 + np.exp(-z)))


def sgmGradient (z):
    return sgm(z) * (1 - sgm(z))

