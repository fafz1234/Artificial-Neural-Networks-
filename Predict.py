from __future__ import division
import numpy as np
import pandas as pd
from sigmoid import sgm

def predict(params_opt, features, classes, hidden_layer_size, input_layer_size, num_classes):
    theta1_opt = params_opt[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size,
                                                                                    (input_layer_size + 1))  # 25x401

    theta2_opt = params_opt[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_classes,
                                                                               (hidden_layer_size + 1))  # 10x26

    m = features.shape[0]
    y_matrix = pd.get_dummies(classes.ravel()).as_matrix()  # 5000x10

    a1 = features  # 5000x401
    z2 = theta1_opt.dot(a1.T)  # 25x401 * 401x5000 = 25x5000
    a2 = np.c_[np.ones((features.shape[0], 1)), sgm(z2.T)]  # 5000x26
    z3 = theta2_opt.dot(a2.T)  # 10x26 * 26x5000 = 10x5000
    a3 = sgm(z3).T  # 10x5000

    a3_binary = np.zeros_like(a3)
    a3_binary[np.arange(len(a3)), a3.argmax(1)] = 1

    accuracy = (np.sum(np.all(y_matrix == a3_binary, axis=1)) / m)

    return a3_binary, accuracy





