import numpy as np
from sigmoid import sgm, sgmGradient
import pandas as pd

def NNCostFunc(params, input_layer_size, hidden_layer_size, num_labels, features, classes, lamb_da):
    theta1 = params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size,
                                                                               (input_layer_size + 1))  # 25x401
    theta2 = params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))  # 10x26

    m = features.shape[0]
    y_matrix = pd.get_dummies(classes.ravel()).as_matrix() # 5000x10

    a1 = features  #5000x401
    z2 = theta1.dot(a1.T)  # 25x401 * 401x5000 = 25x5000
    a2 = np.c_[np.ones((features.shape[0], 1)), sgm(z2.T)]  # 5000x26
    z3 = theta2.dot(a2.T)  # 10x26 * 26x5000 = 10x5000
    a3 = sgm(z3)   # 10x5000

    # cost with regularization
    J = ((-1 * np.sum((np.log(a3.T) * y_matrix) + (np.log(1 - a3).T * (1 - y_matrix)))) / m ) + ((lamb_da) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))) / (2 * m)
    return J

def computeGrad (params, input_layer_size, hidden_layer_size, num_labels, features, classes, lamb_da):

    theta1 = params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size,
                                                                               (input_layer_size + 1))  # 25x401
    theta2 = params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))  # 10x26

    m = features.shape[0]
    y_matrix = pd.get_dummies(classes.ravel()).as_matrix() # 5000x10

    a1 = features  #5000x401
    z2 = theta1.dot(a1.T)  # 25x401 * 401x5000 = 25x5000
    a2 = np.c_[np.ones((features.shape[0], 1)), sgm(z2.T)]  # 5000x26
    z3 = theta2.dot(a2.T)  # 10x26 * 26x5000 = 10x5000
    a3 = sgm(z3)   # 10x5000

    d3 = a3.T - y_matrix    # 5000x10
    d2 = d3.dot(theta2[:, 1:]) * sgmGradient(z2.T)  # 5000x10 * 10x25 = 5000x25

    D1 = d2.T.dot(a1)  # 25x5000 * 5000x401 = 25x401
    D2 = d3.T.dot(a2)  # 10x5000 *5000x26 = 10x26

    theta1_ = np.c_[np.zeros((theta1.shape[0], 1)), theta1[:, 1:]]
    theta2_ = np.c_[np.zeros((theta2.shape[0], 1)), theta2[:, 1:]]

    theta1_grad = (D1/m) + (theta1_ * lamb_da) / m
    theta2_grad = (D2/m) + (theta2_ * lamb_da) / m
    params_grad = np.r_[theta1_grad.ravel(), theta2_grad.ravel()]

    return params_grad