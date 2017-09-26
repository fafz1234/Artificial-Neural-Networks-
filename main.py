import numpy as np
import pandas as pd
from nnCostFunction import NNCostFunc, computeGrad
import scipy.io
from randomWeights import random_weights
from scipy.optimize import minimize
from nnPredict import predict

## import data
data = scipy.io.loadmat('/Users/FAVZ/PycharmProjects/untitled/ex4data1.mat')
weights = scipy.io.loadmat('/Users/FAVZ/PycharmProjects/untitled/ex4weights.mat') #X.insert(0, 'bias', 1)
X = np.insert(data['X'], 0, 1, axis = 1)
y = data['y']
theta1, theta2 = weights['Theta1'], weights['Theta2']
params = np.r_[theta1.ravel(), theta2.ravel()]

hidden_layer_size = 25
input_layer_size = 400
num_classes = 10
lamb_da = 1

theta1_r = random_weights(input_layer_size, hidden_layer_size)
theta2_r = random_weights(hidden_layer_size, num_classes)
params_ = np.r_[theta1_r.ravel(), theta2_r.ravel()]

res = minimize(NNCostFunc, x0 = params_, args = (input_layer_size, hidden_layer_size, num_classes, X, y, lamb_da), method='TNC', jac = computeGrad)
params_opt = res.x

prediction, accuracy = predict(params_opt, X, y, hidden_layer_size, input_layer_size, num_classes)

print 'Model Accuracy is:', "{:.1%}".format(accuracy)





