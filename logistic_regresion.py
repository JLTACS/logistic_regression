import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def forward(X,W,b):
    Z = np.matmul(W,X.T)
    return sigmoid(Z + b)

def cross_entro_error(pY,T):
    return np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY)) * -1

def classification_rate(Y,P):
    return np.mean(Y == P)

def training(X,Y):
    D = X.shape[1]
    W = np.random.randn(D)
    b = 0
    train_costs = []

    print("Ingrese el Learning Rate: ")
    lr = float(input())
    print("Ingrese las épocas: ")
    ages = int(input())

    for i in range(ages):
        pYtrain = forward(X,W,b)
        train_costs.append(cross_entro_error(pYtrain,Y))

        #gradient descent
        W -= lr * X.T.dot((pYtrain-Y))
        b -= lr * (pYtrain-Y).sum()
    
    print("Final train classification_rate:", classification_rate(Y, np.round(pYtrain)))
    
    return W,b,train_costs

    
def logistic_regression(xtest,ytest,W,b): 

    pYtest = np.round(forward(xtest,W,b))
    print("Final test classification_rate:", classification_rate(ytest, pYtest))
    return pYtest





