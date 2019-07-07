# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:06:15 2018

@author: JITESH
"""
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy.optimize import minimize


def sigmoid(z):
    return 1.0/(1 +  np.e**(-z))

def displayData(X):
    fig,ax=plt.subplots(10,10,sharex=True,sharey=True)
    img_num=0
    for i in range(10):
        for j in range(10):
            img=X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            img_num+=1
    return (fig,ax)


def lrCostFunction(theta,X,y,reg_param):
    m = len(y) 
    J =((np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m +
       (reg_param/m)*np.sum(theta**2))   
    # Gradient
    # Non-regularized 
    grad_0 = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    # Regularized
    grad_reg = grad_0 + (reg_param/m)*theta
    grad_reg[0] = grad_0[0] 
    return (J,grad_reg)


def oneVsAll(X, y, num_labels, reg_param):
    #Calculates parameters 
    n = np.size(X,1)
    theta = np.zeros((n,num_labels))
    # Function to find parameters for single logit
    def findOptParam(p_num):
        outcome = np.array(y == p_num).astype(int)
        initial_theta = theta[:,p_num]
        results = minimize(lrCostFunction,
			   initial_theta,
                           method='Newton-CG',
                           args=(X,outcome,reg_param),
                           jac=True,
		           tol=1e-6,
                           options={'maxiter':400,
                                    'disp':True})
        theta[:,p_num] = results.x
    
    
    for digit in range(10):
        findOptParam(digit)
    
    return theta


def predictOneVsAllAccuracy(est_theta,X):
    """
    classifies each observation by using the
    highest predicted probability from possible classifications.
    """

    probs = np.dot(X,est_theta)
    predict = np.argmax(probs,axis=1)
    
    return predict


def predict(theta1,theta2,X):
    m = len(X) 
    if np.ndim(X) == 1:
        X = X.reshape((-1,1))     #one dimensional or not
    D1 = np.hstack((np.ones((m,1)),X))
   
    #hidden layer from theta1 parameters
    hidden_pred = np.dot(D1,theta1.T) # (5000 x 401) x (401 x 25) = 5000 x 25
    ones = np.ones((len(hidden_pred),1)) # 5000 x 1
    hidden_pred = sigmoid(hidden_pred)
    hidden_pred = np.hstack((ones,hidden_pred)) # 5000 x 26
    
    #output layer from new design matrix
    output_pred = np.dot(hidden_pred,theta2.T) # (5000 x 26) x (26 x 10)    
    output_pred = sigmoid(output_pred)
    # Get predictions
    p = np.argmax(output_pred,axis=1)
    
    return p

data=scipy.io.loadmat("ex3data1.mat")
feature=data.get("X")
label=data.get("y").flatten()
label[label==10]=0

X=np.hstack((np.ones((len(label),1)),feature))

#randomly selects 100 data points to display
rand_indices=np.random.randint(0,len(X),100)
sel=feature[rand_indices,:]

#display data
digit_grid, ax=displayData(sel)
digit_grid.show()



reg_param = 1.0
theta = oneVsAll(X,label,10,reg_param)


predictions = predictOneVsAllAccuracy(theta,X)
accuracy = np.mean(label == predictions) * 100

#neural networks


raw_params = scipy.io.loadmat("ex3weights.mat")
theta1 = raw_params.get("Theta1") # 25 x 401
theta2 = raw_params.get("Theta2") # 10 x 26


predictions = (predict(theta1,theta2,feature) + 1) % 10
accuracy = np.mean(label== predictions) * 100



