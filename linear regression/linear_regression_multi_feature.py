# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:24:55 2018

@author: JITESH
"""
def Compute_cost(X,y,theta):
    cost=np.sum((np.dot(X,theta)-y)**2)/(2*len(y))
    return cost

def gradDescent(X,y,theta,alpha,iterations):
    m=len(y)
    J_history=np.zeros(iterations)
    
    for i in range(iterations):
        theta=theta-(alpha/m)*np.sum((np.dot(X,theta)-y)[:,None]*X,axis=0)
        J_history[i]=Compute_cost(X,y,theta)
    return theta,J_history
        
    


import numpy as np
import pandas as pd
data=pd.read_csv("ex1data2.txt",names=["Area","Beds","price"])

area=np.array(data.Area)
beds=np.array(data.Beds)
price=np.array(data.price)
area=np.vstack(area)
beds=np.vstack(beds)
feature=np.hstack((area,beds))

#normalize the data
feature=np.divide((feature-np.mean(feature,axis=0)),np.std(feature,axis=0))
#add a column of constant values for theta0

feature=np.hstack((np.ones_like(area),feature))

#run gradient descent
alpha=0.01
iterations=2000
theta=np.zeros(3)

theta, hist = gradDescent(feature,price, theta, alpha, iterations)

import matplotlib.pyplot as plt
plt.plot(range(iterations),hist,color='blue')



cost1=Compute_cost(feature,price,theta)



























