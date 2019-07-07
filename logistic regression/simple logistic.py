# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:59:37 2018

@author: JITESH
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
#sigmoid function
def sigmoid(z):
    return 1.0/(1+np.e**(-z))

#cost function and gradient
"""def costFunction(X,y,theta):
    m=len(y)
    #J=(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-(1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)
    grad=(np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    return (J,grad)
"""    
def costFunction1(theta,X,y):
    m = len(y) 
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    return (J, grad)


data=pd.read_csv("ex2data1.txt",names=["score1","score2","result"])
feature=data.iloc[:,0:2].values
result=data.iloc[:,2].values
score1=np.array(data.score1)
score2=np.array(data.score2)
#result=np.array(data.result)
score1=np.vstack(score1)
score2=np.vstack(score2)
#result=np.vstack(result)

feature=np.hstack((score1,score2))
feature=np.hstack((np.ones_like(score1),feature))


#visualize the plot 
pos=data[data["result"]==1]["score1"]
neg=data[data["result"]==0]["score1"]

pos2=data[data["result"]==1]["score2"]
neg2=data[data["result"]==0]["score2"]

plt.scatter(pos,pos2,color='black')
plt.scatter(neg,neg2,color='yellow',marker="x")


theta=np.zeros(3)

cost1,grad1=costFunction1(theta,feature,result)
res=minimize(costFunction1,theta,method='Newton-CG',args=(feature,result),jac=True,options={'maxiter':500,'disp':True})

