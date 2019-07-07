# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:19:10 2018

@author: JITESH
"""

def Compute_cost(X,y,theta):
    cost=np.sum((np.dot(X,theta)-y)**2)/(2*len(y))
    return cost


import numpy as np
import pandas as pd
data=pd.read_csv("ex1data2.txt",names=['Area','Beds','Price'])

area=np.array(data.Area)
beds=np.array(data.Beds)
price=np.array(data.Price)

area=np.vstack(area)
beds= np.vstack(beds)
price=np.vstack(price)

X=np.hstack((area,beds))

X = np.hstack((np.ones_like(area),X))

theta=np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,price))

Y=X.T

z=np.dot(Y,X)


asd=np.linalg.inv(z)
asdf=np.dot(Y,price)

asdfg=np.dot(asd,asdf)


cost=Compute_cost(X,price,theta)