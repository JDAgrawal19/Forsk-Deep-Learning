# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:20:32 2018

@author: JITESH
"""
# read data from file and manage it
import pandas as pd
data=pd.read_csv("ex1data1.txt")
#data.columns=['0','1']
feature=data.iloc[:,0].values
price=data.iloc[:,1].values
price=price.reshape(-1,1)
feature=feature.reshape(-1,1)
# add a columns of ones to data x bcoz here value of x is 1
import numpy as np
feature_new=np.append(arr=np.ones((len(feature),1)).astype(float),values=feature.reshape(-1,1),axis=1)    

#declare theta values as 0
theta=np.zeros((2,1)).astype(float)    

#declare alpha and interations
alpha=0.01
iterations=15000

import cost_single_feature    
cost_single_feature.Compute_cost(feature_new,price,theta)

theta_for_graph=[]
J_theta_for_graph=[]
theta=cost_single_feature.gradientDescent(feature_new,price,theta,alpha,iterations,theta_for_graph)

cost_single_feature.visualize_theta(feature_new,price,alpha,iterations,theta_for_graph,J_theta_for_graph)
# plot the data
import matplotlib.pyplot as plt
plt.scatter(feature,price,color='red')
plt.plot(feature,feature_new.dot(theta),color='blue')
plt.title('profit vs population')
plt.xlabel('population')
plt.ylabel('profit')
plt.show()
        
 # plot graph for iterations vs J-theta values for diffrent theta   
plt.plot(range(iterations),J_theta_for_graph,color='blue')

    

from mpl_toolkits.mplot3d import Axes3D

##graphs from 1st technique 
b0=np.linspace(-10,10,100)
b1=np.linspace(-1,4,100)
xx,yy=np.meshgrid(b0,b1,indexing='xy')
Z=np.zeros((len(b0),len(b1)))

for i in range(len(b0)):
    for j in range(len(b1)):
        t = np.array([b0[i],b1[j]])
        t=t.reshape(-1,1)
        Z[i][j]= cost_single_feature.Compute_cost(feature_new,price,t)

from matplotlib import cm


fig = plt.figure()
ax = plt.subplot(111,projection='3d')   
Axes3D.plot_surface(ax,b0,b1,Z,cmap=cm.coolwarm)
plt.show()
fig = plt.figure()
ax = plt.subplot(111)
plt.contour(b0,b1,Z) 



#graphs from second technique

for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = cost_single_feature.Compute_cost(feature_new,price, theta=[[xx[i,j]], [yy[i,j]]])

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)




    
        