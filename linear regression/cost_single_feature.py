# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 13:32:02 2018

@author: JITESH
"""

def Compute_cost(feature,price,theta):
    cost=(feature.dot(theta)-price)**2
    cost=cost.sum(axis=0)
    cost=cost/(len(feature)*2)
    return cost


def gradientDescent(feature_new,price,theta,alpha,iteration,theta_for_graph):
    for i in range(iteration):
        diff_cost=feature_new.dot(theta)-price
        diff_cost=diff_cost*feature_new
        diff_cost=diff_cost.sum(axis=0)
        diff_cost=(diff_cost*alpha)/len(price)
        diff_cost=diff_cost.reshape(-1,1)
        theta=theta-diff_cost
        theta_for_graph.append(theta)
    return theta    


def visualize_theta(feature_new,price,alpha,iterations,theta_for_graph,J_theta_for_graph):
    for i in range(iterations):
        J_theta_for_graph.append(Compute_cost(feature_new,price,theta_for_graph[i]))
    return J_theta_for_graph
    
    