# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:31:29 2017

@author: hubert
"""

import numpy as np
import time


def fast_norm(x):
    y = np.sqrt(np.dot(x, x.T))
    return y

def ker(x,y, sig):
    return np.exp(-fast_norm(x-y)**2/(2*sig**2))
    #return np.dot(x,y)/(fast_norm(x)*fast_norm(y))

    
    
def h(i,j, z1, z2, sigma):
    return np.exp(- ((i-z1)**2 + (j-z2)**2)/(2*sigma**2))    

class mySOM():
    def __init__(self,dim1 = 10, dim2 = 10, sigma_kernel = 10, sigma= 1, eta = 0.1):
        self.dim1 = dim1
        self.dim2 = dim2
        self.sigma_kernel = sigma_kernel
        self.params = np.zeros((3,3,3))
        self.sigma = sigma
        self.eta = eta
       
    def winner(self, X):
        a = 0
        b = 0
        s = 0
        for i in range(self.dim1):
            for j in range(self.dim2):
                sc = ker(X, self.params[i][j], self.sigma_kernel)
                if sc > s:
                    s = sc
                    a = i
                    b = j
        return a, b
        
    def fit(self, X_train, n_it):
        self.params = np.zeros((self.dim1, self.dim2, len(X_train[0])))
        debut = time.time()
        for t in range(n_it):
            if t%(n_it/10.0)==0 :
                act = time.time()
                print(np.str((t*100.0)/n_it)+"%    "+ np.str(act-debut)+"s")
            for l in range(len(X_train)):
                z = self.winner(X_train[l])
                for i in range(self.dim1):
                    for j in range(self.dim2):
                        self.params[i][j] = self.params[i][j] + self.eta*h(i,j,z[0], z[1], self.sigma)*(X_train[l] - self.params[i][j])

                
        
    