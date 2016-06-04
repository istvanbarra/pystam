# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:09:05 2016

@author: istvan
"""

import numpy as np
from scipy.optimize import minimize


class MaximumLikelihood():
    def __init__(self,model, prior):
        self.model=model
        self.prior=prior

    def estimate(self, y, z):
        num_y=len(y)
        #wrap the loglikelihood function
        def f(param_trans):
            param=self.prior.transform_back_param(param_trans)
            value,state=self.model.log_likelihood(param, y,z)
            print('value at {} is {}'.format(param,value[0]))
            
            return -value/num_y
        #optimize function
#        param_initial=np.zeros(self.prior.dim_param)
        param_initial=np.array([np.log(0.01),np.log(0.01),0])
        results=minimize(f, param_initial, method='BFGS')
#        results=minimize(f, param_initial, method='Nelder-Mead')
        

        self.log_likelihood=-num_y*results.fun
        self.param=self.prior.transform_back_param(results.x)

        #calcualte stander error from hession
        gradient=self.prior.gradient(results.x)
        self.variance=results.hess_inv/num_y
        self.high=self.param+1.96*np.multiply(np.sqrt(np.diagonal(self.variance)),gradient)
        self.low=self.param-1.96*np.multiply(np.sqrt(np.diagonal(self.variance)),gradient)
    
    def log_likelihood_profile(self,y,z,param,index,a,b,num_eval):
        x=a+np.arange(0,num_eval+1)*(b-a)/num_eval
             
        f=np.zeros(num_eval+1)
        param_temp=param
        print(param_temp)
        for i in range(num_eval+1):
            print(i)
            param_temp[index]=x[i]  
            print(param_temp)
            f_temp,state_temp=self.model.log_likelihood(param_temp,y,z)
            f[i]=f_temp
        return x,f