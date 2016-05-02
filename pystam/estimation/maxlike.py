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
            value=self.model.log_likelihood(param, y,z)
            return -value/num_y
        #optimize function
        param_initial=np.zeros(self.prior.dim_param)
        
        results=minimize(f, param_initial, method='BFGS')

        self.log_likelihood=-num_y*results.fun
        self.param=self.prior.transform_back_param(results.x)

        #calcualte stander error from hession
        gradient=self.prior.gradient(results.x)
        self.variance=results.hess_inv/num_y
        self.high=self.param+1.96*np.multiply(np.sqrt(np.diagonal(self.variance)),gradient)
        self.low=self.param-1.96*np.multiply(np.sqrt(np.diagonal(self.variance)),gradient)
        
      