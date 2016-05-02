# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:15:20 2016

@author: istvan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:38:23 2015

@author: istvan
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.stats import norm,poisson

class StaticModel():
    '''
    Static model class
    '''
    
    __metaclass__ = ABCMeta
    
    
    @abstractmethod   
    def log_obs_dens(self,param,y,z):
        '''           
        Caclulates the natural logarithm of the observation density 
        at param with covariate z \n
        ------------------------------------------------------ \n      
        Input: \n            
        param - numTheta x dim_param numpy array \n 
        y - 1 x dim_y numpy array \n
        z - 1 x dim_z numpy array \n
        ------------------------------------------------------ \n          
        Returns: \n   
        log_obs - num_param x 1 numpy array \n   
        '''
        
    def log_likelihood(self, param, y,z):
        '''           
        Caclulates the natural logarithm of the likelihood at param
        with covariate z \n
        ------------------------------------------------------ \n      
        Input: \n            
        param - num_param x dim_param numpy matrix \n 
        y - 1 x dim_y numpy matrix \n
        z - 1 x dim_z numpy matrix \n
        ------------------------------------------------------ \n          
        Returns: \n   
        log_like - num_param x 1 numpy array \n   
        '''
        
        num_y=len(y)
        num_param=param.size/np.shape(param)[-1] #!!!!!!!!!
        log_like=np.zeros(num_param)
        
        for i in range(0,num_y):
            log_like=log_like+self.log_obs_dens(param,y[i],z[i])
        return log_like
        
        
class LinearRegression(StaticModel):
    def log_obs_dens(self,param,y,z):
        dim_param=np.shape(param)[-1]
        mean=np.inner(np.take(param,range(0,dim_param-1),-1),z)
        return norm.logpdf(y, loc=mean, scale=np.take(param,dim_param-1,-1))    

class LogisticRegression(StaticModel):
    def log_obs_dens(self, param,y,z):
        z=np.inner(param,z)
        prob=1/(1+np.exp(-z))
        return y*np.log(prob)+(1-y)*np.log(1-prob)

class PoissonRegression(StaticModel):
    def log_obs_dens(self,param,y,z):
        mu=np.exp(np.inner(z,param))
        return np.log(poisson.pmf(y,mu))
            
            
if __name__ == "__main__":
    '''
    Testing the StaticModel class
    '''
    
    par=np.array([[1,1,1,1],[2,2,2,2]])
    z=np.array([[3,1,3],[2,1,2]])
    y=np.array([1,2])

    regression=LinearRegression()
    print('test log_obs_dens: {}'.format(regression.log_obs_dens(par,y[0],z[0])))
    print('test log_likelihood: {}'.format(regression.log_likelihood(par,y,z)))
    
            
            