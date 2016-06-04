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
        num_param=param.size/np.shape(param)[-1] 
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



class DynamicModel():
    '''
    Static model class
    '''
    
    __metaclass__ = ABCMeta
    
    
    @abstractmethod   
    def log_obs_dens(self,param,state,y,z):
        pass

   
    def initial_state(self,param):

        num_param=param.size/np.shape(param)[-1] 
        
        return np.zeros(num_param)
        
     
    def state_transition(self,param,state_prev,y_prev,z_prev):
       
        return state_prev
        
    def log_likelihood(self, param, y,z):
        
        num_y=len(y)
        num_param=param.size/np.shape(param)[-1] 
        log_like=np.zeros(num_param)
        state=self.initial_state(param) 
        
        for i in range(0,num_y):
            log_like=log_like+self.log_obs_dens(param,state,y[i],z[i])
            #print('log_like loglike {}'.format(log_like))
            state=self.state_transition(param,state,y[i],z[i])  
            #print('log_like state {}'.format(state))
        return log_like,state        
        
        
class GARCH(DynamicModel):
    def log_obs_dens(self,param,state,y,z):
#        print('log dens param {}'.format(param))
#        print('log dens state {}'.format(state))
#        print('log dens y {}'.format(y))
#        print('log dens z {}'.format(z))
                
        if(np.any(state)<0):
            raise ValueError('state error {} at param {}'.format(state,param))
        std_dev=np.sqrt(state)
        return norm.logpdf(y, loc=0, scale=std_dev)    

     
    def initial_state(self,param):

        state=np.divide(np.take(param,0,-1),(1- np.take(param,1,-1)-np.take(param,2,-1))) 
        return state
    
    def state_transition(self,param,state_prev,y_prev,z_prev):

        state=np.take(param,0,-1)+np.take(param,1,-1)*y_prev*y_prev+np.take(param,2,-1)*state_prev
        if(np.any(state)<0):     
            print('in state prev state {}'.format(state_prev))
            print('in state const {}'.format(np.take(param,0,-1)))
            print('in state a {}'.format(np.take(param,1,-1)*np.divide(y_prev,np.sqrt(state_prev))))
            print('in state b {}'.format(np.take(param,2,-1)*state_prev))
            raise ValueError('state {}'.fromat(state))
        return state
    




    
class Ssf(DynamicModel):
    __metaclass__ = ABCMeta
    
    
    @abstractmethod   
    def system_matrices(self,param):
        
        
        pass
        
    def update(self,ssf,state,y):
        pass
    def propagate(self):    
        pass
    def log_obs_dens(self,param,state,y,z):
        ssf=self.system_matrices(param)
        
        
    
    def state_transition(self,param,state_prev,y_prev,z_prev):
        ssf=self.system_matrices(param)
         
    
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
    
    par=np.array([[0.05,0.03,0.9],[0.05,0.04,0.91]])
    state=[ 0.71428571,  1. ]
    z=np.array([[3,1,3],[2,1,2]])
    y=np.array([0.11,0.2])      
    garch=GARCH()
    print('test log_obs_dens: {}'.format(garch.log_obs_dens(par[0],state[0],y[0],z[0])))
    print('test initial_state: {}'.format(garch.initial_state(par[0])))
    print('test state_transition: {}'.format(garch.state_transition(par[0],state[0],y[0],z[0])))
    log_like,state=garch.log_likelihood(par,y,z)
    print('test log_likelihood: {} {}'.format(log_like,state))
    
    
            