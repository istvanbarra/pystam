# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:36:57 2016

@author: istvan
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.stats import norm,lognorm,dirichlet,uniform


class Prior:
    '''
    Parameter class
    Implements the evaluation of the log prior density
    and transformation of the variables base on parameter restrictions
    '''

    __metaclass__ = ABCMeta
    def __init__(self, restriction):
        for i in range(0,len(restriction)):
            if(restriction[i] not in ['', 'pos', '01']):
                raise ValueError('Invalid restriction only! Use pos, 01 or empty string')                
        self.restriction=restriction
        self.dim_param=len(restriction)
    
    @abstractmethod   
    def log_prior(self, param):
        '''           
        Caclulates the natural logarithm of the prior at param \n
        ------------------------------------------------------  \n      
        Input: \n            
        param - num_param x dim_param numpy array \n   
        ------------------------------------------------------ \n          
        Returns: \n   
        log_prior - num_param x 1 numpy array \n   
        '''
    @abstractmethod   
    def sample_prior(self, num_param):
        '''           
        Caclulates samples form the prior  \n
        ------------------------------------------------------  \n      
        Input: \n            
        num_param - number of prior samples  \n   
        ------------------------------------------------------ \n          
        Returns: \n   
        param - num_param x dim_param numpy array \n   
        '''
    
    def transform_positive(self,x):
        return np.log(x)

    def transform_back_positive(self,x):
        return np.exp(x)
    
    def transform_01(self,x):
        return np.log(x)-np.log(1-x)
    
    def transform_back_01(self,x):
        return np.exp(x)/(1+np.exp(x))  
    
    def transform_param(self, param):
        '''          
        Transforms param based on the restrictions 
        in res  \n
        ------------------------------------------------------ \n  
        Input: \n            
        param - num_param x dim_param numpy array \n
        ------------------------------------------------------ \n  
        Returns: \n
        param_trans - num_param x dim_param numpy array of the transformed param
        '''
       
        param_trans=param.copy() 
        for  index, value  in np.ndenumerate(param):
            col=index[-1]
            if(self.restriction[col]=='pos'):
                param_trans[index]=self.transform_positive(value)
            elif(self.restriction[col]=='01'):
                param_trans[index]=self.transform_01(value)
        
        return param_trans        

    def transform_back_param(self, param_trans):
        '''            
        Transforms back param_trans based on the restrictions   
        in res. \n 
        ------------------------------------------------------ \n        
        Input:  \n           
        param_trans - num_param x dim_param numpy array \n 
        ------------------------------------------------------ \n        
        Returns:\n 
        param - num_param x dim_param numpy array of the transformed param
        ''' 
        
        param=param_trans.copy()
        for  index, value  in np.ndenumerate(param_trans):
            col=index[-1]
            if(self.restriction[col]=='pos'):
                  param[index]=self.transform_back_positive(value)
            elif(self.restriction[col]=='01'):
                param[index]=self.transform_back_01(value)

           
        return param    
        
    def gradient(self,param_trans):
        g=np.zeros(self.dim_param)
        for i in range(self.dim_param):
            if(self.restriction[i]==''):
                g[i]=1
            elif(self.restriction[i]=='pos'):
                g[i]=np.exp(param_trans[i])
            else:
                g[i]=np.exp(param_trans[i])/np.power(1+np.exp(param_trans[i]),2)
        return g
        
        
class NormalPrior(Prior):
    def __init__(self,restriction,mu=0,sigma=1):
        Prior.__init__(self,restriction)
        self.mu=mu
        self.sigma=sigma
    
    def log_prior(self,param):
        param_trans=self.transform_param(param)
#        print('param_trans {}'.format(param_trans))
#        print('log normal pdf {}'.format(norm.logpdf(param_trans)))
        return np.sum(norm.logpdf(param_trans,loc=self.mu, scale=self.sigma),axis=1)

    def sample_prior(self,num_param): 
        param_trans=norm.rvs(loc=self.mu, scale=self.sigma,size=(num_param,self.dim_param))
        return self.transform_back_param(param_trans)
    
class GARCHPrior(Prior):
    def log_prior(self,param):
        log_p=np.log(lognorm.pdf(param[:,0],1,scale=0.02))
        ab=np.concatenate((param[:,1:3],np.zeros((len(param),1))),axis=1)        
        ab[:,2]=np.ones(len(param))-np.sum(param[:,1:3],axis=1)
#        print('ab {}'.format(ab))      
#        print('dirichlet {}'.format(dirichlet.pdf(ab[0],[4,9,3])))        
#        print('dirichlet all {}'.format(dirichlet.pdf(np.transpose(ab),[4,9,3])))
        log_dirichlet=np.zeros(len(param))
        for i in range(0,len(param)):
            if(np.all(ab[i]>0)):
                log_dirichlet[i]=dirichlet.pdf(ab[i],[3,54,3])
            else:
                log_dirichlet[i]=-np.Inf
#        log_p=log_p+np.log(dirichlet.pdf(np.transpose(ab),[4,9,3]))
        log_p=log_p+log_dirichlet 
        return log_p      
        
    def sample_prior(self,num_param):
        const=lognorm.rvs(1,scale=0.02, size=(num_param,1))        
        ab=dirichlet.rvs([3,54,3], size=num_param)

        return np.concatenate((const,ab[:,0:2]),axis=1)
        
class GARCHUniformPrior(Prior):
    def log_prior(self,param):
        log_p=np.zeros(len(param))        
        return log_p
    def sample_prior(self,num_param):  
        const=uniform.rvs(loc=0.001,scale=0.1,size=(num_param,1))
        a=uniform.rvs(loc=0.01,scale=0.5,size=(num_param,1))
        b=uniform.rvs(loc=0.5,scale=0.45,size=(num_param,1))
        return np.concatenate((const,a,b),axis=1) 
    

if __name__ == "__main__":
    '''
    Testing the Prior class
    '''
    
    restriction=['', '01', 'pos']    
    normal_prior=NormalPrior(restriction)
    
    #Check dim_param
    print('number of parameters: {}'.format(normal_prior.dim_param))    
    
    #One dimensional array    
    param_test1=np.array([0, 0.5, 0.5])
    #two dimensionl array    
    param_test2=np.array([[0,0.2,0.5],[0.1,0.2,0.7]])
    
    #Transforming the parameter
#    param_trans_test1=normal_prior.transform_param(param_test1)
    param_trans_test2=normal_prior.transform_param(param_test2)
    #Transforming back the parameter    
#    param_trans_back_test1=normal_prior.transform_back_param(param_trans_test1)
    param_trans_back_test2=normal_prior.transform_back_param(param_trans_test2)

    print('param_test2:\n{}'.format(param_test2))
    print('param_trans_back_test2:\n{}'.format(param_trans_back_test2))
    
    par=np.array([[1,0.3,1],[3,0.5,3]])
  
    print('test log_prior: {}'.format(normal_prior.log_prior(par)))
    print('test sample_prior: {}'.format(normal_prior.sample_prior(10)))

    import matplotlib.pyplot as plt   
    rest=['pos' , 'pos', '01']
    garch_prior=GARCHPrior(rest)
    param=garch_prior.sample_prior(1000)
    print('param sample {}'.format(param))
    print('GARCH log prior {} '.format(garch_prior.log_prior(param)))
    plt.plot(param[:,1],param[:,2], 'bo')
    plt.show()
    
    print('dirichlet {}'.format(dirichlet.pdf([0.2,0.4,0.4],[4,9,3]))) 