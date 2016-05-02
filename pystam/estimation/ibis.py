# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:23:02 2016

@author: istvan
"""
import numpy as np
import math


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == np.shape(sigma):
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.pinv(sigma)    
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def log_multi_norm(x, mu,sigma):
    num=len(x)
    result=np.zeros((num,1))
    for i in range(0,num):
        result[i]=np.log(norm_pdf_multivariate(x[i], mu, sigma))

    return result
    
def independent_proposal(param,prior):
    '''
    Independent multivariate normal proposal \n 
    ------------------------------------------- \n
    Inputs: \n
    param - num_param x dim_param numpay array \n
    prior - an instance of the prior class
    ------------------------------------------ \n
    Results: \n
    param_new - proposed parameters num_param x dim_param numpay array \n
    log_proposal_dens - logarithm of the proposal density at the old parameters num_param x 1 numpay
    log_proposal_dens_new - logarithm of the proposal density at the proposed parameters num_param x 1 numpay
    
    '''
    
    num_param=np.shape(param)[0]
    ''' Construct proposal '''
    param_trans=prior.transform_param(param)
    mean_trans=np.mean(param_trans,0)
    var_trans=np.cov(param_trans, y=None,rowvar=0)
    param_trans_new= np.random.multivariate_normal(mean_trans,var_trans,num_param)
    param_new=prior.transform_back_param(param_trans_new)

    ''' Calculate proposal '''     
    log_proposal_dens=log_multi_norm(param_trans, mean_trans,var_trans)
    log_proposal_dens_new=log_multi_norm(param_trans_new, mean_trans,var_trans)    

    return param_new, log_proposal_dens, log_proposal_dens_new
    
        

def systemic_resampling(param_cum_w):
    '''
    Systemic resampling \n 
    ------------------------------------------- \n
    Inputs: \n
    param_cum_w - cumulative weights
    ------------------------------------------ \n
    Results: \n
    index - resampled indecies
    '''
    num=len(param_cum_w)
    index=np.zeros(num,dtype=int)
    u=np.random.uniform(0,1,1)
    k=0
    for i in range(0,num):
        p=(u+i)/num
        while(param_cum_w[k]<p):
            k=k+1
        index[i]=k
     
    return index  


def weighted_percentile(data, weights, percentile):
    '''
    Calculates weighted p in [0,100] percentile
    '''
    p=float(percentile)/100

    num_col=np.shape(data)[1]
    result=np.zeros(num_col )
    
    for i in range(0, num_col):
        ind=np.argsort(data[:,i])
        d=data[ind,i]
        w=weights[ind]
        sum_w=w[0]
        j=0;
        while( sum_w<p):
            j+=1
            sum_w+=w[j]
        
        if(j==0):
            result[i]=d[j]
        else:
            sum_w_prev=np.sum(w[:j])
            if(sum_w==p or d[j]==d[j-1]):
                result[i]=d[j]
            else:
                result[i]=d[j-1]+np.exp(np.log(p-sum_w_prev)+np.log(d[j]-d[j-1])-np.log(sum_w-sum_w_prev))
    return result  



class IBIS():
    '''
    Iterated bach importance sampling procedure \n 
    ------------------------------------------- \n
    Inputs: \n
    num_param - number of \theta particles \n
    data - data class \n 
    model - model class \n 
    param - param class  \n
    ------------------------------------------ \n
    Results
    '''
    
    def __init__(self, num_param, model, prior,resampling=systemic_resampling,proposal=independent_proposal):
        
        self.num_param=num_param        
        
        self.model=model
        self.prior=prior
        self.resampling=resampling
        self.proposal=proposal
                
        self.log_obs=np.zeros(num_param)
        self.param_log_w=np.zeros(num_param)
        self.param_log_likelihood=np.zeros(num_param)
        self.param_norm_w=np.zeros(num_param)
        
        self.param=self.prior.sample_prior(num_param)
        self.dim_param=np.shape(self.param)[1] 
        
       
        

    def resample_move(self,y,z):
        print 'Resample move step'
        ''' Resample '''
        param_cum_w=np.cumsum(self.param_norm_w,axis=0)
        index=self.resampling(param_cum_w)
        self.param=self.param[index]
        self.param_log_likelihood=self.param_log_likelihood[index]
        self.param_log_w=np.zeros(self.num_param)
        ''' Move '''
        ''' Proposal '''        
        param_new, log_proposal, log_proposal_new=self.proposal(self.param, self.prior)
        ''' Calculate prior '''
        log_prior=self.prior.log_prior(self.param)
        log_prior_new=self.prior.log_prior(param_new)

        
    
        ''' Calculate likelihood '''                            
        param_log_likelihoodNew=self.model.log_likelihood(param_new, y, z)          
        param_log_likelihoodNew=np.where(np.isnan(param_log_likelihoodNew)==0, param_log_likelihoodNew, -np.Inf)

         
        ''' Accept reject '''
        accept=np.zeros(self.num_param)
        log_u=np.log(np.random.uniform(0,1,self.num_param)) 
        for i in range(0,self.num_param):
            ''' Calculate acceptenc probability '''
            log_prob=(param_log_likelihoodNew[i]+log_prior_new[i]-log_proposal_new[i])-(self.param_log_likelihood[i]+log_prior[i]-log_proposal[i])
            log_prob=min(log_prob,1)
            if(log_u[i] <log_prob):
                self.param[i]=param_new[i]
                self.param_log_likelihood[i]=param_log_likelihoodNew[i]
                accept[i]=1
        return accept  
        
    def update(self,y,z):
        
        
        ''' Update weights '''
        log_obs=self.model.log_obs_dens(self.param,  y, z)
        log_obs=np.where(np.isnan(log_obs) == 0, log_obs, -np.Inf)
        self.param_log_w=self.param_log_w+log_obs
        self.param_log_likelihood=self.param_log_likelihood+log_obs
        
        ''' Calculate normalized weights '''
        self.param_norm_w=np.exp(self.param_log_w-np.amax(self.param_log_w,0))/np.sum(np.exp(self.param_log_w-np.amax(self.param_log_w,0)),0)
        
        

        

    def estimate(self, y, z):
        num_y=np.shape(y)[0]
        num_z=np.shape(y)[0]
        
        self.high=np.zeros((num_y,self.dim_param))
        self.low=np.zeros((num_y,self.dim_param))
        self.median=np.zeros((num_y,self.dim_param))
        self.ess=np.zeros(num_y)
        self.acceptance_rate=np.zeros(num_y) 
        self.marginal_log_likelihood=np.zeros(num_y) 
        
        
        if(num_y!=num_z):
            raise ValueError('Number of row in y ({}) and z ({}) has to be equal'.format(num_y,num_z))
   
        for t in range(0,num_y):
            ''' Update parameters '''
            self.update(y[t],z[t])

            ''' Store stuff '''
            self.ess[t]=1/np.sum(np.power(self.param_norm_w,2))        
            self.high[t,:]=weighted_percentile(self.param,self.param_norm_w,97.5)   
            self.low[t,:]=weighted_percentile(self.param,self.param_norm_w,2.5)  
            self.median[t,:]=weighted_percentile(self.param,self.param_norm_w,50)
            if(t==0):
                self.marginal_log_likelihood[t]=np.inner(self.log_obs,self.param_norm_w)
            else:
                self.marginal_log_likelihood[t]=self.marginal_log_likelihood[t-1]+np.inner(self.log_obs,self.param_norm_w)
            self.acceptance_rate[t]=np.nan
            
            ''' Resample move  if necessary '''     
            if(self.ess[t]<self.num_param/2):
                accept=self.resample_move(y[0:(t+1)],z[0:(t+1)])
                self.acceptance_rate[t]=np.mean(accept)   
    