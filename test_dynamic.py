# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:04:11 2016

@author: istvan
"""

from pystam.estimation.maxlike import MaximumLikelihood
from pystam.estimation.ibis import IBISDynamic
from pystam.model.model import GARCH 
from pystam.prior.prior import NormalPrior,GARCHPrior,GARCHUniformPrior

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



state=[{'a':np.array([1,2]),'P':np.array([[1,2],[3,4]])}, [np.array([1,2]),np.array([[1,2],[3,4]])] ]
print(state[0]['P'])
#np.random.seed(1)
#
#def generate_garch(param,num_obs):
#    y=np.zeros(num_obs)
#    state=np.zeros(num_obs)    
#    for t in range(0,num_obs):
#        if(t==0):
#            state[t]=np.divide(np.take(param,0,-1),(1- np.take(param,1,-1)-np.take(param,2,-1)))
#        else:
#            state[t]=np.take(param,0,-1)+np.take(param,1,-1)*y[t-1]*y[t-1]+np.take(param,2,-1)*state[t-1]
##        print('state at {} is {}'.format(t,state[t]))    
#        y[t]=norm.rvs(scale=np.sqrt(state[t])) 
#    return y,state
#    
#num_sim=7000
#y,state=generate_garch([0.01,0.04,0.9],num_sim)
#
##plt.plot(y*y)
##plt.plot(state)
##plt.show()
#
## Paramter restrictions.
#restriction=['pos', 'pos', '01']    
## Creating a NormalPrior prior class instance.
#normal_prior=NormalPrior(restriction)
#garch_prior=GARCHPrior(restriction)
#garch_uniform_prior=GARCHUniformPrior(restriction)
#garch=GARCH()
## Creating a MaximumLikelihood class.
#garch_maxlike=MaximumLikelihood(garch,garch_prior)
## Estimating a regression model on the simulated data using maximum likelihood.
#x=np.zeros(num_sim)
##print(garch.log_likelihood(np.array([0.01,0.03,0.95]),y,x)[0])
#
##llx,lly=garch_maxlike.log_likelihood_profile(y,x,np.array([0.01,0.04,0.9]),1,0.001,0.08,200)
##plt.plot(llx,lly)
##plt.show()
#garch_maxlike.estimate(y,x)
#print('Maximum likelihood estimates:\n {}'.format(garch_maxlike.param))
#print('Maximum likelihood estimates:\n {}'.format(garch_maxlike.low))
#print('Maximum likelihood estimates:\n {}'.format(garch_maxlike.high))
#
#
#garch_ibis=IBISDynamic(1000,garch,garch_uniform_prior)
#plt.plot(garch_ibis.param[:,1],garch_ibis.param[:,2],'ro')
#plt.show()
## Estimating a regression model on the simulated data using IBIS. 
#garch_ibis.estimate(y, x)
#print('Iterated batch importance sampling estimates:\n {}'.format(garch_ibis.median[-1]))
#plt.plot(garch_ibis.median[:,2])
#plt.show()
#plt.plot(garch_ibis.ess)
#plt.show()
#print(garch_ibis.acceptance_rate)
#plt.plot(garch_ibis.acceptance_rate)
#plt.show()