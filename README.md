# pystam

```python
import numpy as np
from scipy.stats import norm

from pystam.estimation.ibis import IBIS
from pystam.estimation.maxlike import MaximumLikelihood
from pystam.model.model import LinearRegression
from pystam.prior.prior import NormalPrior

def generate_linear_regression(param,num_obs):
    """
    Generating num_obs observations from the a regression model using parameters
    in param.
    
    Input:
        param : 1D numpy array or list
            The first element is the constant, while the last element is the
            standard error of the error term 
        num_obs : integer
            The number of observations
        
    Output:
        y : 1D numpy array
            The simulated dependent variable            
        x : 2D numpy array
            The simulated regressors
    """
    
    # Number of regressors in the regression.  
    num_x=len(param)-1    

    # Generating the regressors.
    x=np.concatenate((np.ones((num_obs,1)),norm.rvs(size=(num_obs,num_x-1))),axis=1)

    # Generating the dependent variable.
    y=np.inner(x,param[0:num_x])+param[num_x]*norm.rvs(size=num_obs)    
    
    return y,x

# Generating data from the linear regression model.
y,x=generate_linear_regression([1,0.5,-0.5,2],1000)
# Creating a LinearRegression model class instance. 
regression=LinearRegression()
# Paramter restrictions. 
restriction=['', '', '', 'pos']    
# Creating a NormalPrior prior class instance.
normal_prior=NormalPrior(restriction)

# Creating an IBIS class using 1000 parameter particles.
regression_ibis=IBIS(1000,regression,normal_prior)
# Estimating a regression model on the simulated data using IBIS. 
regression_ibis.estimate(y, x)
print('Iterated batch importance sampling estimates:\n {}'.format(regression_ibis.median[-1]))

# Creating a MaximumLikelihood class.
regression_maxlike=MaximumLikelihood(regression,normal_prior)
# Estimating a regression model on the simulated data using maximum likelihood.
regression_maxlike.estimate(y, x)
print('Maximum likelihood estimates:\n {}'.format(regression_maxlike.param))
``` 


