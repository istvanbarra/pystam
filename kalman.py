# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:54:03 2016

@author: istvan
"""
import numpy as np

def system_matrices(param):
    ssf=dict()
    ssf['Z']=np.array([1,0])
    ssf['H']=np.array([param[0]])
    ssf['T']=np.array([[1,1],[0,1]])
    ssf['Q']=np.array([[param[1],0],[0,param[2]]])
    
    return ssf
    
class SystemMatrices():
        def set_matrices(self,param):
            self.ssf=dict()
            self.ssf['Z']=np.array([[1,0]])
            self.ssf['Z_i']=np.array([[0,-1]])
            self.ssf['H']=np.array([param[0]])
            self.ssf['H_i']=np.array([-1])            
            self.ssf['T']=np.array([[1,1],[0,1]])
            self.ssf['T_i']=np.array([[1,-1],[0,2]]) 
            self.ssf['Q']=np.array([[param[1],0],[0,param[2]]])

            self.ssf['a']=np.array([[0],[0]])
            self.ssf['P']=np.array([[1,0],[0,1]])
            
        def check_dimensions(self):
            self.obs_dim=self.ssf['Z'].shape[0]
            self.state_dim=self.ssf['Z'].shape[1]
            print('state dimension: {}'.format(self.state_dim))
            print('obs dimension: {}'.format(self.obs_dim))
            
        def check_time_variation(self):
            self.ssf_time_varying=dict()      
            self.matrices=list(self.ssf.keys())            
            for m in self.matrices:
                index=(m.split('_')[-1]=='i')
                mat=m.split('_')[0]
                if(index and np.any(np.not_equal(self.ssf[m], -1))):
                    self.ssf_time_varying[mat]=self.ssf[m][self.ssf[m]!=-1].ravel()

            self.time_varying_matrices=self.ssf_time_varying.keys()
            print(self.ssf_time_varying)
        def update_matrices(self,x):
            for m in self.time_varying_matrices:
               self.ssf[m][self.ssf[m+'_i']!=-1]=x[self.ssf_time_varying[m]] 
   
def kalman_filter(param,x,sys_matrices,y):
    
    #number of observations
    y_num=len(y)
        
    #create system matrices    
    sys_matrices.set_matrices(param)
    #check the dimensions of the system matrices   
    sys_matrices.check_dimensions()
    #check which matrices are timevarying
    sys_matrices.check_time_variation()
    
    #initialize matrices
    filtered_mean=np.zeros((y_num,sys_matrices.state_dim))
    filtered_var=np.zeros((y_num,sys_matrices.state_dim*sys_matrices.state_dim))
    predicted_mean=np.zeros((y_num,sys_matrices.state_dim))
    predicted_var=np.zeros((y_num,sys_matrices.state_dim*sys_matrices.state_dim))
    K=np.zeros((y_num,sys_matrices.state_dim*sys_matrices.obs_dim))
    F_inv=np.zeros((y_num,sys_matrices.obs_dim*sys_matrices.obs_dim))
    v=np.zeros((y_num,sys_matrices.obs_dim))  
    
    
    #recursion
    for i in range(0,y_num):
        sys_matrices.update_matrices(x[i])
        if(i==0):
            predicted_mean_t=sys_matrices.ssf['a']
            predicted_var_t=sys_matrices.ssf['P']
        
        #rewriting the whole stuff using prediction only
        #Check ~np.isnan(y) if this is empty than skipp update and do prediction other wise use then on Z[~np.isnan(y),:]        
        nonmissing=~np.isnan(y)        
        
        if(np.all(nonmissing)==False):
            filtered_mean_t=predicted_mean_t
            filtered_var_t=predicted_var_t
            
            
        else:
            #update
            y_t=y[i][nonmissing]
            Z_t=sys_matrices.ssf['Z'][nonmissing,:]
            H_t=sys_matrices.ssf['H'][nonmissing,nonmissing]
        
        
            v_t=y_t-np.dot(Z_t,predicted_mean_t) 
            print('v_t: {}'.format(v_t))
            F_t=np.dot(Z_t,np.dot(predicted_var_t,np.transpose(Z_t)))+H_t 
            print('F_t: {}'.format(F_t))
            M_t=np.dot(predicted_var_t,np.transpose(Z_t)) 
            F_inv_t=np.linalg.pinv(F_t)
            filtered_mean_t=predicted_mean_t+np.dot(M_t,np.dot(F_inv_t,v_t))
            filtered_var_t=predicted_var_t-np.dot(M_t,np.dot(F_inv_t,np.transpose(M_t)))
            
            #store stuff
        predicted_mean[i,:]=predicted_mean_t.ravel()
        predicted_var[i,:]=predicted_var_t.ravel()
        filtered_mean[i,:]=filtered_mean_t.ravel()
        filtered_var[i,:]=filtered_var_t.ravel()
        K[i,:]=np.dot(M_t,F_inv_t).ravel()
        F_inv[i,:]=F_inv_t.ravel()
        v[i,:]=v_t.ravel()        
        
        T_t=sys_matrices.ssf['T']
        Q_t=sys_matrices.ssf['Q']
        #predict
        predicted_mean_t=np.dot(T_t,filtered_mean_t)
        predicted_var_t=np.dot(T_t,np.dot(filtered_var_t,np.transpose(T_t)))+Q_t
        
     
        print('F_inv at {} is : {}'.format(i,F_inv))
        print('v at {} is : {}'.format(i,v))
        return filtered_mean,filtered_var,predicted_mean,predicted_var,K,F_inv,v
   
def kalman_smoother(param,x,sys_matrices,predicted_mean,predicted_var,K,F_inv,v):
    #number of observations
    y_num=len(predicted_mean)
        
    #create system matrices    
    sys_matrices.set_matrices(param)
    #check the dimensions of the system matrices   
    sys_matrices.check_dimensions()
    #check which matrices are timevarying
    sys_matrices.check_time_variation()
    
    #initialize matrices
    smoothed_mean=np.zeros((y_num,sys_matrices.state_dim))
    smoothed_var=np.zeros((y_num,sys_matrices.state_dim*sys_matrices.state_dim))
        
    
    for i in range(y_num-1,-1,-1):
        sys_matrices.update_matrices(x[i])
        K_t=np.reshape(K[i,:],(sys_matrices.state_dim,sys_matrices.obs_dim))
        F_inv_t=np.reshape(F_inv[i,:],(sys_matrices.obs_dim,sys_matrices.obs_dim))
        v_t=np.reshape(F_inv[i,:],(sys_matrices.obs_dim,1))
        L_t=sys_matrices.ssf['T']-np.dot(K_t,sys_matrices.ssf['Z'])
        predicted_mean_t=np.reshape(predicted_mean[i,:],(sys_matrices.state_dim,1))
        predicted_var_t=np.reshape(predicted_var[i,:],(sys_matrices.state_dim,sys_matrices.state_dim))
        
        if(i==(y_num-1)):
            r_t=np.dot(np.transpose(sys_matrices.ssf['Z']),np.dot( F_inv_t,v_t)) 
            N_t=np.dot(np.transpose(sys_matrices.ssf['Z']),np.dot( F_inv_t,sys_matrices.ssf['Z']))
        else:
            r_t=(np.dot(np.transpose(sys_matrices.ssf['Z']),np.dot( F_inv_t,v_t)) 
                + np.dot(np.transpose(L_t),r_t))
            N_t=(np.dot(np.transpose(sys_matrices.ssf['Z']),np.dot( F_inv_t,sys_matrices.ssf['Z']))
                + np.dot(np.transpose(L_t),np.dot(N_t,L_t)))
        
        
        smoothed_mean_t=predicted_mean_t-np.dot(predicted_var_t,r_t)
        smoothed_var_t=predicted_var_t-np.dot(predicted_var_t,np.dot(N_t,predicted_var_t))
     
        #store stuff
        smoothed_mean[i,:]=smoothed_mean_t.ravel()
        smoothed_var[i,:]=smoothed_var_t.ravel()
    
        return smoothed_mean,smoothed_var
        
def kalman_log_likelihood(param,x,sys_matrices,y):
        
    
#ssf=SystemMatrices()
#ssf.set_matrices([0.1,0.2,0.3])
#ssf.check_time_variation()
#ssf.check_dimensions()
#ssf.update_matrices(np.array([1,2,3]))
#print(ssf.ssf)


#Check filtering
y=np.array([1,3,5])
x=np.array([[1,1,1], [2,2,2],[3,3,3] ])
ssf=SystemMatrices()
kalman_filter([0.1,0.2,0.3],x,ssf,y)




