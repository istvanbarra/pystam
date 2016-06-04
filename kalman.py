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
        def set_matrices(self,param,x):
            self.Z=np.array([1,0])
            self.H=np.array([param[0]])
            self.T=np.array([[1,1],[0,1]])
            self.Z_i=np.array([0,-1])
            self.Q=np.array([[param[1],0],[0,param[2]]])
            self.a=np.array([0,0])
            self.P=np.array([[1,0],[0,1]])
        def check_timevarying(self):
            try: 
                print(self.Z_i[self.Z_i!=-1])
            except AttributeError:
                print("Z is not timeverying")
            self.Z_i[self.Z_i!=-1]


   
def kalman_filter(param,x,sys_matrices,y):

    num_y=len(y)
    
    for i in range(0,num_y):
        sys_matrices.set_matrices(param,x[i])
        
        

   
ssf=SystemMatrices()
ssf.set_matrices([0.1,0.2,0.3],[])
ssf.check_timevarying()
#


a=np.array([[1,2],[3,4]])
b=np.array([5,6,7,8] )
index=np.array([[1,0],[3,2]])
print(np.where(index!=-1))
print((index!=-1).ravel())
a[index!=-1]=b[index[index!=-1].ravel()]
print(a)

