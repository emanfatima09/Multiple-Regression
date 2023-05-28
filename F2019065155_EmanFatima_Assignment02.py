#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\92318\\OneDrive\\Desktop\\ex1data2.txt',header=None)

df.columns = ['X1','X2','y']
X = df[['X1','X2']]
y = df['y']
m = len(y)

mu = X.mean()
sigma = X.std()


def featureNormalize(X):
    X=(X-np.array(X.mean()))/np.array((X.std()))
    return X


def costFunction(X,y,theta):
    J=1/(2*m)*sum(((X@theta)-y)**2)
    return J


alpha = 0.03

X=featureNormalize(X)
X.insert(0,'X0',np.ones(m))
theta = np.zeros(len(X.columns))
costFunction(X,y,theta)
J_hist=[]
z=[]
for g in range(0,400):
    z.append(g)
    
def gradientDescent(X,y,theta,alpha,iterations):
    for i in z:
        theta =  theta - (alpha * (1/m) *(((X@theta)-y)@X))
        value=costFunction(X,y,theta)
        J_hist.append(value)
    return J_hist,theta

hist,theta_opt = gradientDescent(X,y,theta,alpha,400)

plt.plot(hist)

print('optimized theta without normal equation',theta_opt)

def normalEquation(X,y):
    a_X=X.T
    n_X=a_X@X
    new_theta = np.zeros(len(X.columns))
    new_theta=np.linalg.inv(n_X)@(a_X@y)
    return new_theta

theta_opt=normalEquation(X,y)
print(normalEquation(X,y))

def predict(ary,theta):
    pred = pd.DataFrame(np.array(ary))
    pred = featureNormalize(pred.T)
    ary0 = np.ones(1,dtype=int)
    ary1 = np.array(pred,dtype=int)
    pred = np.column_stack([ary0,ary1])
    return pred@np.array(theta_opt)

predict(np.array([1534,3]),theta_opt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




