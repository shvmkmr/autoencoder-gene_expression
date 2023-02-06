#!/usr/bin/env python
# coding: utf-8

# Imports


from tensorflow.python.client import device_lib
import time
import os
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from sys import getsizeof as m_size
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def total_zero(df):
    return ((df == 0).astype(int).sum(axis=1)).sum()


def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# In[ ]:






# In[7]:


st=time.time()
data=pd.read_csv('Test.csv',index_col=0)
end=time.time()
print(end-st)


# In[8]:




labels=data['Classes']


# In[10]:


data.drop('Classes',axis=1,inplace=True)


# In[11]:


total_genes=data.columns


# In[12]:



# In[14]:


x=data.to_numpy()
print(x.shape)


# In[32]:


G=get_available_gpus()
print(G)
G=len(G)
print(G)


# In[ ]:


model=load_model('sample3.h5')
print(model.summary())
# In[16]:


print("Score neural network")
pred = model.predict(x)






print(mse(pred,x))



