#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2 as cvt
import matplotlib.pyplot as ppplot


# In[6]:


imagry = cvt.imread(r"C:\Users\SDKH\Downloads\icpaths\Cats\cat.jpg")
imagry = cvt.cvtColor(imagry, cvt.COLOR_BGR2RGB)
imagrys = imagry.shape
print(imagry.shape)


# In[7]:


ppplot.imshow(imagry)
ppplot.show()


# In[8]:


cumpixs = imagry.reshape((-1, 3))
print(cumpixs.shape)


# In[9]:


from sklearn.cluster import KMeans 


# In[10]:


all_colors = 18
rgbc = KMeans(n_clusters = all_colors)


# In[11]:


rgbc.fit(cumpixs)


# In[12]:


centersc = rgbc.cluster_centers_


# In[13]:


import numpy as nu
centersc = nu.array(centersc, dtype = "uint8")
print(centersc)


# In[14]:


j = 1
ppplot.figure(0, figsize=(27, 19))
clrs = []
for c in centersc:
    ppplot.subplot(1, 18, j)
    ppplot.axis("off")
    j += 1

    clrs.append(c)
    s = nu.zeros((80, 80, 3), dtype = "uint8")
    s[:, :, :] = c
    ppplot.imshow(s)
    
ppplot.show()


# In[15]:


imagryr = nu.zeros((259*194,3), dtype = "uint8")
print(imagryr.shape)


# In[16]:


clrs


# In[17]:


rgbc.labels_


# In[18]:


for j in range(imagryr.shape[0]):
    imagryr[j] = clrs[rgbc.labels_[j]]

imagryr = imagryr.reshape((imagrys))
ppplot.imshow(imagryr)
ppplot.show()


# In[ ]:





# In[ ]:




