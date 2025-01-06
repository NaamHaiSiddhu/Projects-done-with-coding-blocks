#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing import image
from pathlib import Path
import matplotlib.pyplot as ppplot
import os
#from langchain.embeddings import HuggingFaceEmbeddings
#embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


# In[2]:


givenpath = Path(r"C:\Users\SDKH\ImagesofPokemonsT")
alldirectory = givenpath.glob("*")
imgs = []
imgclass = []
imgmappingclass = {"Abra": 0, "Bellsprout": 1, "Clefairy": 2, "Pikachu": 3}
classmappingimg = {0: "Abra", 1: "Bellsprout", 2: "Clefairy" , 3: "Pikachu"}
imgpths = []


# In[3]:


for j in alldirectory:
    imgi = str(j).split("\\")[-1]
    print(imgi)
    countsi = 0

    for imgj in j.glob("*.jpg"):
        imgij = image.load_img(imgj, target_size = (30, 30))
        img = image.img_to_array(imgij)
        imgs.append(img)
        imgclass.append(imgmappingclass[imgi])
        countsi += 1

    print(countsi)
        


# In[4]:


import numpy as nu


# In[5]:


print(len(imgs))
print(len(imgclass))


# In[6]:


print(imgclass)


# In[7]:


import random
random.seed(20)


# In[8]:


XI = nu.array(imgs)
YI = nu.array(imgclass)


# In[9]:


print(XI.shape)
print(YI.shape)


# In[10]:


from sklearn.utils import shuffle
XI, YI = shuffle(XI, YI, random_state = 1)
XI = XI/255.0


# In[11]:


def imageportrait(imgi, imgclass):
    ppplot.title(classmappingimg[imgclass])
    ppplot.imshow(imgi)
    ppplot.show()


# In[12]:


for j in range(20):
    imageportrait(XI[j] , YI[j])


# In[14]:


splitsc = int(XI.shape[0]*0.75)
XNI = nu.array(XI)
YNI = nu.array(YI)
XI, YI = XNI[:splitsc, :], YNI[:splitsc]
XIT, YIT = XNI[splitsc:, :], YNI[splitsc:]
print(XI.shape, YI.shape)
print(XIT.shape, YIT.shape)


# In[15]:


def softmax(Z):
    smz = nu.exp(Z)
    smr = smz/nu.sum(smz, axis=1, keepdims = True)
    return smr
    
class ThreeLPArch:
    def __init__(self, isiz, archlayerneuroncount, coutsiz):
        nu.random.seed(3)
        DICT = {}
        DICT["VFirst"] = nu.random.randn(isiz, archlayerneuroncount[0])
        DICT["CFirst"] = nu.zeros((1, archlayerneuroncount[0]))
        DICT["VSec"] = nu.random.randn(archlayerneuroncount[0], archlayerneuroncount[1])
        DICT["CSec"] = nu.zeros((1, archlayerneuroncount[1]))
        DICT["VThird"] = nu.random.randn(archlayerneuroncount[1], coutsiz)
        DICT["CThird"] = nu.zeros((1,coutsiz))
        self.DICT = DICT
        
    def inpropforwardc(self, X):
        VFirst, VSec, VThird = self.DICT["VFirst"], self.DICT["VSec"], self.DICT["VThird"]
        CFirst, CSec, CThird = self.DICT["CFirst"], self.DICT["CSec"], self.DICT["CThird"]
        ZFIRST = nu.dot(X, VFirst) + CFirst
        CTVFIRST = nu.tanh(ZFIRST)
        ZSEC = nu.dot(CTVFIRST, VSec) + CSec
        CTVSEC = nu.tanh(ZSEC)
        ZTHIRD = nu.dot(CTVSEC, VThird) + CThird
        COUT = softmax(ZTHIRD) 
        self.setofacts = (CTVFIRST, CTVSEC, COUT)
        return COUT
    def inprobackwardc(self, X, Y, rat = 0.02):
        VFirst, VSec, VThird = self.DICT["VFirst"], self.DICT["VSec"], self.DICT["VThird"]
        CFirst, CSec, CThird = self.DICT["CFirst"], self.DICT["CSec"], self.DICT["CThird"]
        CTVFIRST, CTVSEC, COUT = self.setofacts
        sThird = COUT - Y
        s = X.shape[0]
        swlthird = nu.dot(CTVSEC.T, sThird)
        sblthird = nu.sum(sThird, axis = 0)/float(s)
        
        sSec = (1- nu.square(CTVSEC))*nu.dot(sThird, VThird.T)
        swlsec = nu.dot(CTVFIRST.T, sSec)
        sblsec = nu.sum(sSec, axis = 0)/float(s)

        sFirst = (1- nu.square(CTVFIRST))*nu.dot(sSec, VSec.T)
        swlfirst = nu.dot(X.T, sFirst)
        sblfirst = nu.sum(sFirst, axis = 0)/float(s)

        self.DICT["VFirst"] -= rat*swlfirst
        self.DICT["CFirst"] -= rat*sblfirst

        self.DICT["VSec"] -= rat*swlsec
        self.DICT["CSec"] -= rat*sblsec

        self.DICT["VThird"] -= rat*swlthird
        self.DICT["CThird"] -= rat*sblthird

    def hypothesiz(self, X):
        YHAT = self.inpropforwardc(X)
        return nu.argmax(YHAT, axis = 1)
    def analysis(self):
        VFirst, VSec, VThird = self.DICT["VFirst"], self.DICT["VSec"], self.DICT["VThird"]
        CTVFIRST, CTVSEC, COUT = self.setofacts
        print("First set of Weights ", VFirst.shape)
        print("First set of acts", CTVFIRST.shape)
        print("Second set of Weights ", VSec.shape)
        print("Second set of acts", CTVSEC.shape)
        print("Final set of Weights ", VThird.shape)
        print("Final set of acts", COUT.shape)


# In[16]:


def catentropycross(Y_HOT, h):
    cec = -nu.mean(Y_HOT* nu.log(h))
    return cec
def yoht(Y, sizeability):
    sy = Y.shape[0]
    YOHT = nu.zeros((sy, sizeability))
    YOHT[nu.arange(sy), Y] = 1
    return YOHT 


# In[25]:


def trainingdatasts(XS, YS, threelparch, iters, rat, history = True):
    collectionofcosts = []
    distinctcs = 2
    Y_HOTS = yoht(YS, 4)

    for i in range(iters):
        YHAT = threelparch.inpropforwardc(XS)
        cost = catentropycross(Y_HOTS, YHAT)
        collectionofcosts.append(cost)
        threelparch.inprobackwardc(XS, Y_HOTS, rat)

        if (history):
            print("Iteration %d Cost %.4f"%(i, cost))

    return collectionofcosts


# In[26]:


threelparch = ThreeLPArch(isiz = 2700, archlayerneuroncount = [90, 40], coutsiz = 4)


# In[27]:


print(XI.shape)


# In[28]:


XI = XI.reshape(XI.shape[0], -1)
print(XI.shape)
XIT = XIT.reshape(XIT.shape[0], -1)
print(XIT.shape)


# In[30]:


costs = trainingdatasts(XI, YI, threelparch, 200, 0.0018)


# In[31]:


from matplotlib import pyplot as ppplot


# In[32]:


ppplot.style.use("seaborn-v0_8")
ppplot.title("Cost vs iter")
ppplot.plot(costs)
ppplot.show()


# In[37]:


def scorsc(XS, YS, threelparch):
    resultsc = threelparch.hypothesiz(XS)
    classacc = nu.sum(resultsc == YS)/YS.shape[0]
    return classacc


# In[38]:


print("Accuracy Score is : %.4f"% scorsc(XI, YI, threelparch))
print("Accuracy Score is : %.4f"% scorsc(XIT, YIT, threelparch))


# In[39]:


from sklearn.metrics import classification_report


# In[41]:


print(classification_report(threelparch.hypothesiz(XI), YI))


# In[42]:


from sklearn.metrics import confusion_matrix
from visualize import plot_confusion_matrix


# In[43]:


resultsc = threelparch.hypothesiz(XI)
confusionmatrix = confusion_matrix(resultsc, YI)
print(confusionmatrix)


# In[44]:


plot_confusion_matrix( confusionmatrix, classes = [ "Abra", "Bellsprout", "Clefairy", "Pikachu"] , title = "Confusion Matrix for Pokemons Fours")


# In[45]:


resultsct = threelparch.hypothesiz(XIT)
confusionmatrixt = confusion_matrix(resultsct, YIT)
print(confusionmatrixt)


# In[46]:


print(classification_report(threelparch.hypothesiz(XIT), YIT))


# In[48]:


plot_confusion_matrix( confusionmatrixt, classes = [ "Abra", "Bellsprout", "Clefairy", "Pikachu"] , title = "Confusion Matrix for Pokemons Fours Test")


# In[50]:


for j in range(YI.shape[0]):
    if YI[j] != resultsc[j]:
        imageportrait(XI[j].reshape(30, 30, 3), YI[j])
        print("Hypothesis %d %s" % (j, classmappingimg[resultsc[j]])) 


# In[ ]:




