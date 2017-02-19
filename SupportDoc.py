# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:49:23 2016

@author: ading
"""

#import re
import numpy as np
import random



## generate synthetic data
def evalution():

    Xts = np.zeros((57*6,57*6))
    
    for i in range(57*6):
        Xts[i,i*6%(57*6)+i/57] = 1
    
    Xts = np.reshape(Xts, (57*6,57,6))
    
    return Xts
    
    
## calculate probabilty
def CalProb(pLabel):
    m,d = pLabel.shape
    prob = np.zeros(m)
    for i in range(m):
        a = pLabel[i,:]
        prob[i] = np.exp(a[d-1])/np.sum(np.exp(a))
    return prob
    
## read the data
def preprocess():
    
    ### read negative data
    file = open("part-00000-13") 
    
    date = []
    apath = []
    for line in file.xreadlines():
        if len(line)>0:
            path = line.strip().split(',')  
            if path!=['']:
                apath.append(path)
                plen = len(path)
                for j in range(plen-1):
                    ## store the date
                    dt = path[j].split(';')[0]
                    if dt not in date:
                        date.append(dt)
                                          
    
    file.close()
    
    date.sort() ## sort the date
    
    num = len(apath)
    Xn = np.zeros((num,57,6)) ## create feature matrix
    
    for i in range(num):
        path = apath[i]
        for j in range(len(path)):
            ## index the location of current date
            idx = date.index(path[j].split(';')[0])
            ## tp value
            tp = int(path[j].split(';')[1])
            Xn[i,idx,tp]+=1
    ## build labels for negative data
    ## 10 means negative; 01 means positive
    yn = np.zeros((2,num))
    yn[0,:] = np.ones((1, num))
    
    
    ## read positive data
    file = open("part-00000-14") 
    
    date = []
    apath = []
    for line in file.xreadlines():
        if len(line)>0:
            path = line.strip().split(',')  
            if path!=['']:
                apath.append(path)
                plen = len(path)
                for j in range(plen-1):
                    dt = path[j].split(';')[0]
                    if dt not in date:
                        date.append(dt)
                                          
    
    file.close()
    
    date.sort() ## sort the date
    
    num1 = len(apath)
    Xp = np.zeros((num1,57,6)) ## create feature matrix
    
    for i in range(num1):
        path = apath[i]
        for j in range(len(path)):
            idx = date.index(path[j].split(';')[0])
            tp = int(path[j].split(';')[1])
            #print tp
            if tp!=-1:
                Xp[i,idx,tp]+=1
    
    yp = np.zeros((2,num1))
    yp[1,:] = np.ones((1,num1))
    
    ## random sampling negative data
    random.seed(0)
    ran_a = range(num)
    ran_b = random.sample(ran_a, num1)
    yn2 = yn[:,ran_b]
    Xn2 = Xn[ran_b,:,:]
    
    #print Xn2.shape
    X = np.concatenate((Xn2,Xp),axis=0)
    y = np.hstack((yn2,yp)).transpose()
    
    return X, y
    
    
Xtr, ytr = preprocess()
     
#print len(Xtr)