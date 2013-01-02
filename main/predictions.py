__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

import csv as csv
import numpy as np
import scipy
from numpy import *
from sklearn.ensemble import RandomForestClassifier

set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)

from PrepareTitanicData import titandata

data=titandata("train") #(891,8) array
testdata=titandata("test") #(418,7) array
test8 = titandata("test8") #(418,8) array


def genderpred(dataset):
    new = []
    for row in dataset: # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
        if (row[2] == 1):
            new.append(1)
        else:
            new.append(0)
    return np.array(new) # convert to array

def f3sm12pred(dataset):
    new = []
    for row in dataset: # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
        if (row[2] == 1) and not( (row[1] == 3) & (row[7] == 0) ):
            new.append(1)
        elif (row[2] == 0) and (row[3] <= 10) and not(row[1] == 3):
            new.append(1)
        else:
            new.append(0)
    return np.array(new) # convert to array

def newpred(dataset): #3rd class males age<=12 with sibsp = 0,1 and 3rd class females S age<=8 with sibsp=0,1 make it
    new = []
    for row in dataset: # [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
        if (row[2] == 1) and not( (row[1] == 3) & (row[7] == 0) ):
            new.append(1)
        elif (row[2] == 0) and (row[3] <= 12) and not(row[1] == 3):
            new.append(1)
        elif (row[2] == 0) and (row[3] <= 12) and (row[1] == 3) and (0 <= row[4] <= 1 ):
            new.append(1)
        elif (row[2] == 1) and (row[3] <= 8) and (row[1] == 3) and (row[7] == 0) and  (0 <= row[4] <= 1 ):
            new.append(1)
        else:
            new.append(0)
    return np.array(new) # convert to array

def predicttrain(pred):
    comptrain = pred-data[0::,0] #compare pred with reality
    #print np.nonzero(comptrain) #This shows which elements were predicted wrong
    numwrong = sum(abs(comptrain[0::])==1)
    score = 1 - float(numwrong) / float(np.size(comptrain))
    return score

def comparepreds(pred1,pred2): #takes in predictions in form of 418,1 array
    dif = pred1-pred2 #+1,0,-1 for each passenger
    dispass = np.nonzero(dif)[0] # passengers numbers for which the predictions disagree.
    # the [0] due to formatting of nonzero function
    datadispass = testdata[dispass] # the data for those passengers
    numdifpass = np.size(dispass) # how many of them there are
    datadp = np.zeros([numdifpass,8]) # placeholder
    for x in xrange(numdifpass):
        dp = dispass[x] # passenger index for each disagreement
        datadp[x] = insert(datadispass[x],0,dif[dp]) #insert the value of dif into the 0th column
    print "The number of disagreements: %i" %numdifpass
    print "The disagreements: +1 means (pred1=1, pred2=0) and -1 means (pred1=0 and pred2=1)"
    print datadp
    return



newcsv = csv.writer(open('../newpredpy.csv','wb'))
newpredict = newpred(test8)
for x in xrange(418):
    if newpredict[x]==0:
        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if newpredict[x]==1:
        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.
