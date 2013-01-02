
__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

import csv as csv
import numpy as np
import scipy
from numpy import *
from sklearn.ensemble import RandomForestClassifier

set_printoptions(suppress = True) # makes for nice printing without scientific notation
np.set_printoptions(linewidth=132)



# File PrepareTitanicData.py goes through the details of preparing the data.
# It converts sex to 1,0 for F,M, and 0,1,2,3 for city embarked "S","C","Q",""
# Also for no-age we put in placeholder 1000
# for no fare we put placeholder 1000
# Cabin, Ticket #, and name are deleted

from PrepareTitanicData import titandata

data=titandata("train") #(891,8) array
testdata=titandata("test") #(418,7) array
test8 = titandata("test8") #(418,8) array
# Call function titandata which takes an argument string, which must be either "train", "test", or "test8"
#print data[0:10]
#print testdata[0:10]
#print test8[0:10]
# Note that data and test8 are regularized and floated into an array of 8 columns:
# [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# The survival values are 0=die, 1=live, 2=don't know (for test8)
# Note that testdata is regularized and floated into an array of 7 columns
# [class, sex, age, sibsp, parch, fare, embarked]

totdata = vstack((data,test8)) # stacks data on top of test8 to create a (1309,8) array

# Let's build some powerful filtering algorithms.

# First, we might want to ask for only the data which has a certain value for a feature (like class for example)
def datafilter(value,index): return data[data[0::,index] == value]
#this outputs all of the data which has the value 'value' for index 'index'

# Now we implement this for any list of features...
# The function below supercedes datafilter, which does not need to be used, but is included above to
# make it easier to understand the logic of df
def df(features,dataset):
# features is a list of features we want to filter by. For ex. [[1,5],[2,6]]
# gives all data with sibsp = 1 and parch = 2.
    if size(dataset) == 0:
        return []
    datatemp=dataset #start with the regularized data, either data, test8, or testdata
    n = np.size(features,0) #number of features we are filtering by
    for x in xrange(n):
        feat = features[x] #pair [value,index] for each feature
        value = feat[0]
        index = feat[1]
        datatemp = datatemp[datatemp[0::,index] == value]
        if size(datatemp) == 0:
            return []
    return datatemp
# for data and test8 note indices [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# for testdata note indices [0=class, 1=sex, 2=age, 3=sibsp, 4=parch, 5=fare, 6=embarked]
# it is probably better to only use test8, and not testdata, so as to avoid confusion on the indices:
# using only data and test8 will have uniform index categories.

# dfrange will enable us to filter by age range or fare range, as well as ranges in the discrete features
def dfrange(fmin,fmax,index,dataset):
    if size(dataset) == 0:
        return []
    datatemp=dataset #start with the regularized data, either data, test8, or testdata
    truthtable = (datatemp[0::,index] <= fmax) & (datatemp[0::,index]>=fmin)
    datatemp = datatemp[truthtable]
    return datatemp

# Usage examples
#print dfrange(0,1,3,data) # 0 and 1 year olds
#print df([[1,2],[3,1]],dfrange(0,10,3,data)) #3rd class females with age<=10
#print df([[4,4],[1,2]],data) # female passengers with sibsp=4

#We'll also need to replace placeholder values for age = 1000
def convertages(dataset,ageindex): #ageindex=3 for test8 and data, and 2 fro testdata
    for row in dataset:
        if row[ageindex] == 1000:
            if row[ageindex+1]<=2: # replace placeholder age 1000 with age 30 if sibsp<=2
                row[ageindex] = 30
            else:
                row[ageindex] = 10 # replace placeholder age 1000 with age 10 if sibsp>=3
    return dataset

# Now let's actually convert the unknown ages to our best guesses.
# These derive from analysis below which were performed BEFORE doing this conversion
# (i.e. only on the data where the age is given). To repeat and confirm this analysis you can simply
# comment out the data conversion below

data = convertages(data,3)
test8 = convertages(test8,3)
testdata = convertages(testdata,2)
totdata = convertages(totdata,3)


def showstats(datasubset): # use this on any subset of data. don't use this on test data b/c we don't know sur values
    if size(datasubset) == 0:
        return "none"
    nsur = int(np.sum(datasubset[0::,0]))
    ntot = np.size(datasubset[0::,0])
    if ntot == 0:
        per=0
    else:
        per=round(float(nsur)/ntot,3)
    return [nsur, ntot, per] # return the number survived, the total in the datasubset, and the percent survived

# Now we can easily recreate the survival tables from predict.py, but more elegantly:
malestats = showstats(df([[0,2]],data))
femstats = showstats(df([[1,2]],data))

print "female and male stats"
print femstats
print malestats

sexclass=[]
for s in xrange(2):
    for c in xrange(1,4):
        sexclass.append(showstats(df([[s,2],[c,1]],data)))

sca = np.array(sexclass).reshape(2,3,3)
print "Sex-Class"
print sca

#sex-class-embarked
malesce=[]
femsce=[]
for c in xrange(1,4):
    for e in xrange(3):
        malesce.append(showstats(df([[0,2],[c,1],[e,7]],data)))
        femsce.append(showstats(df([[1,2],[c,1],[e,7]],data)))

msce = np.array(malesce).reshape(3,3,3)
fsce = np.array(femsce).reshape(3,3,3)

print "Male Class(1st block is 1st class) and City (rows 0-2 in each block)"
print msce
print "Female Class(1st block is 1st class) and City (rows 0-2 in each block)"
print fsce

print "Male sibsp"
for sibsp in xrange(10):
    print sibsp, showstats(df([[0,2],[sibsp,4]],data))

print "Female sibsp"
for sibsp in xrange(10):
    print sibsp, showstats(df([[1,2],[sibsp,4]],data))

print "Male parch"
for parch in xrange(10):
    print parch, showstats(df([[0,2],[parch,5]],data))

print "Female parch"
for parch in xrange(10):
    print parch, showstats(df([[1,2],[parch,5]],data))

for sib in xrange(3):
    for par in xrange(3):
        print sib,par,"male: %s" %showstats(df([[0,2],[sib,4],[par,5]],data)), \
        "female: %s" %showstats(df([[1,2],[sib,4],[par,5]],data))

print "1st class males sibsp"
for sibsp in xrange(10):
    print sibsp, showstats(df([[0,2],[1,1],[sibsp,4]],data))

print "1st class males parch"
for parch in xrange(10):
    print parch, showstats(df([[0,2],[1,1],[parch,5]],data))

#all young males that survive
#print df([[0,2],[1,0]],dfrange(0,10,3,data))

print "1st and 2nd class French males by sibsp"
for sibsp in xrange(2):
    print "age 0-80, sibsp = %i" %sibsp, showstats(df([[0,2],[1,7],[sibsp,4]],dfrange(0,80,3,dfrange(1,2,1,data))))
    print "age 0-19, sibsp = %i" %sibsp, showstats(df([[0,2],[1,7],[sibsp,4]],dfrange(0,19,3,dfrange(1,2,1,data))))
    print "age 20-80, sibsp = %i" %sibsp, showstats(df([[0,2],[1,7],[sibsp,4]],dfrange(20,80,3,dfrange(1,2,1,data))))

print showstats(df([[0,2],[1,7]],data))

print "3rd class young males by age bin"
for x in xrange(5):
    print showstats(df([[0,2],[3,1]],dfrange(2*x+0.01,2*(x+1),3,data)))

print "3rd class S females under age<=5"
print showstats(df([[1,2],[0,7]],dfrange(3,3,1,dfrange(0,5,3,data))))
print "3rd class S females under 18 with sibsp=0,1"
print df([[1,2],[0,7]],dfrange(0,1,4,dfrange(3,3,1,dfrange(0,18,3,data))))
print showstats(df([[1,2],[0,7]],dfrange(0,1,4,dfrange(3,3,1,dfrange(0,18,3,data)))))


print "young (age<=12) 3rd class males by sibsp"
print "sibsp=0,1:",showstats(df([[0,2],[3,1]],dfrange(0,1,4,dfrange(0,12,3,data))))
print "sibsp=2-8:",showstats(df([[0,2],[3,1]],dfrange(2,8,4,dfrange(0,12,3,data))))


print "3rd class girls then boys (age<=15) with many siblings (sibsp>=2)"
print showstats(df([[1,2]],dfrange(2,8,4,dfrange(3,3,1,dfrange(0,15,3,data)))))
print showstats(df([[0,2]],dfrange(2,8,4,dfrange(3,3,1,dfrange(0,15,3,data)))))

print "3rd class young girls from C or Q with many siblings"
print showstats(df([[1,2],[3,3]],dfrange(2,8,4,dfrange(1,2,7,dfrange(0,15,3,data)))))

print "3rd class S female young (age<=8) with 0 or 1 sibling"
print showstats(df([[1,2],[0,7]],dfrange(0,1,4,dfrange(3,3,1,dfrange(0,8,3,data)))))
print df([[1,2],[0,7]],dfrange(0,1,4,dfrange(3,3,1,dfrange(0,8,3,data))))
#print showstats(df([[1,2],[0,7]],dfrange(0,1,4,dfrange(3,3,1,dfrange(18,80,3,data)))))

# It looks like having lots of siblings is really bad for you. Conversely, having 0 or 1 seems to save
# otherwise damned souls F3Syoung and M3young. Let's modify out previous prediction F3SM12

# Let's collect some prediction functions, which operate on any data set that has rows of size 8
# Therefore use these with data and test8 only

# now let's import various prediction models from predictions.py
from predictions import *

Forest = RandomForestClassifier(n_estimators = 100)
#Create the random forest object which will include all the parameters for the fit
Forest = Forest.fit(data[0::,1::],data[0::,0])
#fit the training data to the training output and create the decision trees
data0 = scipy.delete(data,0,1) # define data0 to be the train data set with the given survival values deleted
rfctrain = Forest.predict(data0)
rfctest = Forest.predict(testdata)
#Take the same decision trees and run on the test data

print "Scores for 'predictions' back on train data for GM, F3SM12, newpred, and RFC"
print predicttrain(genderpred(data))
print predicttrain(f3sm12pred(data))
print predicttrain(newpred(data))
print predicttrain(rfctrain)


print "Comparing predictions"
comparepreds(newpred(test8),rfctest)
comparepreds(f3sm12pred(test8),rfctest)
comparepreds(newpred(test8),f3sm12pred(test8))


# No-age given analysis
print "The average age for passengers. " \
      "NOTE this should be run only when placeholder age for age not given is set to 1000"
print np.mean(dfrange(0,100,3,data)[0::,3]) #mean of age for age <=100
# However, we might do a bit better. Generally, sibsp>1 implies you are a child. Let's see this.
print "Average age by sibsp:"
for sp in xrange(6):
    print "sibsp = %i" %sp, np.mean(dfrange(sp,sp,4,dfrange(0,100,3,data))[0::,3])



