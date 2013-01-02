
__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

import csv as csv
import numpy as np
import scipy
from numpy import *

# I created a new file PrepareTitanicData.py which goes through the details of preparing the data.
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
# Note that testdata is regularized and floated into an array of 7 columns:
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
# gives all data with sibsp = 1 and parch = 2. Age and fare are entered [[min,max],index]
    datatemp=dataset #start with the regularized data, either data, test8, or testdata
    n = np.size(features,0) #number of features we are filtering by
    for x in xrange(n):
        feat = features[x] #pair [value,index] for each feature
        value = feat[0]
        index = feat[1]
        datatemp = datatemp[datatemp[0::,index] == value]
    return datatemp
# for data and test8 note indices [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]
# for testdata note indices [0=class, 1=sex, 2=age, 3=sibsp, 4=parch, 5=fare, 6=embarked]
# it is probably better to only use test8, and not testdata, so as to avoid confusion on the indices:
# using only data and test8 will have uniform index categories.

# dfrange will enable us to filter by age range or fare range
def dfrange(fmin,fmax,index,dataset):
    datatemp=dataset #start with the regularized data, either data, test8, or testdata
    truthtable = (datatemp[0::,index] <= fmax) & (datatemp[0::,index]>=fmin)
    datatemp = datatemp[truthtable]
    return datatemp

# Usage examples
#print dfrange(0,1,3,data) # 0 and 1 year olds

#print df([[1,2],[3,1]],dfrange(0,10,3,data)) #3rd class females with age<=10

#print df([[4,4],[1,2]],data) # female passengers with sibsp=4


def showstats(datasubset): # use this on any subset of data. don't use this on test data b/c we don't know sur values
    nsur = int(np.sum(datasubset[0::,0]))
    ntot = np.size(datasubset[0::,0])
    if ntot == 0:
        per=0
    else:
        per=round(float(nsur)/ntot,3)
    return [str(nsur), str(ntot), str(per)]

# Now we can easily recreate the survival tales from predict.py, but more elegantly:
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
malefam=[]
femfam=[]
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
print df([[0,2],[1,0]],dfrange(0,10,3,data))
