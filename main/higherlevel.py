
__author__ = 'michaelbinger'

# This will combine RFC methods with some powerful filtering methods

import csv as csv
import numpy as np
import scipy

# I created a new file PrepareTitanicData.py which goes through the details of preparing the data.
# It converts sex to 1,0 for F,M, and 0,1,2,3 for city embarked "S","C","Q",""
# Also for no-age we put in placeholder 1000
# for no fare we put placeholder 1000
# Cabin, Ticket #, and name are deleted

from PrepareTitanicData import titandata

data=titandata("train")
testdata=titandata("test")
# call function titandata which takes an argument string, which must be either "train", or "test"
#print data[0:10]
#print testdata[0:10]
# Note that data is regularized and floated into an array of 8 columns:
# [sur, class, sex, age, sibsp, parch, fare, embarked]
# Note that testdata is regularized and floated into an array of 7 columns:
# [class, sex, age, sibsp, parch, fare, embarked]


# Let's build some powerful filtering algorithms.

# First, we might want to ask for only the data which has a certain value for a feature (like class for example)
def datafilter(value,index): return data[data[0::,index] == value]
#this outputs all of the data which has the value 'value' for index 'index'

# Now we implement this for any list of features
def df(lst,dataset):
# lst is a list of features we want to filter by. For ex. [[1,5],[2,6]] gives all data with sibsp = 1 and parch =2
    datatemp=dataset #start with the regularized data, either data or testdata
    n = np.size(lst,0) # number of features we are filtering by
    for x in xrange(n):
        pair = lst[x] #pair=[value,index]
        value = pair[0]
        index = pair[1]
        datatemp = datatemp[datatemp[0::,index] == value]
    return datatemp
# for data note indices [0=sur, 1=class, 2=sex, 3=age, 4=sibsp, 5=parch, 6=fare, 7=embarked]

print datafilter(4,4) # passengers with sibsp=4
print df([[4,4]],data) # passengers with sibsp=4
print df([[1,4],[1,5]],data) # passengers with sibsp=1 and parch=1
print df([[1,4],[1,5],[1,2],[1,1]],data) #passengers with sibsp=1, parch=1, female, and 1st class

print "infants (age=0):", data[data[0::,3]==0]