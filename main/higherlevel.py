
__author__ = 'michaelbinger'

# The structure of this file is to first do the basic data import,
# then create various survival probabilities for categories of increasing
# complexity. Sex only, then sex/class, then sex/class/embarked, then account for age
# Along the way certain changes to the format of 'data' are made
#

import csv as csv
import numpy as np
import scipy

#create the data array from the original file train.csv, as explained in kaggle tutorial
traincsv = csv.reader(open("../train.csv", 'rb'))
traincsv.next()
data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)
#NOTE: data[j] for j=0..890 is of the form of 11 strings:
# ['survived?' 'class' 'name' 'sex' 'age' 'sibsp' 'parch' 'ticket #' 'fare' 'cabin' 'embarked']

#convert sex F,M to 0,1
data[data[0::,3] == "female", 3] = 1
data[data[0::,3] == "male", 3] = 0

#convert embarked strings into numbers for easier coding
data[data[0::,10] == "S", 10] = 0
data[data[0::,10] == "C", 10] = 1
data[data[0::,10] == "Q", 10] = 2
data[data[0::,10] == "", 10] = 3

# for passengers with no age, put age 30
data[data[0::,4] == "", 4] = 30

# Now we will delete the unwanted columns of data. I print it out
# for a passenger so you can see it's doing the right thing
data = scipy.delete(data,9,1) # delete cabin
data = scipy.delete(data,8,1) # delete fare
data = scipy.delete(data,7,1) # delete ticket#
data = scipy.delete(data,2,1) # delete name

# convert the strings to floats
nrow = np.size(data[0::,0]) #number of rows and columns in data array
ncol = np.size(data[0,0::])
#print nrow, ncol
data = data.ravel()
data = np.array([float(x) for x in data])
data = data.reshape(nrow,ncol)
#print data[0:10]

# Now that the data is regularized into an array of 7 columns, let's build some powerful filtering algorithms.
# NOTE: data is all floated and of form [sur, class, sex, age, embark, sibsp, parch]

# First, we might want to ask for only the data which has a certain value for a feature (like class for example)
def datafilter(value,index): return data[data[0::,index] == value]
#this outputs all of the data which has the value 'value' for index 'index'

# Now we implement this for any list of features
def df(lst): # lst is a list of features we want to filter by. For ex. [[1,5],[2,6]] gives all data with sibsp = 1 and parch =2
    datatemp=data #start with the regularized trainging data
    n = np.size(lst,0) # number of features we are filtering by
    for x in xrange(n):
        pair = lst[x] #pair=[value,index]
        value = pair[0]
        index = pair[1]
        datatemp = datatemp[datatemp[0::,index] == value]
    return datatemp

print datafilter(5,5)
print df([[5,5]])
print df([[1,5],[1,6]])