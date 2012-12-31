__author__ = 'michaelbinger'


import csv as csv
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier


#create the data array from the original file train.csv, as explained in kaggle tutorial
traincsv = csv.reader(open("../train.csv", 'rb'))
header = traincsv.next()
data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)

#NOTE: data[j] for j=0..890 is of the form of 11 strings:
# ['survived?' 'class' 'name' 'sex' 'age' 'sibsp' 'parch' 'ticket #' 'fare' 'cabin' 'embarked']
# We will convert sex and embarked to integers, then delete fare(8), cabin(9), ticket#(7), and name(2).
# Then we will convert the array of strings of numbers in to an array of floats
# Finally we will feed in this array into the Random Forest function

# Before manipulating the actual data, let's development some facility with some toy examples

# This illustrates how to delete particular rows or columns of an array
testa = np.array([[1,2,3], [4,5,6], [7,8,9]])
print testa
print scipy.delete(testa,0,0) #deletes 1st (0) row (0)
print scipy.delete(testa,1,0) #deletes 2nd (1) row (0)
print scipy.delete(testa,2,0) #deletes 3rd (2) row (0)
print scipy.delete(testa,0,1) #deletes 1st (0) column (1)
print scipy.delete(testa,1,1) #deletes 2nd (1) column (1)
print scipy.delete(testa,2,1) #deletes 3rd (2) column (1)

# This illustrates how to convert an array of strings into an array of floats.
# I'm sure there is a more elegant method but this works. Paul?
testb = np.array([['1','2','3'],['4','5','6'],['7','8','9']])
testb = testb.ravel() #flattens array, bc I could only find a way to convert lists of strings to floats
print testb
testb = np.array([float(x) for x in testb]) # convert to floats
print testb
testb = testb.reshape(3,3) #reshape back into array
print testb
print type(testb) # confirms that we have a type numpy.ndarray again!


#convert sex F,M to 1,0
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
print data[1]
data = scipy.delete(data,9,1) # delete cabin
print data[1]
data = scipy.delete(data,8,1) # delete fare
print data[1]
data = scipy.delete(data,7,1) # delete ticket#
print data[1]
data = scipy.delete(data,2,1) # delete name
print data[1]
data = scipy.delete(data,5,1) # delete parch for now
print data[1]
data = scipy.delete(data,4,1) # delete sibsp for now

# convert the strings to floats
print data[0:10]
nrow = np.size(data[0::,0]) #number of rows and columns in data array
ncol = np.size(data[0,0::])
print nrow, ncol
data = data.ravel()
data = np.array([float(x) for x in data])
data = data.reshape(nrow,ncol)
print data[0:10]

Forest = RandomForestClassifier(n_estimators = 100)
#Create the random forest object whic will include all the parameters for the fit
Forest = Forest.fit(data[0::,1::],data[0::,0])
#fit the training data to the training output and create the decision trees

#create the TEST data array from the original file test.csv, as explained in kaggle tutorial
testcsv = csv.reader(open("../test.csv", 'rb'))
header = testcsv.next()
testdata=[]
for row in testcsv:
    testdata.append(row)
testdata = np.array(testdata)

#Perform all of the same conversions on the testdata as we did on data (train)
#NOTE the test data doesn't have a survival column, so indices will generally be one less than for data

#convert sex F,M to 1,0
testdata[testdata[0::,2] == "female", 2] = 1
testdata[testdata[0::,2] == "male", 2] = 0

#convert embarked strings into numbers for easier coding
testdata[testdata[0::,9] == "S", 9] = 0
testdata[testdata[0::,9] == "C", 9] = 1
testdata[testdata[0::,9] == "Q", 9] = 2
testdata[testdata[0::,9] == "", 9] = 3

# for passengers with no age, put age 30
testdata[testdata[0::,3] == "", 3] = 30

testdata = scipy.delete(testdata,8,1)
testdata = scipy.delete(testdata,7,1)
testdata = scipy.delete(testdata,6,1)
testdata = scipy.delete(testdata,1,1)
testdata = scipy.delete(testdata,4,1) # delete parch for now
testdata = scipy.delete(testdata,3,1) # delete sibsp for now

nrow = np.size(testdata[0::,0]) #number of rows and columns in data array
ncol = np.size(testdata[0,0::])
print nrow, ncol
testdata = testdata.ravel()
testdata = np.array([float(x) for x in testdata])
testdata = testdata.reshape(nrow,ncol)
print testdata[0:10]

RFCpred = Forest.predict(testdata) #Take the same decision trees and run on the test data

# IMPORTANT NOTE: Because the RFC is random, each time you run this code will result in different predictions!
print "This is our first RFC Prediction on the Test data!!!"
print RFCpred

numpred = np.sum(RFCpred)
totnum = np.size(RFCpred)
print "Summary of RFC predictions"
print numpred, totnum, numpred/totnum


# Let's now compare the Random Forest Classifier (RFC)
# to the gender model(GM) predictions based on sex only
sextest = testdata[0::,1]
print "Predictions of Gender Model on Test data"
print sextest
print "Comparing RFC with Gender Model Predictions for Test Data"
comparetoGM = RFCpred - sextest
print comparetoGM
# 0 means they agree.
# 1 means the RFC predicted survival whereas GM did not
# -1 means GM predicted predicted survival whereas RFC did not
print "The number of -1's and then 1's"
print sum(comparetoGM[0::]==-1) # the number of -1's
print sum(comparetoGM[0::]==1) # number of 1's
print "Total Number of disagreements between RFC and GM"
print sum(abs(comparetoGM[0::])==1)

# Now let's compare the RFC to the F3SM12 model from predict.py, where all women,
# except 3rd class Southampton, and young (<=10) 1st and 2nd class males survive.
f3sm12pred=list([])
for row in testdata: #re-create the F3SM12 predictions on the testdata, but in a nice python list object
    if (row[1] == 1) and not( (row[0] == 3) & (row[3] == 0) ):
        f3sm12pred.append(1)
    elif (row[1] == 0) and (row[2] <= 10) and not(row[0] == 3):
        f3sm12pred.append(1)
    else:
        f3sm12pred.append(0)

f3sm12pred = np.array(f3sm12pred) # convert to array
#print f3sm12pred
comparetoF3SM12 = RFCpred - f3sm12pred
print "Comparing RFC with F3SM12 Model Predictions for Test Data"
print comparetoF3SM12
# 0 means they agree.
# 1 means the RFC predicted survival whereas GM did not
# -1 means GM predicted predicted survival whereas RFC did not
print "The number of -1's and then 1's"
print sum(comparetoF3SM12[0::]==-1)
print sum(comparetoF3SM12[0::]==1)
print "Total Number of disagreements between RFC and F3SM12"
print sum(abs(comparetoF3SM12[0::])==1)

# To understand things further, let's train our forest as usual on the training data,
# but then apply the derived tree back onto the same training data!!

Forest = Forest.fit(data[0::,1::],data[0::,0]) #fit the training data to the training output and create the decision trees
data0 = scipy.delete(data,0,1) # define data0 to be the train data set with the given survival values deleted
RFCpred = Forest.predict(data0) #Take the same decision trees and run back on the train data
comptrain = RFCpred-data[0::,0] #compare RFC predictions with reality
#print comptrain
print np.nonzero(comptrain) #This shows which elements were predicted wrong
numwrong = sum(abs(comptrain[0::])==1)
scoretrain = 1 - float(numwrong) / float(np.size(comptrain))
print "The number wrong and 'score' when applying the learned decision tree back onto the train data"
print numwrong, np.size(comptrain)
print scoretrain
# We consistently score around 85% on the train data set!! That bodes well!
# NOTE this assumes we are not using sibsp and parch data... i.e. using a "4 feature" training set:
# only sex, class, age, and embarking city go into the predictive tree


# Now let's try to optimize the n_estimators variable in the RFC
# We'll do this by running the RFC multiple times for each
# of several values of n_estimators, and see what we find out!

Forest = RandomForestClassifier(n_estimators = 100)
#Create the random forest object which will include all the parameters for the fit

numforests = 3 #number of times we perform the RFC on the training data
# NOTE: It takes about 40 sec to run this code on my 4 yr old macbook pro for numforests=100, so don't make it too big

ndgm = [] # This empty list will be populated by the total number of
# disagreements between RFC and GM, for each iteration of RFC
nd2 = [] # Ditto for F3SM12
for x in xrange(numforests):
    Forest = Forest.fit(data[0::,1::],data[0::,0]) #fit the training data to the training output and create the decision trees
    RFCpred = Forest.predict(testdata) #Take the same decision trees and run on the test data
    comparetoGM = RFCpred-sextest
    comparetoF3SM12 = RFCpred-f3sm12pred
    nd = sum(abs(comparetoGM[0::])==1)
    ndgm.append(nd)
    nd = sum(abs(comparetoF3SM12[0::])==1)
    nd2.append(nd)

ndgm = [float(x) for x in ndgm]
ndgm = np.array(ndgm)
nd2 = [float(x) for x in nd2]
nd2 = np.array(nd2)

print "Total Number of disagreements between RFC and GM"
print ndgm
print np.sum(ndgm)/np.size(ndgm)
print "Total Number of disagreements between RFC and F3SM12"
print nd2
print np.sum(nd2)/np.size(nd2)

# Running this with n_estimators=100 and numforests=100 I find ndgm = 48 and nd2 = 14 on average.
# Running this with n_estimators=20 and numforests=100 I find ndgm = 49 and nd2 = 16 on average.
# Running this with n_estimators=5 and numforests=100 I find ndgm = 52 and nd2 = 25 on average.
# Running this with n_estimators=3 and numforests=100 I find ndgm = 52 and nd2 = 28 on average.
# Running this with n_estimators=40 and numforests=100 I find ndgm = 49 and nd2 = 15 on average.
# Running this with n_estimators=70 and numforests=100 I find ndgm = 48 and nd2 = 14 on average.
# Running this with n_estimators=250 and numforests=100 I find ndgm = 48 and nd2 = 13 on average.

# This all tells us that the predictions don't seem to depend too much on the
# n_estimators variable, so long as it is not too small.


# Now let's see which passengers my F3SM12 model and the RFC disagree on...
# maybe we can learn something!

Forest = RandomForestClassifier(n_estimators = 100)
#Create the random forest object which will include all the parameters for the fit

numforests = 10 #number of times we perform the RFC on the training data
# NOTE: It takes about 40 sec to run this code on my 4 yr old macbook pro for numforests=100, so don't make it too big

nd2 = []
passdisagree = []
pd = []
disagreecount = np.zeros(np.size(f3sm12pred))
for x in xrange(numforests):
    Forest = Forest.fit(data[0::,1::],data[0::,0]) #fit the training data to the training output and create decision trees
    RFCpred = Forest.predict(testdata) #Take the decision trees and run on the test data
    comparetoF3SM12 = RFCpred-f3sm12pred
    disagreecount = disagreecount + abs(comparetoF3SM12)
    pd = np.nonzero(comparetoF3SM12) #tells us which elements of the array are nonzero, i.e. for which passengers there is disagreement
    passdisagree.append(pd) #create a list of lists of disagreements btw RFC and F3SM12, appended each iteration
    nd = sum(abs(comparetoF3SM12[0::])==1) #the number of disagreement for each iteration
    nd2.append(nd) # a list containing number of disagreements for every iteration

nd2 = [float(x) for x in nd2]
nd2 = np.array(nd2)
passdisagree = np.array(passdisagree)
passdis = passdisagree[0::,0] # removes extra brackets that somehow get in there and screw things up if you're not careful

print "Here's the number of disagreements between RFC and F3SM12 for each forest run (numforests)"
print nd2
print "Here's a list of passengers (by index in array testdata) that we disagree on"
print passdis
print "Here's the first iteration"
print passdis[0]
print "with my (F3SM12) predictions"
print "NOTE: 1/0 below means that RFC thinks that passenger will die/live (since RFC disagrees with me)"
print f3sm12pred[passdis[0]]
print "And now the actual passenger data (class, sex, age, city) for those disagreements"
print testdata[passdis[0]]

# Since RFC predictions vary probabilistically, let's see which passengers
# RFC disagrees with F3SM12 more than 50% of the time.

print "Here is the probability that the above RFC disagrees with F3SM12 for each passenger"
#print disagreecount
disagprob = disagreecount/numforests
print disagprob

print "Here are the passenger indices for which this prob is >= 0.5 (after 100 iterations)"
finaldisagreeRFC = np.nonzero(disagprob >= 0.5)
# Note there is a lot going on in the above line. The argument disagprob>=0.5 creates a truth table
# with False if the average disagreement probability is below 0.5, and True if it is greater
# Since python interprets True as 1 and False as 0, then when the function np.nonzero is applied,
# it returns only those elements which are True, i.e. for which there is greater than 50% disagreement between
# RFC and F3SM12, averaged over all forest iterations.
print finaldisagreeRFC


# I ran the above code, with the 4-feature RFC (class,sex,age,city) 100 times (numforests=100)
# and found the RFC more often than not disagreed with F3SM12 for these 15 passengers:
# (array([  8,  19,  32,  49,  64,  80,  94, 117, 153, 201, 206, 263, 313, 347, 354]),)

#write these out as an array (disagreement passengers==>dapass)
dapass = np.array([8,  19,  32,  49,  64,  80,  94, 117, 153, 201, 206, 263, 313, 347, 354])

#create an array with these averaged RFC predictions.
ntest = np.size(f3sm12pred) #size of the test data set = 418.
rfcfinalpred = np.zeros(ntest) #initialize an array of ntest=418 zeros
for x in xrange(ntest):
    rfcfinalpred[x] = f3sm12pred[x] # each RFC prediction is the same as F3SM12, except...
    if x in dapass: # if x is in dapass, then...
        rfcfinalpred[x] = not(rfcfinalpred[x]) # flip the prediction. not(0)=1, not(1)=0.

#print rfcfinalpred
#print rfcfinalpred-f3sm12pred

# Let's write these averaged RFC predictions to a file

newcsv = csv.writer(open('../RFC1stpredictionpy.csv','wb'))

for x in xrange(ntest):
    if rfcfinalpred[x]==0:
        newcsv.writerow(["0"]) # writerow takes a list and writes it to a row.
    if rfcfinalpred[x]==1:
        newcsv.writerow(["1"]) # We only need the predictions, not the other passenger data.


