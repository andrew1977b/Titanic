__author__ = 'michaelbinger'

# The purpose of this file is to collect various hints, help, shortcuts to things relevant to our projects.
# Everyone please add to this!!!

import csv as csv
import numpy as np
import scipy

print "READING IN THE DATA FROM A FILE"
#create the data array from the original file train.csv, as explained in kaggle tutorial
traincsv = csv.reader(open("../train.csv", 'rb')) # go back one directory, find file train.csv, and open it
print type(traincsv) #gives type = _csv.reader. We need to convert this to type list and then array!

traincsv.next() # remove first row of header info. Note in the tutorial they suggest the line
# header = traincsv.next()
# This is completely unnecessary and confusing. They are just creating a new object called header for no reason.
# Try commenting out the above line and you'll see that the following is now the first line of data:
#['survived' 'pclass' 'name' 'sex' 'age' 'sibsp' 'parch' 'ticket' 'fare' 'cabin' 'embarked']
data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)
print data[0]
#NOTE: data[j] for j=0..890 is of the form of 11 strings:
# ['survived?' 'class' 'name' 'sex' 'age' 'sibsp' 'parch' 'ticket #' 'fare' 'cabin' 'embarked']

print "BE CAREFUL WITH DATA TYPES!!!"
#NOTE: be careful with data types... the ages are strings (of numbers)
print data[0,4], type(data[0,4])
print data[0,4] > 60.0 #True for unknown reasons. Note we are comparing the string "22" with float 60.0
print data[0,4] == 22 #False because its a string on the lhs
print data[0,4].astype(np.float) == 22 #True

print "DELETING ROWS AND COLUMNS FROM ARRAYS"
# This illustrates how to delete particular rows or columns of an array
testa = np.array([[1,2,3], [4,5,6], [7,8,9]])
print testa
print scipy.delete(testa,0,0) #deletes 1st (0) row (0)
print scipy.delete(testa,1,0) #deletes 2nd (1) row (0)
print scipy.delete(testa,2,0) #deletes 3rd (2) row (0)
print scipy.delete(testa,0,1) #deletes 1st (0) column (1)
print scipy.delete(testa,1,1) #deletes 2nd (1) column (1)
print scipy.delete(testa,2,1) #deletes 3rd (2) column (1)

print "CONVERTING ARRAYS OF STRINGS INTO FLOATS"
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

print "JUST 'FOR' FUN!!"
print "xrange(10)"
for x in xrange(10):
    print x
print "xrange(3,9)"
for x in xrange(3,9):
    print x
print "xrange(1,13,2)"
for x in xrange(1,13,2):
    print x
print "xrange(1,13,1)"
for x in xrange(1,13,1):
    print x

print "INTEGRATION!!"
from scipy import integrate
print integrate.quad(lambda x: x**3 ,0, 2)
# As far as I can tell, lambda defines a function "on the fly" for use in the integral

def f(x): return x**3
print f(3)
print integrate.quad(lambda x: f(x) ,0, 2)

#Special functions (here a Bessel function)
from scipy.special import jv
print integrate.quad(lambda x: jv(2.5,x), 0, 4.5)

#Zeros of the Bessel function
from scipy.special import jn_zeros
print jn_zeros(1,2)

print "MORE LAMBDA STUFF"
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print filter(lambda x: x % 3 == 0, foo)
#[18, 9, 24, 12, 27]
print map(lambda x: x * 2 + 10, foo)
#[14, 46, 28, 54, 44, 58, 26, 34, 64]
print reduce(lambda x, y: x + y, foo)
#139
foo = [1,-2,4,-7]
print reduce(lambda x, y: x*y, foo)
#56
print "WORD PLAY"
sentence = 'It is raining cats and dogs'
words = sentence.split()
print words
# ['It', 'is', 'raining', 'cats', 'and', 'dogs']
lengths = map(lambda word: len(word), words)
print lengths
# [2, 2, 7, 4, 3, 4]