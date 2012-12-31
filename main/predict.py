__author__ = 'michaelbinger'

# The structure of this file is to first do the basic data import,
# then create various survival probabilities for categories of increasing
# complexity. Sex only, then sex/class, then sex/class/embarked, then account for age
# Along the way certain changes to the format of 'data' are made
#

import csv as csv
import numpy as np

#create the data array from the original file train.csv, as explained in kaggle tutorial
traincsv = csv.reader(open("../train.csv", 'rb'))
traincsv.next()
data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)
#NOTE: data[j] for j=0..890 is of the form of 11 strings:
# ['survived?' 'class' 'name' 'sex' 'age' 'sibsp' 'parch' 'ticket #' 'fare' 'cabin' 'embarked']

Npass = np.size(data[0::,0].astype(np.float))  # number of passengers
Nsur = np.sum(data[0::,0].astype(np.float))  # number of survivers
persur = Nsur / Npass #percent surviving

Wstats = data[0::,3] == "female" # creates a table with value True if female and False if not
Mstats = data[0::,3] != "female" # similar to above but opposite truth values

Won = data[Wstats,0].astype(np.float) # women only data, in the form of '0' or '1' for survival values
Mon = data[Mstats,0].astype(np.float) # men only data

perWsur = np.sum(Won) / np.size(Won) #percent of women surviving
perMsur = np.sum(Mon) / np.size(Mon) # percent men

print "Total precent survived"
print persur
print "Percent female survived"
print perWsur
print "Percent male survived"
print perMsur

#Results
#0.383838383838
#0.742038216561
#0.188908145581

# Now we are going to create a survival table taking class in to account
nclass = 3
surtab = np.zeros((2,3)) #define a survival table full of zeros for now

# for i = 0,1,2 we'll go through and write to surtab the survival probabilities for each of the 6 sex/class combos
for i in xrange(3):
    wonstats = data[(data[0::,3] == "female")&(data[0::,1].astype(np.float) == i+1),0]
    monstats = data[(data[0::,3] != "female")&(data[0::,1].astype(np.float) == i+1),0]
    surtab[0,i] = np.mean(wonstats.astype(np.float))
    surtab[1,i] = np.mean(monstats.astype(np.float))
print "Sex-Class Survival Table"
print surtab

# result is
# [[ 0.96808511  0.92105263  0.5       ]
# [ 0.36885246  0.15740741  0.13544669]]

# Now account for sex, class, and embarked location (S,C,or Q)
surtab = np.zeros((2,3,3))
numsurtab = np.zeros((2,3,3)) #will be number survived for each category
totsurtab = np.zeros((2,3,3)) #will be total number in each category

#convert embarked strings into numbers for easier coding
data[data[0::,10] == "S", 10] = 0
data[data[0::,10] == "C", 10] = 1
data[data[0::,10] == "Q", 10] = 2
data[data[0::,10] == "", 10] = 3

for i in xrange(3):
    for j in xrange(3):
        wonstats = data[ (data[0::,3] == "female")
                         &(data[0::,1].astype(np.float) == i+1)
                         &(data[0::,10].astype(np.float) == j) ,0]
        surtab[0,i,j] = np.mean(wonstats.astype(np.float)) #percent survived for each sex/class/embark combo
        numsurtab[0,i,j] = np.sum(wonstats.astype(np.float)) #total number survived for each
        totsurtab[0,i,j] = np.size(wonstats.astype(np.float)) #total number in each combo
        monstats = data[ (data[0::,3] != "female")
                         &(data[0::,1].astype(np.float) == i+1)
                         &(data[0::,10].astype(np.float) == j) ,0]
        surtab[1,i,j] = np.mean(monstats.astype(np.float)) #percent survived for each sex/class/embark combo
        numsurtab[1,i,j] = np.sum(monstats.astype(np.float)) #total number survived for each
        totsurtab[1,i,j] = np.size(monstats.astype(np.float)) #total number in each combo
        surtab[surtab != surtab] = 0

wnocity = data[(data[0::,3] == "female")
               &(data[0::,10].astype(np.float) == 3) ,0]
# there are only two passengers, both female, with no embarked city listed. they both survived
print "Sex-Class-Embarked Survival Table"
print surtab
print "Total survived by sex-class-embarked"
print numsurtab
print "Total number of passengers by sex-class-embarked"
print totsurtab

# Results: note S,C,Q are columns 1,2,3 respectively while class = row number.
# female matrix is first, then male
#Sex-Class-Embarked Survival Table
#[[[ 0.95833333  0.97674419  1.        ]
#  [ 0.91044776  1.          1.        ]
# [ 0.375       0.65217391  0.72727273]]
#
#[[ 0.35443038  0.4047619   0.        ]
# [ 0.15463918  0.2         0.        ]
#[ 0.12830189  0.23255814  0.07692308]]]
#Total survived by sex-class-embarked
#[[[ 46.  42.   1.]
#  [ 61.   7.   2.]
# [ 33.  15.  24.]]
#
#[[ 28.  17.   0.]
# [ 15.   2.   0.]
#[ 34.  10.   3.]]]
#Total number of passengers by sex-class-embarked
#[[[  48.   43.    1.]
#  [  67.    7.    2.]
# [  88.   23.   33.]]
#
#[[  79.   42.    1.]
# [  97.   10.    1.]
#[ 265.   43.   39.]]]
# NOTE : 3rd class females from Southampton (S) don't make it, surviving only 37.5%

# Now look at age
agemax = 60. # anyone older than this will have there age reset to 60.
data[data[0::,4] == "",4] = 0 # for any passenger whose age is not given, set value to 0

print "A cautionary note about data types"
#NOTE: be careful with data types... the ages are strings (of numbers)
print data[0,4], type(data[0,4])
print data[0,4] > agemax #True for unknown reasons. Note we are comparing the string "22" with float 60.0
print data[0,4] == 22 #False because its a string on the lhs
print data[0,4].astype(np.float) == 22 #True

data[ data[0::,4].astype(np.float) > agemax, 4] = agemax
# resetting old folks age to 60

agebin = 10.
nagebins = agemax / agebin #number of age bins.
nagebins = np.int(nagebins) # convert this to integer

surtab = np.zeros((2,nagebins))


for i in xrange(nagebins):
    wonstats = data[ (data[0::,3] == "female")
                     &(data[0::,4].astype(np.float) > i*agebin+0.01) #the +0.01 weeds out 0's (unknowns)
                     &(data[0::,4].astype(np.float) <= (i+1)*agebin),0]
    surtab[0,i] = np.mean(wonstats.astype(np.float))
    monstats = data[ (data[0::,3] != "female")
                     &(data[0::,4].astype(np.float) > i*agebin+0.01)
                     &(data[0::,4].astype(np.float) <= (i+1)*agebin),0]
    surtab[1,i] = np.mean(monstats.astype(np.float))


noagef = data[(data[0::,3] == "female")&(data[0::,4].astype(np.float) == 0), 0]
noagem = data[(data[0::,3] != "female")&(data[0::,4].astype(np.float) == 0), 0]

print "Female no-age-given survival figures"
print np.mean(noagef.astype(np.float)),\
np.sum(noagef.astype(np.float)),\
np.size(noagef.astype(np.float))
print "Male no-age-given survival figures"
print np.mean(noagem.astype(np.float)),\
np.sum(noagem.astype(np.float)),\
np.size(noagem.astype(np.float))
# Results
# 0.679245283019 36.0 53
# 0.129032258065 16.0 124
# No age given survival rates are 68% and 13%, resp, for F and M
# This is a little less than the 74% and 19% baseline for all passengers. Makes sense.

print "Sex-Age Survival Table"
print surtab
# Results: These are age bins of 0.01-10, 10.01-20, ... 50.01-60 for female then male
# Note that those few passengers over 60 are counted as being 60
#[[  0.61290323   0.73913043   0.75308642   0.83636364   0.67741935  0.92857143]
# [  0.57575758   0.14492754   0.15436242   0.23         0.21818182  0.12765957]]

# NOTE: Since females almost always survive regardless, let's look more closely at young males.

agemax = 10.
agebin = 2.
# We now have 5 bins each of 2 years, up to age 10
nagebins = agemax / agebin #number of age bins.
nagebins = np.int(nagebins) # convert this to integer

surtab = np.zeros((1,3, nagebins))
numsurtab = np.zeros((1,3, nagebins))
totsurtab = np.zeros((1,3, nagebins))

for i in xrange(3):
    for j in xrange(nagebins):
        monstats = data[ (data[0::,3] != "female")
                     &(data[0::,1].astype(np.float) == i+1)
                     &(data[0::,4].astype(np.float) > j*agebin+0.01)
                     &(data[0::,4].astype(np.float) <= (j+1)*agebin), 0]
        surtab[0,i,j] = np.mean(monstats.astype(np.float))
        surtab[surtab != surtab] = 0
        numsurtab[0,i,j] = np.sum(monstats.astype(np.float)) #total number survived for each
        totsurtab[0,i,j] = np.size(monstats.astype(np.float)) #total number in each combo

print "Young Male (<=10) Survival, by class (row) and age bins (column) of 2 years"
print surtab
print "Young males that survived"
print numsurtab
print "Total young males"
print totsurtab
print "Total young male survival figures"
print np.sum(numsurtab),np.sum(totsurtab),np.sum(numsurtab)/np.sum(totsurtab)

#Young Male (<=10) Survival, by class (row) and age bins (column) of 2 years
#[[[ 1.          1.          0.          0.          0.        ]
#  [ 1.          1.          0.          1.          0.        ]
# [ 0.28571429  0.5         1.          0.          0.4       ]]]
#Young males that survived
#[[[ 1.  1.  0.  0.  0.]
#  [ 6.  2.  0.  1.  0.]
# [ 2.  3.  1.  0.  2.]]]
#Total young males
#[[[ 1.  1.  0.  0.  0.]
#  [ 6.  2.  0.  1.  0.]
# [ 7.  6.  1.  3.  5.]]]
#Total young male survival figures
#19.0 33.0 0.575757575758

#NOTE: it looks like 1st and 2nd class young boys (<=10) survive, whereas 3rd class boys are screwed
# Let's stop here and write our predictions to a file.
# To summarize, all females except 3rd class Southampton survive, as well as 1st and
# 2nd class young boys
# Let's call this the F3SM12 model.


testcsv = csv.reader(open("../test.csv",'rb'))
header = testcsv.next()
newcsv = csv.writer(open('../F3SM12predictionpy.csv','wb'))

for row in testcsv:
    # We need to first make sure all the data is complete.
    # For passengers without age, input 1000. For those without embarking city, put "X"
    if row[3] == "":
       row[3] = "1000"
    if row[9] == "":
       row[9] = "X"
    if (row[2] == 'female') and not((row[0] == "3")&(row[9] == "S")):
        row.insert(0,'1')
        newcsv.writerow(row)
    elif (row[2] != 'female') and (float(row[3]) <= 10) and not(row[0] == "3"):
        row.insert(0,'1')
        newcsv.writerow(row)
    else:
        row.insert(0,'0')
        newcsv.writerow(row)
