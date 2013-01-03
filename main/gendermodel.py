__author__ = 'michaelbinger'

import csv as csv
import numpy as np

#create the data array from the original file train.csv, as explained in kaggle tutorial
traincsv = csv.reader(open("../train.csv", 'rb'))
header = traincsv.next()
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

# Open the test data file to read in the test data, then write our first Gender Model (GM) predictions
testcsv = csv.reader(open("../test.csv",'rb'))
header = testcsv.next()
gmcsv = csv.writer(open('../gendermodelpy.csv','wb'))

for row in testcsv:
    if row[2] == 'female':
        row.insert(0,'1')
        gmcsv.writerow(row)
    else:
        row.insert(0,'0')
        gmcsv.writerow(row)


