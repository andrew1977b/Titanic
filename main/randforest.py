__author__ = 'michaelbinger'


import csv as csv
import numpy as np

traincsv = csv.reader(open("../train.csv", 'rb'))

header = traincsv.next()

data=[]
for row in traincsv:
    data.append(row)
data = np.array(data) #create the data array from the original file train.csv, as explained in kaggle tutorial


#From sklearn.ensemble import RandomForestClassifier

Npass = np.size(data[0::,0].astype(np.float))  # number of passengers
Nsur = np.sum(data[0::,0].astype(np.float))  # number of survivers
persur = Nsur / Npass #percent surviving

print persur