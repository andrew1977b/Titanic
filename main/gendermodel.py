__author__ = 'michaelbinger'

import csv as csv
import numpy as np

traincsv = csv.reader(open("../train.csv", 'rb'))

header = traincsv.next()

data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)

Npass = np.size(data[0::,0].astype(np.float))
Nsur = np.sum(data[0::,0].astype(np.float))
persur = Nsur / Npass

Wstats = data[0::,3] == "female"
Mstats = data[0::,3] != "female"

Won = data[Wstats,0].astype(np.float)
Mon = data[Mstats,0].astype(np.float)

perWsur = np.sum(Won) / np.size(Won)
perMsur = np.sum(Mon) / np.size(Mon)

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

surtab = np.zeros((2,3,4))

print surtab
print persur
print perWsur
print perMsur
