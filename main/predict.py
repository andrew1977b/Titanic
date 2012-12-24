__author__ = 'michaelbinger'

import csv as csv
import numpy as np

traincsv = csv.reader(open("../train.csv", 'rb'))

header = traincsv.next()

data=[]
for row in traincsv:
    data.append(row)
data = np.array(data)

