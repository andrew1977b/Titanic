__author__ = 'Andrew'

import csv as csv
import numpy as np

#read in an comma-separated-value (CSV) file.  The function uses the name of excel file as the main argument, reads it in, and returns an array
#all this really does right now is encapsulate Michael's code into a function.

def convert_csv_file_to_array (csv_file):
    traincsv = csv.reader(open("../train.csv", 'rb'))
    traincsv.next()
    data = []
    fulldata=[]
    for row in traincsv:
        fulldata.append(row)
    return data = np.array(fulldata)

#take a numpy array and write it to a csv file
#def write_array_to_csv_file (array, write_file_name):
    #newcsv = csv.writer(open('../'+write_file_name'+.csv','wb'))

#data = convert_csv_file_to_array("train")