#predict testing data using pre-trained model
import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

#get path
testDataPath = sys.argv[1]
outputPath = sys.argv[2]

#the features which are selected
#consider = [0,1,5,7,8,9,11,12,13]
consider = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

#read testing data 
test_x = []

file_test = open(testDataPath ,"r")
csvCursor = csv.reader(file_test)

rowNo = 0
for row in csvCursor:
    if rowNo%18 in consider:
        if rowNo %18 == 0:
            test_x.append([])
            for i in range(2,11):
                test_x[rowNo//18].append(float(row[i]) )
        else :
            for i in range(2,11):
                if row[i] !="NR":
                    test_x[rowNo//18].append(float(row[i]))
                else:
                    test_x[rowNo//18].append(0)
    rowNo += 1

file_test.close()
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

#load weight model
w = np.zeros(len(test_x[0]))
w = np.load('model.npy')


#get ans.csv with your model 
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

file_w = open(outputPath, "w+")
fw = csv.writer(file_w)
fw.writerow(["id","value"])

for i in range(len(ans)):
    fw.writerow(ans[i]) 
file_w.close()
