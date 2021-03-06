import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

#the features which are selected
#consider = [0,1,5,7,8,9,11,12,13]
consider = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

#decide whether to save and load parameter
is_loadParameter = 0  # 1: load / 0: don't load
is_saveParameter = 1  # 1: save / 0: don't save
#set learning rate and iteration
lr = 10
iteration = 888888
#set regularization parameter
lamda = 0

#create a data list to store the training data
data = list()
for i in range(18):
    data.append([])

#open the file of training data
trainingDataPath = 'train.csv'
file_train = open(trainingDataPath, 'r', encoding = 'big5') 
csvCursor = csv.reader(file_train)

#read training data  
rowNo = 0
for row in csvCursor:
    if rowNo!=0:
        for i in range(3, 27):
            if row[i] != "NR":
                data[(rowNo-1)%18].append(float(row[i]))
            else:
                data[(rowNo-1)%18].append(float(0)) 
    rowNo += 1
file_train.close()

#x:features, y:pm2.5 label 
x = []
y = []
for i in range(12): #12 month
    for j in range(471): #(20*24)-9 = 480-9 = 471 data in a month
        x.append([])
        for t in range(18):
            if t in consider:
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x, x**2), axis=1)

# add bias
bias = np.ones((x.shape[0], 1))
x = np.concatenate((bias, x), axis=1)

#weight
w = np.zeros(len(x[0]))
#load model
if is_loadParameter == 1:
    w = np.load('model.npy')

#close form solution
#w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

#start training
x_t = x.transpose()
w_grad = np.zeros(len(x[0]))
#load w_grad model
if is_loadParameter == 1:
    w_grad = np.load('w_grad.npy')

#gradient decent with adagrad
for i in range(iteration):
    h = np.dot(x, w)
    loss = h-y
    cost = math.sqrt(np.sum(loss**2) / len(x))
    grad = np.dot(x_t,loss) 
    w_grad += grad**2
    w = w - lr * grad/np.sqrt(w_grad)

    #print cost
    if i % 1000 ==0:
        print ("Round # ", i, " cost : ", cost)

# save model
if is_saveParameter == 1:
    np.save('model.npy',w) #save weight and bias
    np.save('w_grad', w_grad) #save w_grad