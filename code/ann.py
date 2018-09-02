import pandas as pd
import numpy as np
import sigmoid as sig

#Load files into dataframes
train_df = pd.read_csv("../train.csv",sep=",")
test_df = pd.read_csv("../test.csv",sep=",")

#Initializations
input_layer_size = 784
hidden_layer_size = 50
num_labels = 10
train_labels = train_df.values[:,0]
m = np.size(train_labels)
delta1 = 0
delta2 = 0

#Randomly initialize weights of neural network
w1=np.random.randn(input_layer_size,hidden_layer_size)*0.01 
w2=np.random.randn(hidden_layer_size,num_labels)*0.01

a1 = train_df.values[:,1:]
a2 = sig.sigmoid(np.matmul(a1,w1))
a3 = sig.sigmoid(np.matmul(a2,w2))

print(np.shape(train_labels))
for i in range(m):
    #Forward Propogation  
    print(i)  
    y = np.zeros(num_labels)
    y[train_labels[i]] = 1
    print(y[train_labels[i]])
    del3 = a3[i] - y
    del2 = np.dot(np.dot(np.dot(w2,del3),a2),1-a2)
    delta2 = delta2 + np.dot(del3,a2)
    delta1 = delta1 + np.dot(del2,a1)