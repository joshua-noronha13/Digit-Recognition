import keras 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy 
import pandas as pd
import numpy as np

#Load files into dataframes
train_df = pd.read_csv("../train.csv",sep=",")
test_df = pd.read_csv("../test.csv",sep=",")

#Initializations
input_layer_size = 784
hidden_layer_size = 50
num_labels = 10
train_labels = train_df.values[:,0]
train_samples = train_df.values[:,1:]
test_samples = test_df.values


model = Sequential([
    Dense(784,input_shape=(784,),activation="relu"),
    Dense(50,activation="sigmoid"),
    Dense(10,activation="softmax")
])

model.summary()

model.compile(Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=["accuracy"])

model.fit(train_samples,train_labels,batch_size=10,epochs=2,shuffle=True,verbose=2)

y = model.predict(test_samples,batch_size=10,verbose=0)

y = np.argmax(y,axis=1)
print(y)
df = pd.DataFrame(y)
df.index = range(1,28001)
df.to_csv("output.csv")
