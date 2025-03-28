import numpy as np 
import pandas as pd 
import tensorflow as tf
from keras import datasets, layers, models, utils, optimizers
import matplotlib.pyplot as plt 
np.random.seed(10)
###############################################
############ Code From yenlung ################
###############################################

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print('train data= ',len(x_train))
print('test data=', len(x_test))

x = x_train[87]
# print(x.shape)
# plt.imshow(x, cmap='gray')
# plt.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10) 

print(y_train[87])

# Create a nerual network model
model = models.Sequential()

model.add(layers.Dense(20, input_dim=784))
model.add(layers.Activation('relu'))

model.add(layers.Dense(10))
model.add(layers.Activation('relu'))

model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

model.compile(loss='mse', optimizer=optimizers.Adam(), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=20)

y_predict = np.argmax(model.predict(x_test), axis=-1)
print(y_predict)

def test(test_no):
    plt.imshow(x_test[test_no].reshape(28, 28), cmap='gray')
    print('predict= ', y_predict[test_no])
    plt.show()
test(87)

score = model.evaluate(x_test, y_test)
print('Loss= ', score[0])
print('Accuracy= ', score[1])

