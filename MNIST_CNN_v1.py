import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, models, utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 1 channel, and normalize
x_train = x_train.reshape(60000, 28, 28, 1)/255 
x_test = x_test.reshape(10000, 28, 28, 1)/255

# Show the sample image
# X = x_train[9487][:,:,0]
# plt.imshow(X, cmap='gray')
# plt.show()

"""
One-hot encoding
Why do we need one-hot encoding?
Because the output of the model is a probability distribution
The output is a vector of 10 elements.
The element with the highest probability is the predicted value.
"""
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = models.Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))

# Since the output is a probability distribution
# Softmax is used as the activation function of the output layer.
model.add(Dense(10))
model.add(Activation('softmax'))

# Categorical Crossentropy is used for multi-class classification
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=12)

score = model.evaluate(x_test, y_test)
print('Loss= ', score[0])
print('Accuracy= ', score[1])

# Save model
model_json = model.to_json()
open('handwriting_model_cnn.json', 'w').write(model_json)
model.save_weights('handwriting_weights_cnn.h5')

# Randomly select 5 samples for prediction
y_predict = np.argmax(model.predict(x_test), axis=-1)
pick = np.random.randint(1,9999, 5)

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[pick[i]].reshape(28,28), cmap='Greys')
    plt.title(y_predict[pick[i]])
    plt.axis("off")
plt.show()