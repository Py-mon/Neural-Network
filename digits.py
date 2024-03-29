import tensorflow as tf
import keras.api._v2.keras as keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random

mnist = keras.datasets.mnist
(train, train_answers), (test, test_answers) = mnist.load_data()

# train = keras.utils.normalize(train, axis=1)
# test = keras.utils.normalize(test, axis=1)

# train = np.array(train).reshape(train.shape[0], -1)


# train = np.concatenate(train, axis=1)

# # mutate the image a bit
# for i in range(len(train)):
#     train[i] = imutils.rotate(train[i], random.randint(-10, 10))
# for i in range(len(test)):
#     test[i] = imutils.rotate(test[i], random.randint(-10, 10))


# train = keras.utils.normalize(train, axis=1)
# test = keras.utils.normalize(test, axis=1)

# mutate the image a bit


def shift_sideways(x, amt):
    new = np.zeros_like(x)
    if amt == 0:
        return x
    if amt > 0:
        new[:,amt:] = x[:,:-amt]
    else:
        amt = -amt
        new[:,:-amt] = x[:,amt:]
    return new

def shift_vertically(x, amt):
    new = np.zeros_like(x)
    if amt == 0:
        return x
    if amt > 0:
        new[amt:,:] = x[:-amt,:]
    else:
        amt = -amt
        new[:-amt,:] = x[amt:,:]
    return new

    
import numpy as np

def mutate(digit):
    digit = imutils.rotate(digit, random.randint(-5, 5))
    digit = shift_sideways(digit, random.randint(-1, 1))     
    digit = shift_vertically(digit, random.randint(-1, 1)) 
    return digit    
    
for i in range(len(train)):
    train[i] = mutate(train[i])   
        
for i in range(len(test)):
    test[i] = mutate(test[i])  
    

# img = train[1]
# img = np.array([img])

# fig, ax = plt.subplots()

# ax.imshow(img[0], cmap=plt.cm.binary)

# plt.show()

def train_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # sparse_categorical_crossentropy

    model.fit(train, train_answers, epochs=8,)

    loss, accuracy = model.evaluate(test, test_answers)

    print(loss, accuracy)

    model.save("model.model")
    
#train_model()

model: keras.models.Model = keras.models.load_model("model.model")
# age, bpm
# for i in range(10):
#     img = train[i]
#     print(train_answers[i])

#     fig, ax = plt.subplots()

#     ax.imshow(img, cmap=plt.cm.binary)

#     plt.show()


def read_grid(grid):
    img = np.abs([np.array(grid) - 255])
    pred = model.predict(img)

    return np.argmax(pred)


# image_number = 0

# while os.path.isfile(f"NeuralNetwork/Untitled{image_number}.png"):
#     img = cv2.imread(f"NeuralNetwork/Untitled{image_number}.png")[:, :, 0]
#     img = np.invert(np.array([img]))
#     print(img)
#     pred = model.predict(img)
#     print(np.argmax(pred))

#     fig, ax = plt.subplots()

#     ax.imshow(img[0], cmap=plt.cm.binary)

#     text = fig.text(0.80, 0.98, str(pred), horizontalalignment="center", wrap=True)

#     plt.show()
#     image_number += 1
