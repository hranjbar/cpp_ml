import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# for TF to run on CPU
tf.config.set_visible_devices([], 'GPU')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_train = x_train.astype('float32') / 255.0

y_train = to_categorical(y_train)

model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
])

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
)

history = model.fit(x_train, y_train, epochs=5, batch_size=32)
# save model 
tf.saved_model.save(model, os.path.join(os.getcwd(), "model"))
image = random.choice(x_test)

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()

image = (image.reshape((1, 28, 28, 1))).astype('float32') / 255.0

digit = np.argmax(model.predict(image)[0], axis=-1)
print("Prediction:", digit)
