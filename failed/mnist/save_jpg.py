import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image

(_, _), (x_test, y_test) = mnist.load_data()
'''
x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), 
containing the test data. Pixel values range from 0 to 255.
y_test: uint8 NumPy array of digit labels (integers in range 0-9) 
with shape (10000,) for the test data.
'''
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

image = random.choice(x_test)
print(image.shape)

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()

# save image to jpg
im = Image.fromarray(image)
im.save("handWritten_digit.jpg")
