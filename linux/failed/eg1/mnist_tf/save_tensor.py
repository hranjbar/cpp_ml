import random
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_datasets as tfds


ds, info = tfds.load('mnist', split='train', with_info=True)


#ds = random.choice(ds_test)
#image, label = random.choice(ds_test)


#plt.imshow(image, cmap=plt.get_cmap('gray'))
#plt.show()

# save image to jpg
#im = Image.fromarray(image)
#im.save("handWritten_digit.jpg")
