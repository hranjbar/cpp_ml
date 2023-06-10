import random
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_datasets as tfds


(_, ds_test), _ = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))



image = random.choice(ds_test)


plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()

# save image to jpg
im = Image.fromarray(image)
im.save("handWritten_digit.jpg")
