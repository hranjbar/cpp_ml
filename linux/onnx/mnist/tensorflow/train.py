import tensorflow as tf
from tensorflow import keras
import os

# for TF to run on CPU
tf.config.set_visible_devices([], 'GPU')

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

train_images = train_images.reshape(-1, 28, 28, 1) / 255.0 # TARGET_CHANNELS_LAST = "nhwc"
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

in_shape = train_images.shape
in_spec = tf.TensorSpec(in_shape[1:], tf.float64)
print(f"input shape: {in_shape}, input data type: {train_images.dtype},\n\t input spec: {in_spec}")
#o_shape = train_labels.shape
#o_spec = tf.TensorSpec(o_shape[1:], tf.float32)
#print(f"output shape: {o_shape}, output data type: {train_labels.dtype},\n\t output spec: {o_spec}")

# keras.layers.InputLayer(input_shape=(28, 28, 1), dtype=tf.float64, name="image"),
def create_model():
	model = keras.models.Sequential([
		keras.layers.InputLayer(input_shape=(28, 28, 1), dtype=tf.float64, name="image"),
        	keras.layers.Conv2D(32, (3, 3), activation='relu'),
        	keras.layers.MaxPooling2D((2, 2)),
        	keras.layers.Flatten(),
        	keras.layers.Dense(100, activation='relu'),
        	keras.layers.Dense(10, activation='softmax')
	])
	'''x0 = keras.Input(shape=(28, 28, 1), dtype=tf.float64)
	x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x0)
	x = keras.layers.MaxPooling2D((2, 2))(x)
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(100, activation='relu')(x)
	y = keras.layers.Dense(10, activation='softmax')(x)
	model = keras.Model(inputs=x0, outputs=y)'''
	optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
	model.compile(
        	optimizer=optimizer,
        	loss='categorical_crossentropy',
        	metrics=['accuracy']
	)
	return model

net = create_model()
net.summary()

history = net.fit(train_images, train_labels, epochs=5, batch_size=60,
		validation_data=(test_images, test_labels))

# Save the entire model as a SavedModel.
net.save('mnist_classifier')

'''image = random.choice(x_test)
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()

image = (image.reshape((1, 28, 28, 1))).astype('float32') / 255.0

digit = np.argmax(model.predict(image)[0], axis=-1)
print("Prediction:", digit)'''
