import tensorflow as tf
import tensorflow_datasets as tfds
import os

# disable GPU
tf.config.set_visible_devices([], 'GPU')


#input = tf.keras.Input(shape=(5,))

# create model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

'''output = tf.keras.layers.Dense(5, activation=tf.nn.relu)(input)
output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(output)
model = tf.keras.Model(inputs=input, outputs=output)'''

model.compile()

# Export the model to a SavedModel
model.save('model', save_format='tf')
