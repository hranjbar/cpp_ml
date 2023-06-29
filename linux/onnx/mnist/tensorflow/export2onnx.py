import tensorflow as tf
from tensorflow import keras
import os
import tf2onnx
import onnx

saved_model_path = os.path.join(os.getcwd(), "mnist_classifier")
net = keras.models.load_model(saved_model_path)
net.summary()

#(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
#test_labels = keras.utils.to_categorical(test_labels, num_classes=10)
#test_images = test_images.reshape(-1, 28, 28, 1) / 255.0


input_signature = [tf.TensorSpec((1, 28, 28), tf.float32, name='image')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(net, input_signature, opset=13)
onnx.save(onnx_model, os.path.join(saved_model_path, "mnist_classifier.onnx"))


'''samples = 100
correct = 0
for img, label in zip(test_images[:samples], test_labels[:samples]):
	img = img.reshape(-1, 28, 28, 1) # img's shape must be nhwc
	#print(img.shape) # (1, 28, 28, 1)
	#print(label) # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
	#print(label.argmax())
	y_pred = net.predict(img).reshape(10,)
	#print(y_pred.argmax())
	if y_pred.argmax() == label.argmax():
		correct += 1
print(f"accuracy: {correct / samples * 100:5.2f} out of {samples} samples")'''

#_, accuracy_d = net.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * accuracy_d))
