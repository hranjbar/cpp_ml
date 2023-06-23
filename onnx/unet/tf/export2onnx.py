import tensorflow as tf
from tensorflow import keras
import os
import tf2onnx
import onnx

saved_model_path = os.path.join(os.getcwd(), "model")
net = keras.models.load_model(saved_model_path)
net.summary()


input_signature = [tf.TensorSpec((1, 3, 128, 128), tf.float32, name='spect_vol')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(net, input_signature, opset=13)
onnx.save(onnx_model, os.path.join(saved_model_path, "unet.onnx"))


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
