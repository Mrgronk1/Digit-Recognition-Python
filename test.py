from main import forward_layer
from main import sigmoid
from main import softmax
import numpy as np
from mnist import MNIST

model = np.load("model.npz")
mnist = MNIST(".")
mnist.gz = True

weights1 = model["weights1"]
weights2 = model["weights2"]

biases1 = model["biases1"]
biases2 = model["biases2"]

images,labels = mnist.load_testing()

correct = 0

#testing loop
for image, label in zip(images, labels):

    image = np.array(image) / 255

    hidden_layer = forward_layer(image, weights1, biases1, sigmoid)
    output_layer = forward_layer(hidden_layer, weights2, biases2, softmax)

    prediction = np.argmax(output_layer)
    
    if prediction == label:
        correct += 1

accuracy = (correct / len(images)) * 100
print(f"Accuracy: {accuracy}")