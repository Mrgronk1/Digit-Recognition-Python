import numpy as np
from mnist import MNIST

mnist = MNIST(".")
mnist.gz = True

images,labels = mnist.load_training()

learning_rate = 0.01

def main():
    weights1 = (np.random.randn(128, 784)) / np.sqrt(784) 
    weights2 = (np.random.randn(10, 128)) / np.sqrt(128)

    biases1 = np.zeros(128)
    biases2 = np.zeros(10)

    for epoch in range(30):

        total_loss = 0

        for image, label in zip(images, labels):

            image = np.array(image) / 255

            hidden_layer = forward_layer(image, weights1, biases1, sigmoid)
            output_layer = forward_layer(hidden_layer, weights2, biases2, softmax)

            loss = cross_entropy(output_layer, label)

            #gradient desent for output layer 
            target = np.zeros(10)
            target[label] = 1
            output_error = output_layer - target

            weights2_gradients = np.outer(output_error, hidden_layer)
            biases2_gradients = output_error

            #gradient desent for hidden layer
            hidden_error = np.dot(output_error, weights2)
            sigmoid_derivative = hidden_layer * (1 - hidden_layer)
            hidden_delta = hidden_error * sigmoid_derivative
            weights1_gradients = np.outer(hidden_delta, image)
            biases1_gradients = hidden_delta

            #update weights and biases for hidden layer
            weights1 -= learning_rate * weights1_gradients
            biases1 -= learning_rate * biases1_gradients

            #update weights and biases for output layer
            weights2 -= learning_rate * weights2_gradients
            biases2 -= learning_rate * biases2_gradients

            total_loss += loss

        total_loss = total_loss / len(images)
        print(f"total loss: {total_loss}, Epoch amount: {epoch}" )

    np.savez(
    "model.npz",
    weights1=weights1,
    biases1=biases1,
    weights2=weights2,
    biases2=biases2
)

    print("Model saved successfully!")

def weighted_sum(inputs, weights , bias):

    total = 0

    for input, weight in zip(inputs, weights):
        total += (input * weight) 
    
    return total + bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z)) 

def cross_entropy(predictions, label):
    return -np.log(predictions[label])

# def prosess_neuron(inputs, weights, bias):
#     return sigmoid(weighted_sum(inputs, weights, bias))

def forward_layer(inputs, weights, biases, func):
    i = np.dot(inputs, weights.T) + biases
    return func(i)

if __name__ == "__main__":
    main()