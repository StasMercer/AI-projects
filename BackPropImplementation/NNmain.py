import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import trange

import matplotlib
import matplotlib.pyplot as plt

# np.random.seed(42)


class Layer:
    """
    This is just a dummy class that is supposed to represent the general
    functionality of a neural network layer. Each layer can do two things:
     - forward pass - prediction
     - backward pass - training
    """

    def __init__(self):
        pass

    def forward(self, inp):
        # a dummy layer returns the input
        return inp

    def backward(self, inp, grad_outp):
        pass


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, inp):
        return np.maximum(0, inp)

    def backward(self, inp, grad_outp):

        return grad_outp * np.where(inp <=0 , 0, 1)


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, inp):
        return 1 / (1 + np.exp(-inp))

    def backward(self, inp, grad_outp):
        return grad_outp * inp * (1 - inp)


class Dense(Layer):
    def __init__(self, inp_units, outp_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(
            loc=0.0, scale=np.sqrt(2 / (inp_units + outp_units)),
            size=(inp_units, outp_units))
        self.biases = np.ones(outp_units)
        self.biases = np.expand_dims(self.biases, axis=0)
        self.neuron_count = outp_units

    def get_neuron_count(self):
        return self.neuron_count

    def forward(self, inp):
        return np.dot(inp, self.weights) + self.biases

    def backward(self, inp, grad_outp):

        # calculate deltas for every neuron in this layer by getting sum 
        # of products of prew delta and weight
        # example: we have 2 output neurons and 3 input
        # deltas from out neurons: [[3, 5]]
        # weights: [[1, 3], [2, 1], [3, 8]]
        # res: [18, 11, 49]
        # res is our deltas for 3 input neurons, and we return it 
        res = np.sum(grad_outp * self.weights, axis=1)

        # add 1 more dim for res, so it would be: [[18, 11, 49]]
        res = np.expand_dims(res, axis=0)
        
        #update biases 
        self.biases -= self.learning_rate * grad_outp

        # update weights
        self.weights = self.weights - self.learning_rate*(grad_outp*inp.T)

        return res

    def __str__(self) -> str:
        return f"Dense {self.neuron_count}"
        

class MLP():
    def __init__(self):
        self.layers = []

    def add_layer(self, neuron_count, inp_shape=None, activation='ReLU'):
        if len(self.layers) == 0 and inp_shape is None:
            raise ValueError("Must defined input shape for first layer")

        if inp_shape is None:
            inp_shape = self.layers[-2].get_neuron_count()

        self.layers.append(Dense(inp_shape, neuron_count))
        if activation == 'sigmoid':
            self.layers.append(Sigmoid())
        elif activation == 'ReLU':
            self.layers.append(ReLU())
        else:
            raise ValueError("Unknown activation function", activation)

    def forward(self, X):
        activations = []

        layer_input = X
        for l in self.layers:
            
            activations.append(l.forward(layer_input))
            layer_input = activations[-1]

        return activations

    def predict(self, X):
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def fit(self, X, y, epochs=20):
        loss = 1
        losses = []
        pbar = trange(epochs)
        for i in pbar:

            for xi, yi in zip(X, y):
                x_train = np.expand_dims(xi, axis=0)
                y_train = np.expand_dims(yi, axis=0)
                
                # add input as first array in activations
                activations = [x_train]
                frw = self.forward(x_train)
                activations += frw
                loss = np.square(np.subtract(y_train, activations[-1])).mean()
                
                    
                # for our implementation with delta rule, we do not need net products
                acts_without_net = activations[::2]

                # calculate initial gradient of error
                grad_out = -(y_train - acts_without_net[-1])


                layers = self.layers.copy()

                # I could write it better, but i didnt want to)
                
                # so for every activation from last to first
                # I take last layer, give current gradient to it
                # (it is error for now), calculate new gradient by 
                # multiplying previous one with gradient of the current layer with respect to activation
                for idx, act in enumerate(acts_without_net[::-1]):

                    # take activation, calculate grad
                    current_layer = layers.pop(-1)
                    grad_out = current_layer.backward(act, grad_out)

                    # take dense, calculate grad and update weights
                    current_layer = layers.pop(-1)
                    grad_out = current_layer.backward(acts_without_net[-idx-2], grad_out)
                    
                    # this is something I didnt want to do, but life sucks :)
                    if len(layers) == 0:
                        break

            losses.append(loss)
            if i % 5 == 0:
                pbar.set_description(f"Loss : {loss}")
        
        _, ax = plt.subplots()
        ax.plot(losses)
        ax.set(xlabel="Epochs", ylabel="Loss", title="Learning Procces")
        plt.show()
            
            
                
            
    # simple functin to test our network
    def evaluate(self, X, y):
        correct = 0
        total = 0
        for x_test, y_test in zip(X, y):
            pred = self.predict(x_test)
            if y_test[pred[0]] == 1:
                correct +=1
            
            total += 1
            
        print(f"Network test accuracy: {(correct / total)*100:.2f} %")

if __name__ == '__main__':
    df = pd.read_csv('iris.csv')

    df['class'] = df['class'].map({
        "Iris-setosa": [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1]
    })

    df['class'] = df['class'].apply(np.array)

    y_train = df['class']

    df.drop(columns='class', inplace=True)

    x_train = df

    x_train, x_test,y_train,y_test=train_test_split(x_train.to_numpy(), y_train.to_numpy(), test_size=.3)

    network = MLP()
    network.add_layer(8, 4, activation='ReLU')
    # network.add_layer(5, activation='sigmoid')
    network.add_layer(3, activation='sigmoid')

    network.fit(x_train, y_train, epochs=50)
    network.evaluate(x_test, y_test)


