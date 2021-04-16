import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import trange

np.random.seed(42)


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
        # print('relu back')
        # print(grad_outp, inp)
        # print()
        return grad_outp * np.where(inp <=0 , 0, 1)


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, inp):
        return 1 / (1 + np.exp(-inp))

    def backward(self, inp, grad_outp):
        return grad_outp * inp * (1 - inp)


class Dense(Layer):
    def __init__(self, inp_units, outp_units, learning_rate=1):
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

        # calculate next delta errors for calculating weights in prew layers
        # res = dot product of current error * weights
        # print(grad_outp)
        # print('weights: \n', self.weights)
        res = np.sum(grad_outp * self.weights, axis=1)
        # print(grad_outp,inp.T)
        res = np.expand_dims(res, axis=0)
        # print(f"inp shape: {inp.shape}")
        # print(f"weights shape: {self.weights.shape}")
        # print(f"weights:\n {self.weights}")
        self.biases -= self.learning_rate * grad_outp
        # print(f"gOut:\n {grad_outp}")
        # print(f"res: \n{res}")
        # print(f"inp:\n {inp.T}")
        
        self.weights = self.weights - self.learning_rate*(grad_outp*inp.T)
        # print('weights: \n', self.weights)
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
        for i in trange(epochs):
            for xi, yi in zip(X, y):
                x_test = np.expand_dims(xi, axis=0)
                y_test = np.expand_dims(yi, axis=0)

                activations = [x_test]
                frw = self.forward(x_test)
                activations += frw
                # print(activations[-1])
                acts_without_net = activations[::2]
                grad_out = -(y_test - acts_without_net[-1])
                # print(grad_out)
                # print(grad_out.shape)

                layers = self.layers.copy()
                
                # print(f"len acts: {len(acts_without_net)}")
                # print(f"range: {list(range(len(acts_without_net), 0, -1))}")

                # I could write it better, but i didnt want to)
                for idx, act in enumerate(acts_without_net[::-1]):
                    # print("idx:", idx)
                    
                    current_layer = layers.pop(-1)
                    # print('act:\n', act)
                    grad_out = current_layer.backward(act, grad_out)
                    
                    # print('acts without:\n', acts_without_net[-idx-2])
                    current_layer = layers.pop(-1)
                    grad_out = current_layer.backward(acts_without_net[-idx-2], grad_out)
                    # print(f"grad shape: {grad_out.shape}")
                    if len(layers) == 0:
                        break
            
            
    def evaluate(self, X, y):
        correct = 0
        wrong = 0
        for x_test, y_test in zip(X, y):
            pred = self.predict(x_test)
            if y_test[pred[0]] == 1:
                correct +=1
            else:
                wrong += 1
            
        print(correct, wrong)

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
    network.add_layer(5, 4, activation='sigmoid')
    network.add_layer(3, activation='sigmoid')

    network.fit(x_train, y_train, epochs=10)
    network.evaluate(x_test, y_test)


