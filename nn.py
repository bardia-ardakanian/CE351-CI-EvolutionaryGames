import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes: list):
        """
        Initiate neural netwrok
        layer_sizez: sizes of input layer, hidden layer and output layer. (e.g. 4, 10, 2)
        """

        self.input_size = layer_sizes[0]
        self.hidden_sizes = layer_sizes[1]
        self.output_size = layer_sizes[2]

        self.weights = [np.random.normal(0, 2/self.input_size, size=(self.hidden_sizes, self.input_size)),
                        np.random.normal(0, 2 / self.hidden_sizes, size=(self.output_size, self.hidden_sizes))]

        self.biases = [np.zeros((self.hidden_sizes, 1)),
                       np.zeros((self.output_size, 1))]


    def activation(self, x: np.ndarray):
        """
        Activation function
        """
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x)) 
        
        return sigmoid(x)


    def forward(self, x: list):
        """
        Feed forward phase
        """
        x = np.array(x).reshape(-1, 1)
        
        _hidden = self.activation(self.weights[0] @ x + self.biases[0])
        _output = self.activation(self.weights[1] @ _hidden + self.biases[1])

        return _output
