import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random


class SingleLayerPerceptron:
    def __init__(self, learning_rate, bias):
        self.learning_rate = learning_rate
        self.is_bias = bias

        if bias == 0:
            self.bias = 0
        else:
            x = random.uniform(-1, 1)
            self.bias = x

        np.random.seed(1)
        # Get Range of weights is -1 to 1
        self.weights_M = 2 * np.random.random(2) - 1
        self.acc = 0
        self.confusion_matrix = []

    def signum(self, x):
        if x >= float(0):
            return 1
        else:
            return -1

    def calc_net_value(self, input_vector):
        net_value = np.dot(input_vector, np.transpose(self.weights_M))
        net_value += self.bias
        return net_value

    def algorithm(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                net_value = self.calc_net_value(inputs[i])
                prediction = self.signum(net_value)

                if outputs[i] != prediction:
                    error = outputs[i] - prediction
                    self.weights_M = self.weights_M + (self.learning_rate*error*inputs[i])
                    self.bias = self.bias + self.learning_rate*error*self.is_bias

                    # print(self.weights_M, self.bias)

        msg = "Training Done Successfully - Single Layer Perception"
        return msg

    def adaline(self, inputs, outputs, epochs , threshold):

        flag = True
        for epoch in range(epochs):
            for i in range(len(inputs)):
                prediction = self.calc_net_value(inputs[i])

                error = outputs[i] - prediction
                self.weights_M = self.weights_M + (self.learning_rate*error*inputs[i])
                self.bias = self.bias + self.learning_rate*error*self.is_bias

            meanSqaureError = 0
            for i in range(len(inputs)):
                meanSqaureError += pow(outputs[i] - self.calc_net_value(inputs[i]),2)
            meanSqaureError = (1/len(inputs))*meanSqaureError
            print("MOATAZ >> " , meanSqaureError)
            if meanSqaureError < threshold :
                break

        msg = "Training Done Successfully - Adaline"
        return msg

    def testing(self, inputs, outputs):
        wrong = 0
        right = 0
        confusion_matrix = np.zeros((2, 2))
        for i in range(len(inputs)):
            net_value = self.calc_net_value(inputs[i])
            prediction = self.signum(net_value)

            if outputs[i] != prediction:
                wrong = wrong+1
            else:
                right = right+1

            x = 1 if outputs[i] > 0 else 0
            y = 1 if prediction > 0 else 0

            confusion_matrix[x, y] += 1

        self.acc = (right / (right+wrong))*100
        self.confusion_matrix = confusion_matrix
        msg = "Testing Finish Accuracy is " + str(self.acc)
        return msg

    def draw_line(self, data):
        inputs = data.training_input
        x = inputs[:, 0]
        y = (self.weights_M[0]*x + self.bias)/-self.weights_M[1]
        plt.plot(x, y, color="red", linewidth=3)
        plt.scatter(inputs[0:30, 0], inputs[0:30, 1])
        plt.scatter(inputs[30:60, 0], inputs[30:60, 1])
        plt.xlabel(data.dataset[0, data.feature1])
        plt.ylabel(data.dataset[0, data.feature2])
        plt.show()

