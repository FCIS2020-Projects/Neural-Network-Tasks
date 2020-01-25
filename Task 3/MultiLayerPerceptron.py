import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random

# 1. User Input:
# • Enter number of hidden layers
# • Enter number of neurons in each hidden layer
# • Enter learning rate (eta)
# • Enter number of epochs (m)
# • Add bias or not (Checkbox)
# • Choose to use Sigmoid or Hyperbolic Tangent sigmoid as the activation     function
#
# 2. Initialization:
# • Number of features = 4.
# • Number of classes = 3.
# • Weights + Bias = small random numbers
#
# 3. Classification:
# • Sample (single sample to be classified).

class MultiLayerPerceptron:
    def __init__(self, number_hiden_layer , size_layer,learning_rate, bias):
        self.number_hiden_layer = number_hiden_layer
        self.number_layer = number_hiden_layer + 1
        self.size_layer = size_layer

        self.learning_rate = learning_rate
        self.is_bias = bias
        self.outNodes = 3
        self.numFeatures = 4

        if bias == 0:
            self.Bias = 0
        else:
            self.Bias = self.init_bias()

        self.weights = self.init_weights()

        self.z_values = []
        self.z_der_values = []
        self.segma_values = []
        self.confusion_matrix = []

    def sigmoid(self, x):
        result = 1.0 / (1.0 + np.exp(-x))
        return result

    def sigmoid_der(self, z):
        z_der = z * (1-z)
        return z_der

    def hyperbolic_tangent_sigmoid(self, x):
        result = np.tanh(x)
        return result

    def hyperbolic_tangent_sigmoid_der(self, z):
        z_der = (1-z)*(1+z)
        return z_der

    def calc_net_value(self, input_vector,weights_M,Bias):
        net_value = np.dot(input_vector, np.transpose(weights_M))
        if self.is_bias != 0 :
            net_value += Bias
        return net_value

    def init_bias(self):
        Bias=[]
        for layer in range(self.number_layer):
            temp = []
            if layer == self.number_layer-1 :
                for node in range(self.outNodes):
                    t = random.uniform(-1, 1)
                    temp.append(t)
            else:
                for node in range(self.size_layer):
                    t = random.uniform(-1, 1)
                    temp.append(t)
            Bias.append(temp)
        return Bias #list"layers" of list"nodes" = (float)

    def init_weights(self):
        weights = []
        np.random.seed(1)
        for i in range(self.number_hiden_layer):
            weights_neuron = []
            for j in range(self.size_layer):
                if i == 0 :
                    weights_M = 2 * np.random.random(self.numFeatures) - 1
                else:
                    weights_M = 2 * np.random.random(self.size_layer) - 1

                weights_neuron.append(weights_M)
            weights.append(weights_neuron)

        weights_neuron = []
        for k in range(self.outNodes):
            weights_M = 2 * np.random.random(self.size_layer) - 1
            weights_neuron.append(weights_M)

        weights.append(weights_neuron)

        return weights #list"layer" of list"Nodes" = np.array()

    def feedforward(self,inputs , activation_fn):
        for i in range(self.number_layer):
            z_layer = []
            z_der_layer = []
            for j in range(len(self.weights[i])):
                if i == 0 :
                    if self.is_bias != 0 :
                        net_value = self.calc_net_value(inputs,np.transpose(self.weights[i][j]),self.Bias[i][j])
                    else:
                        net_value = self.calc_net_value(inputs,np.transpose(self.weights[i][j]),0)

                else :
                    if self.is_bias != 0:
                        net_value = self.calc_net_value(self.z_values[i-1],np.transpose(self.weights[i][j]),self.Bias[i][j])
                    else:
                        net_value = self.calc_net_value(self.z_values[i-1],np.transpose(self.weights[i][j]),0)

                if activation_fn == 0:
                    z = self.sigmoid(net_value)
                    z_der = self.sigmoid_der(z)
                else:
                    z = self.hyperbolic_tangent_sigmoid(net_value)
                    z_der = self.hyperbolic_tangent_sigmoid_der(z)

                z_layer.append(z)
                z_der_layer.append(z_der)

            self.z_values.append(z_layer)
            self.z_der_values.append(z_der_layer) #list"Layer" of list"nodes" = float
        return self.z_values[-1] #return Output Layer

    def sigma(self, bool, z, y, lnum):

        if bool == 0:
            c = (z - y) * z * (1 - z)
            return c

        else:
           # print("****hiden sigma generation****")
            sum = 0
            i=0;
            for node in range(len(self.z_values[lnum])):
                for edge in range (len (self.weights[lnum][node])):

                 #   print("suuuuuuuuuuuuuu",sum)

                #    print("self.weights[lnum][node][edge]",self.weights[lnum][node][edge],"self.segma_values[i][node]",self.segma_values[i][node])
                    sum += self.weights[lnum][node][edge] * self.segma_values[i][node]
            i =i+1
            c = sum * z * (1 - z)
        #    print("c:",c,"z:",z,"sum",sum)
            return c

    def backward(self, actual_y):
        # print("************************backward************************")
        self.segma_values.clear()
        i = self.number_layer
        while (i >0):
            j = 0
            sigma_result = []
            if (i == self.number_layer ):  # for output layer
                while (j < 3):
                    s = self.sigma(0, self.z_values[i - 1][j], actual_y[j], i-1);
                    # print("s:", s)
                    sigma_result.append(s)
                    j = j + 1

            else:
                while (j < self.size_layer):  # for hiden layer
                    s = self.sigma(1, self.z_values[i - 1][j], 0, i);
                    # print("s:", s)
                    sigma_result.append(s)
                    j = j + 1
            i = i - 1
            self.segma_values.append(sigma_result)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def updateweights(self , input):
        # print("************************updateweights************************")
        inv=len(self.segma_values)-1
        for i in range(self.number_layer):
            for w in range(len(self.weights[i])):
                if i == 0:

                    self.weights[i][w] = self.weights[i][w] + (self.learning_rate * input *self.segma_values[inv-i][w])
                    if self.is_bias !=0:
                        self.Bias[i][w] = self.Bias[i][w] + (self.learning_rate * self.segma_values[inv-i][w]) *self.is_bias
                    # print("i:",i,"w:",w,"-----", self.weights[i][w])
                else:

                    self.weights[i][w] = self.weights[i][w] + (self.learning_rate * np.array(self.z_values[i][w]).astype(float) *self.segma_values[inv-i][w])
                    if self.is_bias !=0:
                        self.Bias[i][w] = self.Bias[i][w] + (self.learning_rate* np.array(self.z_values[i][w]).astype(float) * self.is_bias)
                    # print("i:",i,"w:",w,"-----",self.weights[i][w])

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    def algorithm(self, inputs, outputs, epochs,activation_fn):
        print("epochs",epochs,"inputs",len(inputs))
        for epoch in range(epochs):
            for i in range(len(inputs)):
                prediction = self.feedforward(inputs[i],activation_fn)

                tempPrediction = []
                if prediction[0] > prediction[1] and prediction[0] > prediction[2]:
                    tempPrediction.append(1)
                    tempPrediction.append(0)
                    tempPrediction.append(0)
                elif prediction[1] > prediction[0] and prediction[1] > prediction[2]:
                    tempPrediction.append(0)
                    tempPrediction.append(1)
                    tempPrediction.append(0)
                else:
                    tempPrediction.append(0)
                    tempPrediction.append(0)
                    tempPrediction.append(1)

                if (int(outputs[i])==0  and tempPrediction[0]<1) or (int(outputs[i])==1 and tempPrediction[1]<1) or (int(outputs[i])==2 and tempPrediction[2]<1):
                    self.backward(prediction)
                    self.updateweights(inputs[i])

        msg = "Training Done Successfully - Multi Layer Perceptron"
        return msg

    def testing(self,inputs,outputs,activationfn):
        wrong = 0
        right = 0
        confusion_matrix = np.zeros((3, 3))
        for i in range(len(inputs)):
            prediction = self.feedforward(inputs[i], activationfn)
            c = -1;
            if prediction[0] > prediction[1] and prediction[0] > prediction[2]:
                prediction[0] = 1
                prediction[1] = 0
                prediction[2] = 0
                c=0
            elif prediction[1] > prediction[0] and prediction[1] > prediction[2]:
                prediction[0] = 0
                prediction[1] = 1
                prediction[2] = 0
                c=1
            else :
                prediction[0] = 0
                prediction[1] = 0
                prediction[2] = 1
                c=2
            if (int(outputs[i]) == 0 and prediction[0] < 1) or (int(outputs[i]) == 1 and prediction[1] < 1) or (
                    int(outputs[i]) == 2 and prediction[2] < 1):
                wrong = wrong + 1
            else:
                right = right + 1

            x = int(outputs[i])
            y = c

            confusion_matrix[x, y] += 1
        accuracy = right / (right+wrong)
        self.confusion_matrix = confusion_matrix

        msg = "testing Done Successfully - Accuracy Is " + str(accuracy)
        return msg