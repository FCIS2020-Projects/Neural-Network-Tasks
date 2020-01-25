import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from numba.cuda import threadfence_system

from SingleLayerPerceptron import *
from InputHelper import *
from MultiLayerPerceptron import *
import random

Form = tk.Tk()
Form.title("Single Layer Perceptron Task")
Form.geometry("650x420")

RD = 40 ;

def Create_Plot(tX,tY):
    xx = tX[1:].tolist()
    yy = tY[1:].tolist()

    X = [float(i) for i in xx]
    Y = [float(i) for i in yy]

    plt.scatter(X[0:50],Y[0:50])
    plt.scatter(X[50:100],Y[50:100])
    plt.scatter(X[100:150],Y[100:150])
    plt.xlabel(tX[0])
    plt.ylabel(tY[0])
    plt.show()


combo_algotype_sel = tk.StringVar(Form)
combo_algotype_sel.set("Algorithm Type")

combo_algotype = ttk.Combobox(Form, width=20 , textvariable = combo_algotype_sel )
combo_algotype.place(x = 250 , y = 10)
combo_algotype.config(values =('Single Layer per.', 'Adaline' ,'Multi Layer Per.'))


label_X = tk.Label(Form , text = "Choose Feature in X-Axis")
label_X.place(x = 20 , y = 10 + RD)
label_Y = tk.Label(Form , text = "Choose Feature in Y-Axis")
label_Y.place(x = 20 , y = 40 + RD)


selection1 = tk.StringVar(Form)
selection1.set("X-Axis")

selection2 = tk.StringVar(Form)
selection2.set("Y-Axis")

comboBox1 = ttk.Combobox(Form, width=10 , textvariable = selection1)
comboBox1.place(x = 170 , y = 10 + RD)
comboBox1.config(values =('X1', 'X2','X3','X4'))

comboBox2 = ttk.Combobox(Form, width=10 , textvariable = selection2)
comboBox2.place(x = 170 , y = 40 + RD)
comboBox2.config(values =('X1', 'X2','X3','X4'))


def update_plot():
    if(comboBox1.current() == -1 or comboBox2.current() == -1):
        tk.messagebox.showinfo("ERROR !" , "Please Choose Features")
    else:
        Create_Plot(data.dataset[:,comboBox1.current()], data.dataset[:,comboBox2.current()])


B = tk.Button(Form , text = "Show Chart" , width = 50 , command = update_plot)
B.place(x=270,y=30 + RD)

# ///////////////////////////////////
label_class1 = tk.Label(Form , text = "Choose Class 1")
label_class1.place(x = 20 , y = 70 + RD)
label_class2 = tk.Label(Form , text = "Choose Class 2")
label_class2.place(x = 20 , y = 100 + RD)

selection3 = tk.StringVar(Form)
selection3.set("Class 1")

selection4 = tk.StringVar(Form)
selection4.set("Class 2")

comboBox3 = ttk.Combobox(Form, width=10 , textvariable = selection3)
comboBox3.place(x = 170 , y = 70 + RD)
comboBox3.config(values =('Iris-setosa', 'Iris-versicolor','Iris-virginica'))

comboBox4 = ttk.Combobox(Form, width=10 , textvariable = selection4)
comboBox4.place(x = 170 , y = 100 + RD)
comboBox4.config(values =('Iris-setosa', 'Iris-versicolor','Iris-virginica'))

label_LearningRate = tk.Label(Form , text = "Learning Rate")
label_epochs = tk.Label(Form , text = "Epochs")
label_bias = tk.Label(Form , text = "bias")
label_LearningRate.place(x = 20 , y = 130 + RD)
label_epochs.place(x = 20 , y = 160 + RD)
label_bias.place(x = 20 , y = 190 + RD)

entry_learningRate = tk.Entry(Form, width=10  )
entry_learningRate.place(x = 170 , y = 130 + RD)

entry_epochs = tk.Entry(Form, width=10 )
entry_epochs.place(x = 170 , y = 160 + RD)


selection5 = tk.StringVar(Form)
selection5.set("bias")

comboBox5 = ttk.Combobox(Form, width=10 , textvariable = selection5)
comboBox5.place(x = 170 , y = 190 + RD)
comboBox5.config(values =('No', 'Yes'))

label_thresshold = tk.Label(Form , text = "MSE threshold")
label_thresshold.place(x=20 , y=220+RD)
entry_threshold = tk.Entry(Form, width=10 )
entry_threshold.place(x = 170 , y = 220 + RD)

label_feature_input = tk.Label(Form , text = "Enter Features Values")
label_feature_input.place(x=400 , y=260+RD)

label_entry_x1 = tk.Label(Form , text = "Feature 1")
label_entry_x1.place(x=40+340 , y=280+RD)
entry_entry_x1 = tk.Entry(Form, width=10 )
entry_entry_x1.place(x = 40+340 , y = 300 + RD)

label_entry_x2 = tk.Label(Form , text = "Feature 2")
label_entry_x2.place(x=140+340 , y=280+RD)
entry_entry_x2 = tk.Entry(Form, width=10 )
entry_entry_x2.place(x = 140+340 , y = 300 + RD)

#Multi Layer Perception
label_MLP_input = tk.Label(Form , text = "Enter Layer Details")
label_MLP_input.place(x=75 , y=260+RD)

label_num_hidden_layer = tk.Label(Form , text = "# Hideen Layers")
label_num_hidden_layer.place(x=40 , y=280+RD)
entry_num_hidden_layer = tk.Entry(Form, width=10 )
entry_num_hidden_layer.place(x = 50 , y = 300 + RD)


label_num_nodes_layer = tk.Label(Form , text = "# Nodes in Layer")
label_num_nodes_layer.place(x=140 , y=280+RD)
entry_num_nodes_layer = tk.Entry(Form, width=10 )
entry_num_nodes_layer.place(x = 150 , y = 300 + RD)


label_activation_fn = tk.Label(Form , text = "Activation Function")
label_activation_fn.place(x=75 , y=330+RD)

selection6 = tk.StringVar(Form)
selection6.set("Activation Fn")

comboBox6 = ttk.Combobox(Form, width=10 , textvariable = selection6)
comboBox6.place(x = 85 , y = 350 + RD)
comboBox6.config(values =('Sigmoid', 'Hyperbolic'))

data = InputHelper()


def combo_enable(event):
    label_class1.config(state= event)
    label_class2.config(state= event)
    label_X.config(state= event)
    label_Y.config(state= event)

    comboBox1.config(state= event)
    comboBox2.config(state= event)
    comboBox3.config(state=event)
    comboBox4.config(state=event)

    if event == 'disabled' :
        label_MLP_input.config(state='normal')
        label_num_hidden_layer.config(state='normal')
        label_num_nodes_layer.config(state='normal')
        label_activation_fn.config(state='normal')

        entry_num_hidden_layer.config(state = 'normal' )
        entry_num_nodes_layer.config(state = 'normal')
        comboBox6.config(state = 'normal')
    else:
        label_MLP_input.config(state = 'disabled')
        label_num_hidden_layer.config(state = 'disabled')
        label_num_nodes_layer.config(state = 'disabled')
        label_activation_fn.config(state = 'disabled')

        entry_num_hidden_layer.config(state='disabled')
        entry_num_nodes_layer.config(state='disabled')
        comboBox6.config(state='disabled')
def callback(eventObject):
    if combo_algotype.current() == 0 :
        label_thresshold.config(state='disabled')
        entry_threshold.config(state='disabled')
        combo_enable('normal')
    elif combo_algotype.current() == 1 :
        label_thresshold.config(state='normal')
        entry_threshold.config(state='normal')
        combo_enable('normal')
    elif combo_algotype.current() == 2  :
        label_thresshold.config(state='disabled')
        entry_threshold.config(state='disabled')
        combo_enable('disabled')


combo_algotype.bind("<<ComboboxSelected>>", callback)



def run():

    if combo_algotype.get() != 'Multi Layer Per.' :
        if combo_algotype.current() == -1 :
            tk.messagebox.showinfo("ERROR !", "Please Choose Algorithm")
            return
        if comboBox1.current() == -1 or comboBox2.current() == -1:
            tk.messagebox.showinfo("ERROR !", "Please Choose Features")
            return

        if comboBox3.current() == -1 or comboBox4.current() == -1:
            tk.messagebox.showinfo("ERROR !", "Please Choose Classes")
            return

        if entry_learningRate.get() == "":
            tk.messagebox.showinfo("ERROR !", "Please Enter Learning Rate")
            return

        if entry_epochs.get() == "":
            tk.messagebox.showinfo("ERROR !", "Please Enter Epochs")
            return

        if comboBox5.current() == -1:
            tk.messagebox.showinfo("ERROR !", "Please Choose is it Bias or not")
            return

        if combo_algotype.get() == 'Adaline':
            if entry_threshold.get() == "":
                tk.messagebox.showinfo("ERROR !", "Please Enter MSE threshold")
                return

        data.set_data(comboBox1.current(), comboBox2.current(), comboBox3.current(), comboBox4.current())

        inputs = data.training_input
        outputs = data.training_output

        perceptron = SingleLayerPerceptron(float(entry_learningRate.get()), comboBox5.current())

        if (combo_algotype_sel.get() == 'Single Layer per.' ):
            msg = perceptron.algorithm(inputs, outputs, int(entry_epochs.get()))
            tk.messagebox.showinfo("Event", msg)

        elif(combo_algotype_sel.get() == 'Adaline' ):
            msg = perceptron.adaline(inputs, outputs, int(entry_epochs.get()) , float(entry_threshold.get()))
            tk.messagebox.showinfo("Event", msg)
        perceptron.draw_line(data)

        inputs_test = data.testing_input
        outputs_test = data.testing_output

        msg2 = perceptron.testing(inputs_test, outputs_test)
        tk.messagebox.showinfo("Event", msg2)
        show_confusion_matrix(perceptron.confusion_matrix)

        if entry_entry_x1.get() == "" or entry_entry_x2.get() == "" :
            return
        else:
            intput_vector = np.array([float(entry_entry_x1.get()) , float(entry_entry_x2.get())])
            net_value = perceptron.calc_net_value(intput_vector)
            prediction = perceptron.signum(net_value)

            if(prediction == 1):
                label_result = tk.Label(Form, text="Flower Category is : " + comboBox3.get())
                label_result.place(x=220, y=300 + RD)
            else:
                label_result = tk.Label(Form, text="Flower Category is : " + comboBox4.get())
                label_result.place(x=220, y=300 + RD)

    if(combo_algotype_sel.get() == 'Multi Layer Per.' ):
        # number_layer, size_layer, learning_rate, bias, epochs, activation_fn
        data.set_data_MLP()
        inputs = data.training_input
        outputs = data.training_output

        # msg = perceptron.adaline(inputs, outputs, int(entry_epochs.get()), float(entry_threshold.get()))

        MLP = MultiLayerPerceptron(int(entry_num_hidden_layer.get()),int(entry_num_nodes_layer.get()),float(entry_learningRate.get()),comboBox5.current())
        msg = MLP.algorithm(inputs,outputs,int(entry_epochs.get()),comboBox6.current())
        tk.messagebox.showinfo("Event", msg)

        inputs_test = data.testing_input
        outputs_test = data.testing_output

        msg = MLP.testing(inputs_test,outputs_test,comboBox6.current())
        tk.messagebox.showinfo("Event", msg)

        show_confusion_matrix_MLP(MLP.confusion_matrix)

def show_confusion_matrix(confusion_matrix):
    frame = tk.Frame()

    l = tk.Label(frame, text="")
    l.grid(row=0, column=0)

    l = tk.Label(frame, text="Predicted: " + comboBox3.get())
    l.grid(row=0, column=1)

    l = tk.Label(frame, text="Predicted: " + comboBox4.get())
    l.grid(row=0, column=2)

    l = tk.Label(frame, text="Actual: " + comboBox3.get())
    l.grid(row=1, column=0)

    l = tk.Label(frame, text="Actual: " + comboBox4.get())
    l.grid(row=2, column=0)
    for i in range(2):
        for j in range(2):
            l = tk.Label(frame, text=confusion_matrix[i, j])
            l.grid(row=i + 1, column=j + 1)
    frame.place(x=270, y=150 + RD)

def show_confusion_matrix_MLP(confusion_matrix):
    frame = tk.Frame()

    l = tk.Label(frame, text="")
    l.grid(row=0, column=0)

    l = tk.Label(frame, text="Class 1")
    l.grid(row=0, column=1)

    l = tk.Label(frame, text="Class 2")
    l.grid(row=0, column=2)

    l = tk.Label(frame, text="Class 3")
    l.grid(row=0, column=3)

    l = tk.Label(frame, text="" + comboBox3.get())
    l.grid(row=1, column=0)

    l = tk.Label(frame, text="" + comboBox4.get())
    l.grid(row=2, column=0)

    l = tk.Label(frame, text="" + comboBox4.get())
    l.grid(row=3, column=0)
    for i in range(3):
        for j in range(3):
            l = tk.Label(frame, text=confusion_matrix[i, j])
            l.grid(row=i + 1, column=j + 1)
    frame.place(x=270, y=150 + RD)


B2 = tk.Button(Form, text="Run", width=50, command=run)
B2.place(x=270, y=110 + RD)

Form.mainloop()
