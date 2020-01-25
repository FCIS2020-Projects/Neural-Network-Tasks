import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from numba.cuda import threadfence_system

from SingleLayerPerceptron import *
from InputHelper import *
import random

Form = tk.Tk()
Form.title("Single Layer Perceptron Task")
Form.geometry("650x400")

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
combo_algotype.config(values =('Single Layer per.', 'Adaline'))


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
label_feature_input.place(x=60 , y=260+RD)

label_entry_x1 = tk.Label(Form , text = "Feature 1")
label_entry_x1.place(x=40 , y=280+RD)
entry_entry_x1 = tk.Entry(Form, width=10 )
entry_entry_x1.place(x = 40 , y = 300 + RD)

label_entry_x2 = tk.Label(Form , text = "Feature 2")
label_entry_x2.place(x=140 , y=280+RD)
entry_entry_x2 = tk.Entry(Form, width=10 )
entry_entry_x2.place(x = 140 , y = 300 + RD)


data = InputHelper()


def callback(eventObject):
    if combo_algotype.current() == 1  :
        entry_threshold.config(state='normal')
    else:
        entry_threshold.config(state='disabled')

combo_algotype.bind("<<ComboboxSelected>>", callback)



def run():
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


B2 = tk.Button(Form, text="Run", width=50, command=run)
B2.place(x=270, y=110 + RD)

Form.mainloop()
