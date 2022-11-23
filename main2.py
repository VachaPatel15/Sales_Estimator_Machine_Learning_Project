# multiple linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from future.moves import tkinter
from tkinter import *
import pickle

MODEL_PATH = "models/reg_mul.sav"
loaded_model = pickle.load(open(MODEL_PATH,'rb'))

# calculating sales wrt to input
def calculate_sale(arg):
    res = (loaded_model.predict([[tv.get(), radio.get(), newspaper.get()]]))*100
    sale_label.config(text="Expected Sale: {:.2f}/-".format(res[0][0]))

# initialising the tinkter window
master= Tk()
master.geometry("700x400")
master.configure(background='#00FFFF')
master.title('Sales Estimator')

# label to show estimated sale
sale_label = Label(master,bg='#000000', fg="white", font=("Times", 40, "bold"), text="Hello Tkinter!")
sale_label.pack(padx=5, pady=20)

# Scale to take TV advertising investment ammount from user
tv = Scale(master, length=650, label="TV", bg='#CCFFFF', fg="black", from_=0, to=2000, orient=HORIZONTAL , command=calculate_sale)
tv.set(10)
tv.pack(padx=5, pady=5)

# Scale to take Radio advertising investment ammount from user
radio = Scale(master, length=650, label="Radio", bg='#CCFFFF', fg="black",from_=0, to=2000, orient=HORIZONTAL, command=calculate_sale)
radio.set(10)
radio.pack(padx=5, pady=5)

# Scale to take Newspaper advertising investment ammount from user
newspaper = Scale(master, length=650, label="Newspaper", bg='#CCFFFF', fg="black", from_=0, to=2000, orient=HORIZONTAL, command=calculate_sale)
newspaper.set(10)
newspaper.pack(padx=5, pady=5)

# Showing the Application
mainloop()


