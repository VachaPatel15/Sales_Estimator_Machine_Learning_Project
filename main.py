# how much sales the company will get by advertising

# for simplicity, lets consider a single variable(television) and find out how much sales the
# company will get

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os



dataset = pd.read_csv('SalesEstimator.csv')
# print(dataset.head()) prints first 5 rows

x = dataset['TV'].values.reshape(-1, 1)
y = dataset['Sales'].values.reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black')
plt.title('Sales Estimation')
plt.xlabel('Money spend on TV ads ($)')
plt.ylabel('Sales ($)')
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x, y, random_state=42, test_size=0.2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black')
plt.plot(x_test, y_pred, color='blue',linewidth=2)
plt.xlabel('Money spend on TV ads ($)')
plt.ylabel('Sales ($)')
plt.show()

# calculating the coefficients
# print(reg.coef_)
# print(reg.intercept_)

# calculating the R squared value
from sklearn.metrics import r2_score
# print(r2_score(y_test, y_pred))

output = reg.predict([[230.1]])
# print(output)

# multiple linear regression
x = dataset.drop(['Sales'], axis=1)
y = dataset['Sales'].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
mul_reg = LinearRegression()
mul_reg.fit(x_train, y_train)

y_pred = mul_reg.predict(x_test)
# print(mul_reg.coef_)
# print(mul_reg.intercept_)

from sklearn.metrics import r2_score
# print(r2_score(y_test, y_pred))

# taking input from user
print("Enter amount you will invest on: ")
tv = float(input("TV: "))
radio = float(input("Radio: "))
newspaper = float(input("Newspaper: "))

output = (mul_reg.predict([[tv, radio, newspaper]]))*100
print("You will get ${:.2f} sales by advertising ${} on TV, ${} on radio and ${} on newspaper."\
      .format(output[0][0] if output else "not predictable",tv,radio,newspaper))

# saving the model
if not os.path.exists('models'):
    os.makedirs('models')
MODEL_PATH = "models/reg_mul.sav"
pickle.dump(mul_reg, open(MODEL_PATH,'wb'))




