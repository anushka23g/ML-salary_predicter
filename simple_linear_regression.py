import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values   #experience
y=dataset.iloc[:,1].values      #salary

# split the dataset to train it and then test on it 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3, random_state=0)
 
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test, color = 'red')
#plt.show()
plt.plot(x_train, regressor.predict(x_train))
plt.show()