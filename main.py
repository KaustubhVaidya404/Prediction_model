
"""
This is a module which can predict an individuals income
(net-worth) by its age

This models score/efficiency is depended on given input
approx score of training data set is 0.85+ and of testing data set is 0.75+

"""

#importing necessary libraries

import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import random

def studentReg(ages_train,networth_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(ages_train,networth_train)
    return reg

#generating random data for model

mininp = int(input("Enter minimum working age :"))
maxinp = int(input("Enter maximum working age :"))

numpy.random.seed(45) #seed value is random it can be any value

ages = []

for val in range(200):
    ages.append(random.randint(mininp,maxinp))
"""
generating random data for networth by taking slope as 6.25
by using simpel list comprehancing method --> y = mx+c
"""

networth = [val*6.25+numpy.random.normal(scale=40) for val in ages]

#converting data into 2D array using numpy
ages = numpy.reshape(numpy.array(ages),(len(ages),1))
networth = numpy.reshape(numpy.array(networth),(len(networth),1))

#Converting this datasets into two data sets test data set and training data set
from sklearn.model_selection import train_test_split

ages_train, ages_test, networth_train, networth_test  = train_test_split(ages, networth)


#passing training dataset to studentReg for training
reg1 = studentReg(ages_train,networth_train)

print("Coefficient : ",reg1.coef_)
print("Intercept : ",reg1.intercept_)

print("Training data : ",reg1.score(ages_train,networth_train))

print("Testing data : ",reg1.score(ages_test,networth_test))

#visualizing the recived data
"""
plt.figure(figsize=(20,20))
sns.regplot(x=ages_train,y=networth_train,scatter=True, color="b",marker="*")
plt.xlabel("--Age--")
plt.ylabel("--Net Worth--")
plt.title("Linear Regression")
"""
#---------UNCOMMENT ANY OF THESE ABOVE GRAPH IS WITHOUT SCATTER METHOD-----------------------------------------#

"""
plt.figure(figsize=(8,8))
plt.scatter(ages_train,networth_train,color="b",label="train data")
plt.scatter(ages_test,networth_test,color="r",label="test data")

plt.plot(ages_test,reg1.predict(ages_test))

plt.xlabel("Age")
plt.ylabel("Net Worth")
plt.legend(loc=2)
"""

plt.show()