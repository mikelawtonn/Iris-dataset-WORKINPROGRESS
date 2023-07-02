#IMPORT LIBRARIES AND MODULES
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



#LOADING THE DATASET

#Reading in dataset
df = pd.read_csv("Iris.csv")

#Getting first five rows
df.head()

#Deleting the Id column as it is not needed.
df = df.drop(columns = ["Id"])
print(df)


#Before preprocessing we need to get some information about the data. df.info() shows that each attribute
#(column) has 150 values, there are 4 attributes in total and this dataset has no null values. So
#we don't need to do any data preprocessing with regards to missing value handling such as deleting the rows with
#null values.
df.info()

#Checking if the datset is imbalanced, this means that the datset has skewed class proportions - there is a
#significant difference between the number of records belonging to each class. I want the data to be balanced so
#if the dataset is imbalanced then I would need to balance the classes in the preprocessing phase. However, this
#is not necessary as the data is balanced - there are 50 records in each class.

#Display number of samples of each class
df['Species'].value_counts()

#I want to check the distribution for each attribute. I will do this by generating a histogram for each attribute.
#ideally, I want to use data that is in a normal distribtuion as this way i can learn more quickly from the data.
#The histograms show that the Sepal Width attribute has a normal distribution. We can see that the Petal Length
#histogram has a curve that is separated into two sections this suggests that two of the classes are merged for Petal
#Length, this is also the case for Petal Width.

#Histogram for Petal Width
df['PetalWidthCm'].hist()

#Histogram for Petal Length
df['PetalLengthCm'].hist()

#Histogram for Petal Length
df['SepalLengthCm'].hist()

#Histogram for Petal Length
df['SepalWidthCm'].hist()


#Scatter plots to further help determine which attributes are best to seperate the classes.

#Scatter plot for Sepal Length vs Sepal Width. The scatter plot suggests that Sepal Length + Sepal Width attributes
#may help distinguish between Setosa and the other two classes but it would then be difficult to classify the other
#two classes as, for these two attributes, the classes of virginica and versiclolor are merged.

# create list of colors and class labels
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # Generates the scatter plot
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
#Label for x-axis
plt.xlabel("Sepal Length")
#Label for y-axis
plt.ylabel("Sepal Width")
#legend for the plot
plt.legend()

#Scatter plot for Petal Length vs Petal Width. This combination of attributes looks like it could help seperate all
#classes.

colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()

#Scatter plot for Petal Width vs Sepal Width. This combination of attributes looks like it could help seperate all
#classes.


colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i in range(3):
    # filter data on each class
    x = df[df['Species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['PetalWidthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Width")
plt.ylabel("Sepal Width")
plt.legend()