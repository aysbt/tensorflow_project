#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#load the data
data = pd.read_csv('data/Celsius+to+Fahrenheit.csv')
print('Temperature data')
print(data.head())

print(f'Datasets info: \n{data.info()}')
print(f'Datasets statistics: \n{data.describe()}')

#we are going to use all data as for train
x_train = data['Celsius']
y_train = data['Fahrenheit']

model = tf.keras.Sequential()
#we have one weight and one bias
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
print(model.summary())

model.compile(optimizer= tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')
epoch_history = model.fit(x_train, y_train, epochs=200)

celcius_data = [0, 20, 30, 180]
f_result = []
for i in celcius_data:
    result = model.predict([i])
    print(f'{i} Celcius: {result} Fahrenheit')
    f_result.append(result)
