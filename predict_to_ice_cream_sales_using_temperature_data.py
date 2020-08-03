#connect the google colab for dataser file
#from google.colab import drive
#drive.mount('/content/gdrive')

#import libraires
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show


#Load the dataset
sales_df = pd.read_csv('data/SalesData.csv')
#print the information about the Sales Dataset
print(f'Sales Dataset\n {sales_df.head()}')
print(f'Sales Dataset information \n {sales_df.info()}')
print(f'Sales Dataset Statistical information \n {sales_df.describe()}')

sns.distplot(sales_df['Temperature'])
plt.title('Temperature Distribution')
plt.show()

sns.distplot(sales_df['Revenue'])
plt.title('Revenue Distribution')
plt.show()

sns.scatterplot(x='Temperature', y='Revenue', data=sales_df)
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Temperature vs. Revenue')


#create a testing and training Dataset
X_train = sales_df['Temperature']
y_train = sales_df['Revenue']

#Build and train the model
print(f'X Train dateset shape: {X_train.shape} \ny train dataset shape: {y_train.shape}')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units =1, input_shape=[1]))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
epochs_history = model.fit(X_train, y_train, epochs=500)
print(f'Weights of the Model \n{model.get_weights()}')

plt.plot(epochs_history.history['loss'])
plt.title('Model Loss Progress During Trainig')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.show()

plt.scatter(X_train, y_train, color='gray', alpha=0.8, label='Original')
plt.scatter(X_train, model.predict(X_train), color='red',alpha=0.4, label='Predicted')
plt.ylabel('Revenue[Dollars]')
plt.xlabel('Temperature[degC]')
plt.title('Revenue Generated vs. Temperature #Ice Cream Stand')
plt.legend()
plt.show()


for _ in range(100):
    value = int(input('Please input a Temperature value for Revenue Prediction: \n'))
    print('Note: Press "Stop" for the process \n')
    Revenue = model.predict([value])
    print(f'Revenue Prediction for given {value} degC Using Trained ANN is {Revenue}')
    if value == 'Stop' or 'stop':
        break
