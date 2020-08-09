#import Libraires
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import ML Libraires
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import  train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#Upload file from google colab
#from google.colab import drive
#drive.mount('/content/drive')

#Load the file
bike = pd.read_csv('data/bike_sharing_daily.csv')
print(f'First 5 row of the bike dataset \n{bike.head()}')
print(f'Information about the bike dataset \n{bike.info()}')
print(f'Description of the bike dataset \n{bike.describe()}')

#clean up the dataset
#numerical data
X_numerical = bike[['temp','hum','windspeed','cnt']]

#Categorical dataset
X_cat = bike[['season','yr','mnth','holiday','weekday','workingday','weathersit']]

#use one hot code on categorical daat to encode
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)

#concat the numerical and categorical dataset
X_all = pd.concat([X_cat, X_numerical], axis=1)
#print(X_all.columns)
X = X_all.iloc[:, :-1].values
y = X_all.iloc[:, -1:].values

print(f'X shape: {X.shape}\ny shape: {y.shape}')

#normalize the target
scaler = MinMaxScaler()
y =scaler.fit_transform(y)

#split the train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')

#Train the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu',input_shape=(35,)))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
print(model.summary())

model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_history = model.fit(X_train, y_train, epochs=25, batch_size=50, validation_split=0.2)

plt.plot(epochs_history.history['loss'], label='Training Loss')
plt.plot(epochs_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress during Training')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Loss')
plt.legend()

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

def calcaulate_the_scores(y_test, y_pred, X_test):
  # X_test shape for features
  k = X_test.shape[1]
  # length of X_test
  n = len(X_test)

  MSE = mean_squared_error(y_test, y_pred)
  MAE = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
  RMSE = np.sqrt(MSE)
  print(f'RMSE: {RMSE} \nMSE: {MSE} \nMAE: {MAE} \nr2: {r2} \nadj_r2: {adj_r2}')


# we can plot the unscalar results
y_predict = model.predict(X_test)
y_predict_original = scaler.inverse_transform(y_predict)
y_test_original = scaler.inverse_transform(y_test)

calcaulate_the_scores(y_test_original, y_predict_original, X_test)
