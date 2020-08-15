#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import ML libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

#import the data
data = pd.read_csv('data/kc_house_data.csv')
print(f"First 5 row of the data \n{data.head()}")

print(f"Data summary \n{data.info()}")


#####Visualize the data####################
#sns.scatterplot(data['sqft_living'], data['price'])
#plt.title('Price change againts sqrt foot of living area')
#plt.xlabel('Sqrt Living Area')
#plt.ylabel('Price')

#plt.figure(figsize=(16,8))
#sns.heatmap(data.corr(), annot=True)

#data.hist(bins=20, figsize=(20,20), alpha=0.6)

#sns.pairplot(data)

#########################
#####Train and Test data####################
#Train and test data
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above',
                     'sqft_basement','waterfront', 'view', 'condition', 'grade','yr_built',
                     'yr_renovated', 'zipcode','lat', 'long', 'sqft_living15', 'sqft_lot15']
#feature slection
X = data[selected_features]
#target selection
y = data['price']
y = y.values.reshape(-1,1)
print(f" X shape: {X.shape}, y shape: {y.shape} ")

#Scale the Dataset
scaler = MinMaxScaler()

X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

#split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
print(f" X_train: {X_train.shape}, X_test: {X_test.shape} ")
print(f" y_train: {y_train.shape}, y_test: {y_test.shape} ")

#Scructure the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
print(model.summary())

model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_history = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

plt.plot(epochs_history.history['loss'], label='Training Loss')
plt.plot(epochs_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress during Training')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Loss')
plt.legend()
plt.show()

y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color='r')
plt.title('Modelprediction vs True Label')
plt.xlabel('True Label')
plt.ylabel('Model Prediction')
plt.show()

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

  print(calcaulate_the_scores(y_test, y_predict, X_test))
