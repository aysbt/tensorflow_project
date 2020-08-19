#import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the plotly express for interactive plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

#Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Model and Training
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix

#mount from google colab
#from google.colab import drive
#drive.mount('/content/drive')

data = pd.read_csv('data/amazon_alexa.tsv', sep='\t')
print(f'First 5 rows of dataset \n{data.head()}')
print(f'Datasets information \n{data.info()}')
###############Data #visulation#############################################

feedback =  data['feedback'].value_counts()
rating = data['rating'].value_counts()
variation =data['variation'].value_counts()

fig =make_subplots(rows=1, cols=3,
                   specs=[[{'type':'domain'},{'type':'domain'},{'type':'domain'}]])
fig.add_trace(go.Pie(values=feedback.values,
                     labels=['Positive','Negative'],
                     title='Distribution of Feedback',
                     textinfo = 'label+percent',
                     marker_colors=['rgb(56, 75, 126)', 'rgb(18, 36, 37)']
                     ), 1, 1)
fig.add_trace(go.Pie(values=rating.values,
                     labels=rating.index,
                     title='Distribution of Rating',
                     textinfo = 'label+percent',
                     ), 1, 2)
fig.add_trace(go.Pie(values=variation.values,
                     labels=variation.index,
                     title='Distribution of Variation',
                     textinfo = 'label+percent',
                     ), 1, 3)
fig = go.Figure(fig)
fig.update_layout(showlegend=False)
fig.show()

fig2 = px.bar(variation, x=variation.index, y=variation.values, color=variation.index)
fig2.update_layout(showlegend=False)
fig2.show()

#sns.boxenplot(data['variation'], data['rating'], palette = 'spring')
#plt.title("Variation vs Ratings")
#plt.xticks(rotation = 90)
#plt.show()

#sns.violinplot(data['feedback'], data['rating'], palette = 'cool')
#plt.title("feedback wise Mean Ratings")
#plt.show()

##################Data Cleaning##########################################
#get dummies for variation

data.drop(['date','rating'], axis=1, inplace=True)
variation_dummies = pd.get_dummies(data['variation'], drop_first=True)
data.drop(['variation'], axis=1, inplace=True)
data = pd.concat([data, variation_dummies], axis=1)
#count vectorixer for revirew column
vectorizer = CountVectorizer()
data_vectorizer = vectorizer.fit_transform(data['verified_reviews'])
reviews = pd.DataFrame(data_vectorizer.toarray())
print(reviews.head())
data.drop(['verified_reviews'], axis=1, inplace=True)
data = pd.concat([data, reviews], axis=1)
print(data.head())

y = data['feedback']
y = y.values.reshape(-1,1)
X = data.drop(['feedback'], axis=1)

#plit the data train and test sample
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)

#create the model
ANN_classifier = tf.keras.models.Sequential()
ANN_classifier.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(X_train.shape[1],)))
ANN_classifier.add(tf.keras.layers.Dense(units=400, activation='relu'))
ANN_classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
print(ANN_classifier.summary())

ANN_classifier.compile(optimizer= 'Adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs_history = ANN_classifier.fit(X_train, y_train, epochs=20)


#plot the loss epochs_history
plt.plot(epochs_history.history['loss'], label='Training Loss')
plt.title('Model Loss Progress during Training')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Loss')
plt.legend()
plt.show()
#plot the accuracy
plt.plot(epochs_history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy Progress during Training')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Accuracy')
plt.legend()
plt.show()

#confusion confusion_matrix
y_predict = ANN_classifier.predict(X_test)
y_predict = (y_predict > 0.5)

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
plt.show()
