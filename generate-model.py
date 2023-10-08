import tensorflow as tf
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
from keras.models import Sequential
from keras.layers import Dropout


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# First, we get the data
df_final = pd.read_csv('data/3/data.csv')

# Data pre-processing
X = df_final.drop(['Date','Open','High','Low','Close','Change','Action'],axis=1).values
y = df_final['Action'].values

# Split Train And Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale And Standard Variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create Artificial Neural Network
model = Sequential()
model.add(LSTM(45, input_shape=(9,1)))
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid')) #Output Layer
model.compile(optimizer= 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 10, epochs = 1320) #Fitness ANN to Train Dataset 

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

