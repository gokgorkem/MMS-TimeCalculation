import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras import backend
from keras.layers import Dense
import tensorflow as tf
import io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from google.colab import files
uploaded = files.upload()

data = pd.read_csv(io.BytesIO(uploaded['bakimTest.csv']))
print(data)

days = data['time_dif']

days

X = data.drop(['time_dif'], axis=1)

X.head

y = days

y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=50)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_test

X_train

model = Sequential()
model.add(Dense(5, activation='relu', input_dim=5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

from tensorflow.python.keras.metrics import Metric
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics='MeanAbsoluteError')



tf.keras.backend.clear_session()

history = model.fit(X_train,y_train, batch_size=8, validation_data=(X_test,y_test), epochs=2, verbose=1)

company_id = 1
tezgah_id = 7
durum_kodu = 2
sablon_id= 1
durus_durumu= 0

test = pd.DataFrame({"company_id":[company_id],"tezgah_id":[tezgah_id],"durum_kodu":[durum_kodu],"sablon_id":[sablon_id],"durus_durumu":[durus_durumu]})

print(test)

test  = scaler.transform(test)



testDeneme = model.predict(X_test)

testDeneme

print(test)

deneme = model.predict(test)

deneme

print (int(deneme))