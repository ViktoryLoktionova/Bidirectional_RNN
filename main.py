import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

N = 10000 #формируем синусоиду со случайным шумом
data = np.array([np.sin(x/20)for x in range (N)])+ 0.1*np.random.randn(N)
plt.plot(data[:100])

off = 3 #формируем обучающую выборку определяем, сколько отсчетов будет
length = off*2+1
X= np.array([np.diag(np.hstack((data[i:i+off], data[i+off+1:i+length]))) for i in range(N-length)]) #входные значения
Y= data[off:N-off-1] #требуемые выходные. Метод diag создает диагональную матрицу.
print(X.shape, Y.shape, sep ='\n')

model = Sequential()
model.add(Input((length-1, length-1)))
model.add(Bidirectional(GRU(2)) ) #двунаправленный слой
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss = 'mean_squared_error', optimizer = Adam(0.01)) #компилируем НС

histiry = model.fit(X, Y, batch_size=32, epochs=10)

M = 200 #делаем 200 прогнозов
XX = np.zeros(M)
XX[:off] = data[:off]
for i in range (M-off-1):
  x = np.diag ( np.hstack( (XX[i:i+off], data[i+off+1:i+length])) ) #формируем входные данные для НС
  x = np.expand_dims(x, axis=0)
  y = model.predict(x)
  XX[i+off] = y

plt.plot(XX[:M])
plt.plot(data[:M])