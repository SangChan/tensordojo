import numpy as np
import tensorflow as tf

X = []
Y = []

for i in range(3000):
    lst = np.random.rand(100)
    idx = np.random.choice(100, 2, replace=False)
    zeros = np.zeros(100)
    zeros[idx] = 1
    X.append(np.array(list(zip(zeros, lst))))
    Y.append(np.prod(lst[idx]))

print(X[0], Y[0])

model = tf.keras.Sequential([
   tf.keras.layers.SimpleRNN(units=30, return_sequences=True, input_shape=[100,2]),
   tf.keras.layers.SimpleRNN(units=30),
   tf.keras.layers.Dense(1) 
])
model.compile(optimizer='adam', loss='mse')
model.summary()