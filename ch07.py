import numpy as np
import tensorflow as tf

X = []
Y = []
for i in range(6):
    lst = list(range(i,i+4))
    X.append(list(map(lambda c: [c/10], lst)))
    Y.append((i+4)/10)

X = np.array(X)
Y = np.array(Y)

for i in range(len(X)):
    print(X[i], Y[i])

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[4,1]),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()