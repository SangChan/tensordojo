import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

X = np.array(X)
Y = np.array(Y)

history = model.fit(X[:2560], Y[:2560], epochs=100, validation_split=0.2)

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()