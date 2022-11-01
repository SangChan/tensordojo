import tensorflow as tf
import numpy as np

hello = tf.constant('Hello, TensorFlow!') 
print(hello)

a = tf.constant(10)
b = tf.constant (32)
c = tf.add(a,b).numpy()
print(c)

@tf.function
def adder(a, b):
    return a+b
x = tf.Variable(tf.ones(shape=[None, 3]), dtype=tf.float32)
print(x)