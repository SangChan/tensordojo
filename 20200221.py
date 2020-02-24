import tensorflow as tf

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.square(2)+tf.square(3))

x = tf.matmul([[1]],[[2,3]])
print(x)
print(x.shape)
print(x.dtype)

import numpy as np

ndarray = np.ones([3,3])
print("텐서플로 연산은 자동적으로 넘파이 연결을 텐서로 변환합니다.")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print("그리고 넘파이 연산은 자동적으로 텐서를 넘파이 배열로 변환합니다.")
print(np.add(tensor,1))
print(".numpy() 메서드는 텐서를 넘파이 배열로 변환합니다.")
print(tensor.numpy())

x = tf.random.uniform([3, 3])

print("GPU 사용이 가능한가 : "),
print(tf.test.is_gpu_available())

print("텐서가 GPU #0에 있는가 : "),
print(x.device.endswith('GPU:0'))

import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# CPU에서 강제 실행합니다.
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# GPU #0가 이용가능시 GPU #0에서 강제 실행합니다.
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# CSV 파일을 생성합니다.
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print('ds_tensors 요소:')
for x in ds_tensors:
  print(x)

print('\nds_file 요소:')
for x in ds_file:
  print(x)