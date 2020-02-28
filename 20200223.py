from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# tf.keras.layers 패키지에서 층은 객체입니다. 층을 구성하려면 간단히 객체를 생성하십시오.
# 대부분의 layer는 첫번째 인수로 출력 차원(크기) 또는 채널을 취합니다.
layer = tf.keras.layers.Dense(100)
# 입력 차원의 수는 층을 처음 실행할 때 유추할 수 있기 때문에 종종 불필요합니다. 
# 일부 복잡한 모델에서는 수동으로 입력 차원의 수를 제공하는것이 유용할 수 있습니다.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# 층을 사용하려면, 간단하게 호출합니다.
layer(tf.zeros([10, 5]))

# layer는 유용한 메서드를 많이 가지고 있습니다. 예를 들어, `layer.variables`를 사용하여 층안에 있는 모든 변수를 확인할 수 있으며, 
# `layer.trainable_variables`를 사용하여 훈련 가능한 변수를 확인할 수 있습니다. 
# 완전 연결(fully-connected)층은 가중치(weight)와 편향(biases)을 위한 변수를 가집니다. 
layer.variables

# 또한 변수는 객체의 속성을 통해 편리하게 접근 가능합니다. 
layer.kernel, layer.bias

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)

class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])