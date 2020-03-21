from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

model = tf.keras.Sequential()
# 64개의 유닛을 가진 완전 연결 층을 모델에 추가합니다:
model.add(layers.Dense(64, activation='relu'))
# 또 하나를 추가합니다:
model.add(layers.Dense(64, activation='relu'))
# 10개의 출력 유닛을 가진 소프트맥스 층을 추가합니다:
model.add(layers.Dense(10, activation='softmax'))

# 시그모이드 활성화 층을 만듭니다:
layers.Dense(64, activation='sigmoid')
# 또는 다음도 가능합니다:
layers.Dense(64, activation=tf.keras.activations.sigmoid)

# 커널 행렬에 L1 규제가 적용된 선형 활성화 층. 하이퍼파라미터 0.01은 규제의 양을 조절합니다:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# 절편 벡터에 L2 규제가 적용된 선형 활성화 층. 하이퍼파라미터 0.01은 규제의 양을 조절합니다:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# 커널을 랜덤한 직교 행렬로 초기화한 선형 활성화 층:
layers.Dense(64, kernel_initializer='orthogonal')

# 절편 벡터를 상수 2.0으로 설정한 선형 활성화 층:
layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))

model = tf.keras.Sequential([
# 64개의 유닛을 가진 완전 연결 층을 모델에 추가합니다:
layers.Dense(64, activation='relu', input_shape=(32,)),
# 또 하나를 추가합니다:
layers.Dense(64, activation='relu'),
# 10개의 출력 유닛을 가진 소프트맥스 층을 추가합니다:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
# 평균 제곱 오차로 회귀 모델을 설정합니다.
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',       # 평균 제곱 오차
              metrics=['mae'])  # 평균 절댓값 오차

# 크로스엔트로피 손실 함수로 분류 모델을 설정합니다.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
              
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

# 예제 `Dataset` 객체를 만듭니다:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

# Dataset에서 `fit` 메서드를 호출할 때 `steps_per_epoch` 설정을 잊지 마세요.
model.fit(dataset, epochs=10, steps_per_epoch=30)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

model.fit(dataset, epochs=10,
          validation_data=val_dataset)
          
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
print(result.shape)

inputs = tf.keras.Input(shape=(32,))  # 입력 플레이스홀더를 반환합니다.

# 층 객체는 텐서를 사용하여 호출되고 텐서를 반환합니다.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# 컴파일 단계는 훈련 과정을 설정합니다.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5번의 에포크 동안 훈련합니다.
model.fit(data, labels, batch_size=32, epochs=5)


class MyModel(tf.keras.Model):

def __init__(self, num_classes=10):
  super(MyModel, self).__init__(name='my_model')
  self.num_classes = num_classes
  # 층을 정의합니다.
  self.dense_1 = layers.Dense(32, activation='relu')
  self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

def call(self, inputs):
  # 정방향 패스를 정의합니다.
  # `__init__` 메서드에서 정의한 층을 사용합니다.
  x = self.dense_1(inputs)
  return self.dense_2(x)

model = MyModel(num_classes=10)

# 컴파일 단계는 훈련 과정을 설정합니다.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5번의 에포크 동안 훈련합니다.
model.fit(data, labels, batch_size=32, epochs=5)

class MyLayer(layers.Layer):

def __init__(self, output_dim, **kwargs):
  self.output_dim = output_dim
  super(MyLayer, self).__init__(**kwargs)

def build(self, input_shape):
  # 이 층에서 훈련할 가중치 변수를 만듭니다.
  self.kernel = self.add_weight(name='kernel',
                                shape=(input_shape[1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

def call(self, inputs):
  return tf.matmul(inputs, self.kernel)

def get_config(self):
  base_config = super(MyLayer, self).get_config()
  base_config['output_dim'] = self.output_dim
  return base_config

@classmethod
def from_config(cls, config):
  return cls(**config)

model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# 컴파일 단계는 훈련 과정을 설정합니다.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5번의 에포크 동안 훈련합니다.
model.fit(data, labels, batch_size=32, epochs=5)

callbacks = [
  # `val_loss`가 2번의 에포크에 걸쳐 향상되지 않으면 훈련을 멈춥니다.
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # `./logs` 디렉토리에 텐서보드 로그를 기록니다.
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
          
model = tf.keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 가중치를 텐서플로의 체크포인트 파일로 저장합니다.
model.save_weights('./weights/my_model')

# 모델의 상태를 복원합니다.
# 모델의 구조가 동일해야 합니다.
model.load_weights('./weights/my_model')

# 가중치를 HDF5 파일로 저장합니다.
model.save_weights('my_model.h5', save_format='h5')

# 모델의 상태를 복원합니다.
model.load_weights('my_model.h5')

# 모델을 JSON 포맷으로 직렬화합니다.
json_string = model.to_json()
json_string

import json
import pprint
pprint.pprint(json.loads(json_string))

fresh_model = tf.keras.models.model_from_json(json_string)

yaml_string = model.to_yaml()
print(yaml_string)

fresh_model = tf.keras.models.model_from_yaml(yaml_string)

# 간단한 모델을 만듭니다.
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# 전체 모델을 HDF5 파일로 저장합니다.
model.save('my_model.h5')

# 가중치와 옵티마이저를 포함하여 정확히 같은 모델을 다시 만듭니다.
model = tf.keras.models.load_model('my_model.h5')

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential()
  model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(layers.Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.SGD(0.2)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()

x = np.random.random((1024, 10))
y = np.random.randint(2, size=(1024, 1))
x = tf.cast(x, tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=1024).batch(32)

model.fit(dataset, epochs=1)
