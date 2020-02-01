# 케라스와 텐서플로 허브를 사용한 영화 리뷰 텍스트 분류하기
# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전: ", hub.__version__)
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

train_data, test_data = tfds.load(
    name="imdb_reviews",
    split= (tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True)
    
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
    epochs=20,
    verbose=1)
    
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
