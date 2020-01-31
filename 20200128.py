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

split = tfds.Split.TRAIN + tfds.Split.VALIDATION +  tfds.Split.TEST

test_data = tfds.load(
    name="imdb_reviews",
    split=split,
    as_supervised=True)