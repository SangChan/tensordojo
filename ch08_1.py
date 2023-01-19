import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
model = tf.keras.Sequential([
    hub.KerasLayer(handle=mobile_net_url, input_shape=(224,224,3), trainable=False)
])
model.summary()

# 그림 8.2 좌측 전체 네트워크 구조 출력 코드
from tensorflow.keras.applications import MobileNetV2

mobilev2 = MobileNetV2()
tf.keras.utils.plot_model(mobilev2)

# 8.2 ImageNetV2-TopImages 불러오기
import os
import pathlib
content_data_url = '/Users/sangchan.lee/workspace/content/sample_data'
data_root_orig = tf.keras.utils.get_file('imagenetV2', 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz', cache_dir=content_data_url, extract=True)
data_root = pathlib.Path(content_data_url+'/datasets/imagenetv2-top-images-format-val')
print(data_root)

for idx, item in enumerate(data_root.iterdir()):
    print(item)
    if idx == 9:
        break