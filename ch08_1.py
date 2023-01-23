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
import glob
content_data_url = '/Users/sangchan.lee/workspace/content/sample_data'
data_root_orig = tf.keras.utils.get_file('imagenetV2', 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz', cache_dir=content_data_url, extract=True)
data_root = pathlib.Path(content_data_url+'/datasets/imagenetv2-top-images-format-val')
print(data_root)

for idx, item in enumerate(data_root.iterdir()):
    if idx == 9:
        break

# 8.4 ImageNet 라벨 텍스트 불러오기
label_file = tf.keras.utils.get_file('label', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
label_text = None
with open(label_file, 'r') as f:
    label_text = f.read().split('\n')[:-1]
print(len(label_text))
print(label_text[:10])
print(label_text[-10:])

# 폴더 이름이 wordnet의 단어로 수정되었기 때문에, nltk 패키지에서 wordnet을 다운받습니다.
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# wordnet과 인터넷에 올라온 label 텍스트는 조금씩 다르기 때문에 차이를 없애기 위해서 아래의 전처리 작업을 진행합니다.
label_text = [c.lower().replace('-','').replace('_','').replace(' ','') for c in label_text]

# 8.5 이미지 확인
import PIL.Image as Image
import matplotlib.pyplot as plt
import random

all_image_paths = list(data_root.glob('**/*.jpeg'))
all_image_paths = [str(path) for path in all_image_paths]
# 이미지를 랜덤하게 섞습니다.
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print('image_count:', image_count)

plt.figure(figsize=(12,12))
for c in range(9):
    print("======================")
    print("process : #",c+1)
    image_path = random.choice(all_image_paths)
    plt.subplot(3,3,c+1)
    plt.imshow(plt.imread(image_path))
    # idx = int(image_path.split('/')[-2]) + 1
    # plt.title(str(idx) + ', ' + label_text[idx])
    print("random image path =",image_path)
    word = wordnet.synset_from_pos_and_offset('n',int(image_path.split('/')[-2][1:]))
    print("word =",word)
    word = word.name().split('.')[0].replace('-','').replace('_','').replace(' ','')
    plt.title(str(label_text.index(word)) + ', ' + word)
    plt.axis('off')
    print("======================")
plt.show()