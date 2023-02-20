import tensorflow as tf

# 9.18 BSD500 데이터세트 불러오기
tf.keras.utils.get_file('/Users/sangchan.lee/workspace/content/bsd_images.zip', 'http://bit.ly/35pHZlC', extract=True)

#!unzip /content/bsd_images.zip

# 9.19 이미지 경로 저장 및 확인
import pathlib
image_root = pathlib.Path('/Users/sangchan.lee/workspace/content/images')

all_image_paths = list(image_root.glob('*/*.jpg'))
print(all_image_paths[:10])

# 9.20 이미지 확인
import PIL.Image as Image
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
for c in range(9):
    plt.subplot(3,3,c+1)
    plt.imshow(plt.imread(all_image_paths[c]))
    plt.title(all_image_paths[c])
    plt.axis('off')
plt.show()