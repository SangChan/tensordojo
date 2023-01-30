import tensorflow as tf
# 8.8 Stanford Dog Dataset을 Kaggle에서 불러오기

# 2020.02.01 현재 kaggle의 Stanford Dog Dataset 파일 구조가 변경되었습니다. 
# kaggle API를 사용하는 대신에 아래 링크에서 파일을 직접 받아오도록 수정되었습니다.
tf.keras.utils.get_file('/Users/sangchan.lee/workspace/content/labels.csv', 'http://bit.ly/2GDxsYS')
tf.keras.utils.get_file('/Users/sangchan.lee/workspace/content/sample_submission.csv', 'http://bit.ly/2GGnMNd')
tf.keras.utils.get_file('/Users/sangchan.lee/workspace/content/train.zip', 'http://bit.ly/31nIyel')
tf.keras.utils.get_file('/Users/sangchan.lee/workspace/content/test.zip', 'http://bit.ly/2GHEsnO')

import os
os.environ['KAGGLE_USERNAME'] = 'user_id' # 독자의 캐글 ID
os.environ['KAGGLE_KEY'] = 'user_api_token' # 독자의 캐글 API Token
# !kaggle competitions download -c dog-breed-identificationu

# 8.10 labels.csv 파일 내용 확인
import pandas as pd
label_text = pd.read_csv('/Users/sangchan.lee/workspace/content/labels.csv')
print(label_text.head())

# 8.11 labels.csv 정보 확인
label_text.info()

# 8.12 견종 수 확인
label_text['breed'].nunique()

# 8.13 이미지 확인
import PIL.Image as Image
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
for c in range(9):
    image_id = label_text.loc[c, 'id']
    plt.subplot(3,3,c+1)
    plt.imshow(plt.imread('/Users/sangchan.lee/workspace/content/train/' + image_id + '.jpg'))
    plt.title(str(c) + ', ' + label_text.loc[c, 'breed'])
    plt.axis('off')
plt.show()