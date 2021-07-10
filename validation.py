from tensorflow import keras
import sys, os, glob
import numpy as np
import PIL.Image


imsize = (64, 64)

"""
dog1.jpgというファイル名の画像をGoogle Colab上にアップロードする方法は2通りあります。
1つが、下記のコードを実行し画像をアップロードする方法
from google.colab import files
uploaded = files.upload()
2つが、Colab左メニューの>アイコンを押して、目次、コード スニペット、ファイル
の3つ表示されるますが、右のファイルタブから画像をアップロードする方法です。
このファイルタブをクリックするとアップロードと更新の2つがありますが、
アップロードを押すと画像をアップロードすることが可能です。
"""

testpic     = "/home/student/e18/e185701/clear-rain/MicrosoftTeams-image.png"
keras_param = "./cnn.h5"
types = ['jpg','JPG']
files = []
clear = 0
rain = 0

def load_image(path):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

model = keras.models.load_model(keras_param)

img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd) # 精度の表示
prelabel = np.argmax(prd, axis=1)
if prelabel == 0:
    print(">>> 晴れ")
elif prelabel == 1:
    print(">>> 雨")

# photos_dir = "/home/student/e18/e185701/clear-rain/rain/" 
# for ext in types:
#   file_path = os.path.join(photos_dir, '*.{}'.format(ext))
#   files.extend(glob.glob(file_path))
# for i in files:
#   print(i)
#   img = load_image(i)
#   prd = model.predict(np.array([img]))
#   print(prd) # 精度の表示
#   prelabel = np.argmax(prd, axis=1)
#   if prelabel == 0:
#     clear += 1
#     print(">>> 晴れ" + str(clear))
#   elif prelabel == 1:
#     rain += 1
#     print(">>> 雨"+ str(rain))