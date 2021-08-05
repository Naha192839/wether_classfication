from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
import json

# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["clear", "rain"] #0:clear 1:rain
num_classes = len(classes)
image_size = 256


X_train = []
X_test  = []
y_train = []
y_test  = []

files = []
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        print(i)
        if index == 0:
          X_train.append(data)
          y_train.append(index)
        else:

            # angleに代入される値
            # -20
            # -15
            # -10
            #  -5
            # 0
            # 5
            # 10
            # 15
            # 画像を10度ずつ回転
            for angle in range(-10, 10, 5):

                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                y_train.append(index)
                # FLIP_LEFT_RIGHT　は 左右反転
                img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trains)
                X_train.append(data)
                y_train.append(index)
        
print("晴れの数:"+str(y_train.count(0)))
print("雨の数:"+str(y_train.count(1)))
#validation_splitで均等にとるためにtrainをシャッフル
shuffl_num = np.random.randint(0, 100)
np.random.seed(shuffl_num)
np.random.shuffle(X_train)
np.random.seed(shuffl_num)
np.random.shuffle(y_train)

# #テストデータの確保 20%
X_train = X_train[:int(len(X_train) * 0.8)]
y_train = y_train[:int(len(y_train) * 0.8)]

X_test = X_train[int(len(X_train) * 0.8):]
y_test = y_train[int(len(y_train) * 0.8):]

print("訓練データ:"+str(len(y_train)))
print("テストデータ:"+str(len(y_test)))

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)
xy = (X_train, X_test, y_train, y_test)
np.save("./dataset/clear_rain_"+str(image_size)+".npy", xy)
