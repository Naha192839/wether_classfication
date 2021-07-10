from tensorflow import keras

import PIL.Image
import os, glob,json
import numpy as np

import shutil

json_file = open('/home/student/e18/e185701/clear-rain/bdd100k-2/labels/det_20/det_train.json', 'r')
json_object = json.load(json_file)
num = len(json_object) 

weather_list = [[],[]] 


photos_dir = "/home/student/e18/e185701/clear-rain/bdd100k/images/100k/train/" 

image_size = 64
num_testdata = 200
keras_param = "./sky_cnn.h5"

X_train = []
X_test  = []
y_train = []
y_test  = []

def load_image(path):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize((image_size,image_size))
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img
model = keras.models.load_model(keras_param)

for i in range(num):
    if json_object[i]["attributes"]["weather"] == "clear":
        weather_list[0].append(photos_dir+json_object[i]["name"])
    elif json_object[i]["attributes"]["weather"] == "partly cloudy":   
        weather_list[0].append(photos_dir+json_object[i]["name"])    
    elif json_object[i]["attributes"]["weather"] == "rainy":   
        weather_list[1].append(photos_dir+json_object[i]["name"])
        
for index,files in enumerate(weather_list):#index 0:clear 1:rainy
    for i,file in enumerate(files):
        if os.path.isfile(file): #ファイルがあれば処理する
            img = load_image(file)
            prd = model.predict(np.array([img]))
            prelabel = np.argmax(prd, axis=1)
            print(prelabel)
            if prelabel == 0:#空がある画像のみをピックアップ
                print("a")
                if index == 0:
                    shutil.move(file, '/home/student/e18/e185701/clear-rain/clear/')
                    print(i)
                else:
                    shutil.move(file, '/home/student/e18/e185701/clear-rain/rain/')
                    print(i)