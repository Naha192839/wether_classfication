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

image_size = 256
num_testdata = 200
keras_param = "/home/student/e18/e185701/clear-rain/model/sky_imsize"+str(image_size)+"_BDD100K.h5"

X_train = []
X_test  = []
y_train = []
y_test  = []

types = ['jpg','JPG']
files = []
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

"""
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
"""
photos_dir = "/home/student/e18/e185701/clear-rain/sky?" 
for ext in types:
  file_path = os.path.join(photos_dir, '*.{}'.format(ext))
  files.extend(glob.glob(file_path))
for i in files:  
  #信頼度を出す
  img = load_image(i)
  prd = model.predict(np.array([img]))
  prelabel = np.argmax(prd, axis=1)
  if prelabel == 0:
      if max(prd[0]) >= 0.9:
          print(max(prd[0]))
          print(i)
          shutil.move(i, '/home/student/e18/e185701/clear-rain/sky/')
          print("---------------------------------")
  elif prelabel == 1:
      if max(prd[0]) >= 0.9:
          print(max(prd[0]))
          print(i)
          shutil.move(i, '/home/student/e18/e185701/clear-rain/non_sky/')
          print("---------------------------------")
