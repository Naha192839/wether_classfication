from tensorflow import keras
import sys, os, glob
import numpy as np
import PIL.Image
import numpy as np
import tensorflow
import cv2
import datetime

# 画像用
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
# モデル読み込み用
from keras.models import load_model
# Grad−CAM計算用
from tensorflow.keras import models


imsize = 256


testpic     = "/home/student/e18/e185701/clear-rain/sky?/320e54dc-18430c0d.jpg"
keras_param = "./model/sky_imsize"+str(imsize)+"_BDD100K.h5"
types = ['jpg','JPG']
files = []
clear = 0
rain = 0

def load_image(path):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize((imsize,imsize))
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img
    
def grad_cam(input_model, x, layer_name):
    """
    Args: 
        input_model(object): モデルオブジェクト
        x(ndarray): 画像
        layer_name(string): 畳み込み層の名前
    Returns:
        output_image(ndarray): 元の画像に色付けした画像
    """

    # 画像の前処理
    # 読み込む画像が1枚なため、次元を増やしておかないとmode.predictが出来ない
    X = np.expand_dims(x, axis=0)
    preprocessed_input = X.astype('float32') / 255.0    

    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tensorflow.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # 勾配を計算
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tensorflow.cast(output > 0, 'float32')
    gate_r = tensorflow.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    # 重みを平均化して、レイヤーの出力に乗じる
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, (imsize,imsize), cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # ヒートマップを計算
    heatmap = cam / cam.max()

    # モノクロ画像に疑似的に色をつける
    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    # RGBに変換
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    # もとの画像に合成
    output_image = (np.float32(rgb_cam) + x / 2)  

    return output_image

model = keras.models.load_model(keras_param)
#-----------------------------------------------------

img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd) # 精度の表示
prelabel = np.argmax(prd, axis=1)

img = img_to_array(load_img(testpic, target_size=(imsize,imsize)))
target_layer = 'conv2d_3'
cam = grad_cam(model, img, target_layer)
save_img(os.path.join("./gradcam_fig/",str(datetime.datetime.today())+".jpg"),cam)

if prelabel == 0:
    print(">>> 晴れ")
elif prelabel == 1:
    print(">>> 雨")

#----------------------------------------------------------------------------------

# photos_dir = "/home/student/e18/e185701/clear-rain/clear" 
# for ext in types:
#   file_path = os.path.join(photos_dir, '*.{}'.format(ext))
#   files.extend(glob.glob(file_path))
# for i in files:  
#   #信頼度を出す
#   img = load_image(i)
#   prd = model.predict(np.array([img]))
#   prelabel = np.argmax(prd, axis=1)

  
#   if prelabel == 0:
#     clear += 1
#     # print(">>> 晴れ" + str(clear))
#     # print("---------------------------------")
#   elif prelabel == 1:
#     rain += 1
#     print(i)
#     save_img(os.path.join("./gradcam_fig/",str(datetime.datetime.today())+"-"+os.path.basename(i)),img)
#     print(prd) # 精度の表示
#     img = img_to_array(load_img(i, target_size=(imsize,imsize)))
#     target_layer = 'conv2d_3'
#     cam = grad_cam(model, img, target_layer)
#     save_img(os.path.join("./gradcam_fig/",str(datetime.datetime.today())+".jpg"),cam)
#     print(">>> 雨"+ str(rain))
#     print("---------------------------------")
