from numpy.lib.function_base import append
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import datetime,os

# モデル読み込み用
from keras.models import load_model
#混合行列計算用
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score

classes = ["clear", "rain"]
imsize = 256
num_classes = len(classes)
keras_param = "./model/cnn_imsize"+str(imsize)+".h5"
# keras_param = "./model/sky_imsize"+str(imsize)+"_BDD100K.h5"
"""
データを読み込む関数
"""
def load_data():
    X_train, X_test, y_train, y_test = np.load("./dataset/clear_rain_" + str(imsize) + ".npy", allow_pickle=True)
    # 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data()
model = keras.models.load_model(keras_param)

p_test = model.predict_classes(X_test) # 予測

y_test_2 = []
for i in y_test:
    if i[0] == 1:
        y_test_2.append(0)
    else:
        y_test_2.append(1) 

print(y_test_2.count(0))
print(y_test_2.count(1))        

cm = confusion_matrix(y_test_2, p_test)
print(cm)
print("適合率:"+str(precision_score(y_test_2, p_test,average=None)))
print("再現率"+str(recall_score(y_test_2, p_test,average=None)))
print("F値"+str(f1_score(y_test_2, p_test,average=None)))

df = pd.DataFrame({'正解値':y_test_2, '予測値':p_test})

#誤分類を抽出
df2 = df[df['正解値']!=df['予測値']]

print(df)
print("------------------------------------------------")
print(df2)

for t in range(len(classes)):

  print(f'■ 正解値「{t}」に対して正しく予測（分類）できなかったケース')

  # 正解値が t の行を抽出
  index_list = list(df2[df2['正解値']==t].index.values)
  print(index_list)
  # matplotlib 出力
  n_cols = 7 #7列
  n_rows = ((len(index_list)-1)//n_cols)+1 #indexによって行は変化する

  fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols ,figsize=(6.5, 0.9*n_rows), dpi=120)
  for i,ax in enumerate( np.ravel(ax) ):
    if i < len(index_list):

      p = index_list[i]
      ax.imshow(X_test[p],interpolation='nearest',vmin=0.,vmax=1.,cmap='Greys')

      # # 予測（分類）を左上に表示
      # t = ax.text(1, 1, f'{p_test[p]}', verticalalignment='top', fontsize=8, color='tab:red')
      # t.set_path_effects([pe.Stroke(linewidth=2, foreground='white'), pe.Normal()]) 

      # 予測（分離）に対応する出力層のニューロンの値を括弧で表示
      s = model.predict( np.array([X_test[p]]) ) # 出力層の値 
      s = s[0]
      t = ax.text(5, 2, f'({s[s.argmax()]:.3f})', verticalalignment='top', fontsize=6, color='tab:red')
      t.set_path_effects([pe.Stroke(linewidth=2, foreground='white'), pe.Normal()]) 

      # 目盛などを非表示に
      ax.tick_params(axis='both', which='both', left=False, labelleft=False, 
                     bottom=False, labelbottom=False)

    else :
      ax.axis('off') # 余白処理

  plt.savefig(os.path.join("./fig/error_image/",str(datetime.datetime.today())+".jpg"))