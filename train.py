from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime


classes = ["clear", "rain"]
num_classes = len(classes)
image_size = 256

"""
データを読み込む関数
"""
def load_data():
    X_train, X_test, y_train, y_test = np.load("./clear_rain.npy", allow_pickle=True)
    # 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test
"""
モデルを学習する関数
"""
def train(X, y, X_test, y_test):
    model = keras.Sequential()

    # Xは(1200, 64, 64, 3)
    # X.shape[1:]とすることで、(64, 64, 3)となり、入力にすることが可能です。
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))#64*64の画像を入力 3*3で畳こみ　32のフィルタを出力　ゼロパティングをして元の大きさで出力する
    model.add(Activation('relu')) #relu関数　0以下を0に　ノイズ除去
    model.add(Conv2D(32,(3,3))) #3*3で畳こみ　32のフィルタを出力
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) #マックスプーリング　2*2の行列内での最大値をとることで特徴的な部分を目立たせる
    # model.add(Dropout(0.1))#過学習を抑えるため入力値の10%を0にする

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())#平坦にする　32*32*64
    model.add(Dense(512)) #全結合　次元数512
    model.add(Activation('relu'))
    # model.add(Dropout(0.45))
    model.add(Dense(2)) #全結合　次元数2 あるかない
    model.add(Activation('softmax'))

    # https://keras.io/ja/optimizers/
    # 今回は、最適化アルゴリズムにRMSpropを利用
    opt = RMSprop(lr=0.00005, decay=1e-6)
    #opt = keras.optimizers.Adam(lr=0.00005, decay=1e-6)
    # https://keras.io/ja/models/sequential/
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


    batch_num = 512
    epoch_num = 10
    print("batch:"+str(batch_num))
    print("epoch:"+str(epoch_num))
    history = model.fit(X, y, batch_size=batch_num, epochs=epoch_num, validation_split = 0.1)
    
     # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    # # Plot training & validation accuracy values
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig(os.path.join("./acc_fig/",str(datetime.datetime.today())+"acc.jpg"))
    # plt.clf()

    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.savefig(os.path.join("./loss_fig/",str(datetime.datetime.today())+"loss.jpg"))
    
    # HDF5ファイルにKerasのモデルを保存
    model.save('./cnn.h5')
    
    model.summary()
    
    return model

"""
メイン関数
データの読み込みとモデルの学習を行います。
"""
def main():
    # データの読み込み
    X_train, y_train, X_test, y_test = load_data()

    # モデルの学習
    model = train(X_train, y_train, X_test, y_test)

main()
