'''
CNN
モデル、重み読み込み
特徴量出力

'''


import sys
import numpy as np
import math
import scipy.misc
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dense
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import pylab as plt
import os

nb_classes = 72
# # input image dimensions
IMG_SIZE = 32
# # IMG_SIZE, IMG_COLS = 127, 128

BATCH_SIZE = 128
STEP = 50
COLOR_CHANNELS=3

class train_data():
    def __init__(self):pass

    def load_data(self):
        # ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
        ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

        self.X_train = np.zeros([nb_classes * 160, IMG_SIZE, IMG_SIZE],
                                dtype=np.float32)


        # ここの処理は謎
        for i in range(nb_classes * 160):
            self.X_train[i] = scipy.misc.imresize(ary[i],
                                                  (IMG_SIZE, IMG_SIZE))
            # self.X_train[i] = scipy.misc.imresize(ary[i],
            #                                       (IMG_SIZE, IMG_SIZE), mode='F')


        # なぜか漢字の"平"が入ってたので削除
        for i in range(160):
            self.X_train = np.delete(self.X_train,960,0)
        # なぜか漢字の"開"も入ってた...
        for i in range(160):
            self.X_train = np.delete(self.X_train,10200,0)


        # # 規格化(最大値1にしたらいまいち)
        self.X_train = self.X_train.astype(np.float32)/np.max(self.X_train)
        self.Y_train = self.X_train.astype(np.float32)/np.max(self.X_train)

        self.X_train = np.subtract(self.X_train,1)
        self.X_train = np.power(self.X_train,2)
        self.X_train = np.negative(self.X_train)
        print(np.max(self.X_train))
        print(np.min(self.X_train))

        self.X_train = np.add(self.X_train,1)
        print(np.max(self.X_train))
        print(np.min(self.X_train))

        self.X_train = np.power(self.X_train,1/2)
        print(np.max(self.X_train))
        print(np.min(self.X_train))

        self.X_train = self.X_train.astype(np.float32)

        # self.X_train = self.X_train.astype(np.float32)/np.max(self.X_train)
        # self.Y_train = self.X_train.astype(np.float32)/np.max(self.X_train)
        # self.X_train = self.X_train.astype(np.float32)/(np.max(self.X_train)/1)
        # self.Y_train = self.X_train.astype(np.float32)/(np.max(self.X_train)/1)
        # self.Y_train = self.X_train.astype(np.float32)

        print(self.X_train.shape)
        print(np.max(self.X_train))
        print(np.min(self.X_train))

        # リサイズ
        self.X_train = self.X_train.reshape(self.X_train.shape[0],
                                            IMG_SIZE, IMG_SIZE, 1)
        self.Y_train = self.X_train.reshape(self.X_train.shape[0],
                                            IMG_SIZE, IMG_SIZE, 1)

        # # 一応
        # self.X_train = np.array(self.X_train)
        # self.Y_train = np.array(self.Y_train)

        print(self.X_train.shape)
        print(self.Y_train.shape)

def img_dropout(mydata):
    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
    datagen.fit(mydata.X_train)
    return datagen

class cnn_net():
    def __init__(self): pass

    # # 初期重み設定??
    # def my_init(self,shape, name=None):
    #     value = np.random.random(shape)
    #     return K.variable(value,name=name)

    def model_structure(self,model):
        input_img = Input(shape=(IMG_SIZE,IMG_SIZE,1))
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        self.encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        x = Convolution2D(8, 3, 3, activation='relu',
                        border_mode='same')(self.encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        self.decoded = Convolution2D(1, 3, 3, activation='sigmoid',
                                     border_mode='same')(x)

        self.autoencoder = Model(input_img, self.decoded)


        op = "adadelta"
        op = "adam"

        loss = "mean_squared_error"
        loss = "binary_crossentropy"
        # self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        # self.autoencoder.compile(optimizer='adam',
        #                          loss='mean_squared_logarithmic_error')
        self.autoencoder.compile(optimizer=op,loss=loss)



def plot_result(mynet,mydata):
    # テスト画像を変換
    decoded_imgs = mynet.autoencoder.predict(mydata.X_train)

    # 何個表示するか
    n = 10

    li = np.arange(50,len(mydata.X_train),160)

    plt.figure(figsize=(20, 4))
    # for i in range(n):
    for i in range(n):
        # オリジナルのテスト画像を表示
        ax = plt.subplot(2, n, i+1)
        plt.imshow(mydata.X_train[li[i]].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 変換された画像を表示
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[li[i]].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_hidden(mynet,X_train):
    # input_img = Input(shape=(IMG_SIZE,IMG_SIZE,))
    n = 10
    encoded_imgs = mynet.autoencoder.predict(X_train)

    # encoded_imgs = mynet.autoencoder.predict(X_train[:n])
    # encoder = Model(input_img, mynet.encoded)
    # encoded_imgs = encoder.predict(X_train[:n])
    # encoded_imgs = encoder.predict(x_test[:n])

    plt.figure(figsize=(10, 8))
    for i in range(n):
        for j in range(8):
            ax = plt.subplot(8, n, j*n + i+1)
            plt.imshow(encoded_imgs[i][j], interpolation='none')
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def get_feat(mynet,mydata):
    print(mynet.encoded)



def save_model(mynet):
    json_string = mynet.autoencoder.to_json()
    f_model = './model'

    open(os.path.join(f_model,'cnn_model.json'), 'w').write(json_string)
    mynet.autoencoder.save_weights(os.path.join(f_model,'cnn_model_weights.hdf5'))


def main():
    mydata = train_data()
    mydata.load_data()

    f_model = './model'
    weights_filename = 'cnn_model_weights.hdf5'

    mynet = cnn_net()
    model = Sequential()
    mynet.model_structure(model)

    # # model読み込み
    # json_string = open(os.path.join(f_model, model_filename)).read()
    # mynet.autoencoder = model_from_json(json_string)

    # mynet.autoencoder.summary()

    op = "adam"
    loss = "binary_crossentropy"
    mynet.autoencoder.compile(optimizer=op,loss=loss)

    # 重み読み込み
    mynet.autoencoder.load_weights(os.path.join(f_model,weights_filename))

    plot_hidden(mynet,mydata.X_train)
    # get_feat(mynet,mydata)


if __name__ == "__main__" :
    main()
