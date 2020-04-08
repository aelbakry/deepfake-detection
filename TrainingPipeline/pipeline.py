import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm,trange
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
from torchvision.transforms import ToTensor
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import torch.nn as nn
# import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

max_frames = 10
df_train1 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_0/metadata.json')
df_train2 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_1/metadata.json')
df_train3 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_2/metadata.json')
df_train4 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_3/metadata.json')


df_train_list = [df_train1 , df_train2]
df_train = df_train2

LABELS = ['REAL','FAKE']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()




def get_paths(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_0/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths2(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_1/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths3(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_2/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths4(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_3/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def read_img(path):
    frames = []
    for i in range(max_frames):
        frames.append(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB))
    return frames


paths=[]
y=[]


images1 = list(df_train1.columns.values)
print(len(images1))
for x in images1:

    try:
        paths.append(get_paths(x))
        y.append(to_categorical(np.full((max_frames), LABELS.index(df_train1[x]['label'])), num_classes=2))
    except Exception as err:
        # print(err)
        pass

images2 = list(df_train2.columns.values)
print(len(images2))
for x in images2:

    try:
        paths.append(get_paths2(x))
        y.append(to_categorical(np.full((max_frames), LABELS.index(df_train2[x]['label'])), num_classes=2))
    except Exception as err:
        # print(err)
        pass

images3 = list(df_train3.columns.values)
print(len(images3))
for x in images3:

    try:
        paths.append(get_paths3(x))
        y.append(to_categorical(np.full((max_frames), LABELS.index(df_train3[x]['label'])), num_classes=2))
    except Exception as err:
        # print(err)
        pass

images4 = list(df_train4.columns.values)
print(len(images4))
for x in images4:

    try:
        paths.append(get_paths4(x))
        y.append(to_categorical(np.full((max_frames), LABELS.index(df_train4[x]['label'])), num_classes=2))
    except Exception as err:
        # print(err)
        pass

print(np.shape(paths))
print(np.shape(y))

X=[]
for img in tqdm(paths):
    X.append(read_img(img))

print(np.shape(X))

dataset_size = len(X)

tf_img = lambda i: ToTensor()(i).unsqueeze(0)
embeddings = lambda input: resnet(input)

X_embedded = []
with torch.no_grad():
    for faces in tqdm(X):
        vid_embs = []
        for i in range(max_frames):
            t = tf_img(faces[i]).to(device)
            e = embeddings(t).squeeze().cpu().tolist()
            vid_embs.append(e)
        X_embedded.append(vid_embs)


print(np.shape(X_embedded))


def lstm():
    """Build a simple LSTM network. On the training sample"""
    # Model.
    model = Sequential()
    model.add(LSTM(2048, return_sequences=True, input_shape=(max_frames, 512) ,dropout=0.5))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

model = lstm()

optimizer = Adam(lr=1e-5/5, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])

print(model.summary())

X_embedded = np.reshape(X_embedded, (dataset_size, max_frames, 512))
y = np.reshape(y, (dataset_size, max_frames, 2))
history = model.fit(X_embedded, y, epochs=100, batch_size=64, validation_split=0.2, shuffle=True)

# print(history)

plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/home/aelbakry1999/Results/accuracy_loss.png')
