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
from sklearn.metrics import confusion_matrix




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
df_train5 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_4/metadata.json')

df_test = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_5/metadata.json')




df_train_list = [df_train1 , df_train2]
df_train = df_train2

LABELS = ['REAL','FAKE']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()



#read respective path for each training folder
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

def get_paths5(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_4/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths_test(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_5/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

#function to read video frames from given paths
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
        y.append(LABELS.index(df_train1[x]['label']))
    except Exception as err:
        # print(err)
        pass

images2 = list(df_train2.columns.values)
print(len(images2))
for x in images2:

    try:
        paths.append(get_paths2(x))
        y.append(LABELS.index(df_train2[x]['label']))
    except Exception as err:
        # print(err)
        pass

images3 = list(df_train3.columns.values)
print(len(images3))
for x in images3:

    try:
        paths.append(get_paths3(x))
        y.append( LABELS.index(df_train3[x]['label']))
    except Exception as err:
        # print(err)
        pass

images4 = list(df_train4.columns.values)
print(len(images4))
for x in images4:

    try:
        paths.append(get_paths4(x))
        y.append(LABELS.index(df_train4[x]['label']))
    except Exception as err:
        # print(err)
        pass

images5 = list(df_train5.columns.values)
print(len(images5))
for x in images5:

    try:
        paths.append(get_paths5(x))
        y.append(LABELS.index(df_train5[x]['label']))
    except Exception as err:
        # print(err)
        pass

y = to_categorical(y, num_classes=2)

paths_test = []
y_test = []

images_test = list(df_test.columns.values)
print(len(images_test))
for x in images_test:

    try:
        paths_test.append(get_paths_test(x))
        y_test.append(LABELS.index(df_test[x]['label']))
    except Exception as err:
        # print(err)
        pass

y_test = to_categorical(y_test, num_classes=2)
print(np.shape(paths))
print(np.shape(y))

print(np.shape(paths_test))
print(np.shape(y_test))

X=[]
for img in tqdm(paths):
    X.append(read_img(img))


print(np.shape(X))

X_test=[]
for img in tqdm(paths_test):
    X_test.append(read_img(img))

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

X_test_embedded = []
with torch.no_grad():
    for faces in tqdm(X_test):
        vid_embs = []
        for i in range(max_frames):
            t = tf_img(faces[i]).to(device)
            e = embeddings(t).squeeze().cpu().tolist()
            vid_embs.append(e)
        X_test_embedded.append(vid_embs)

print(np.shape(X_test_embedded))

def lstm():
    """Build a simple LSTM network. On the training sample"""
    # Model.
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False, input_shape=(max_frames, 512) ,dropout=0.5))
    model.add((Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

model = lstm()

optimizer = Adam(lr=1e-5/5, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])



print(model.summary())

X_embedded = np.reshape(X_embedded, (dataset_size, max_frames, 512))
# y = np.reshape(y, (dataset_size, max_frames, 2))
y = np.reshape(y, (dataset_size, 2))

history = model.fit(X_embedded, y, epochs=20, batch_size=64, shuffle=True)

model.save_weights("model.h5")
# print(history)

y_preds = model.predict_classes(np.reshape(X_test_embedded, (np.shape(X_test_embedded)[0], max_frames, 512)))
# y_preds = np.argmax(y_preds, axis=0)

print(y_preds)
print(np.shape(y_test))

y_test = np.array(y_test).argmax(axis=1)
y_preds = np.reshape(y_preds, (np.shape(y_preds)[0], 1))

print(y_test)
print(y_preds)

conf_matrix = confusion_matrix(y_test, y_preds)

tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

print("-------------- Confusion Matrix -------------- ")
print(conf_matrix)


print('True Positives: {}, True Negatives: {}, False Positives: {}, False Negatives: {}'.format(tp, tn, fp, fn))

precision = tp / (tp + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
f1 = 2*(precision*recall)/ (precision+recall)
print("-------------- Model Scores -------------- ")
print('Precision: {}, Accuracy: {}, Recall: {}, F1-score: {}'.format(precision, accuracy, recall, f1))




plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('/home/aelbakry1999/Results/accuracy_loss.png')
