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
df_train6 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_5/metadata.json')
df_train7 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_6/metadata.json')
df_train8 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_7/metadata.json')
df_train9 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_8/metadata.json')
df_train10 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_9/metadata.json')


df_test = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_10/metadata.json')

# youtube_faces = pd.read_json('/home/aelbakry1999/YouTubeFaces/aligned_images_DB')



df_train_list = [df_train1 , df_train2]
df_train = df_train2

LABELS = ['REAL','FAKE']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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

def get_paths6(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_5/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths7(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_6/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths8(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_7/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths9(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_8/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

def get_paths10(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_9/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

"""Method to get paths for VoxCeleb2 Dataset"""
def get_paths_vox(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/VoxCeleb2/'+ x + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

"""Method to get paths for testing Dataset"""
def get_paths_test(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/dfdc_train_part_10/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            # print(path)
            raise Exception
    return image_paths

"""function to read video frames from given paths"""
def read_img(path):
    frames = []
    for i in range(max_frames):
        frames.append(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB))
    return frames


paths=[]
y=[]

images1 = list(df_train1.columns.values)
print("Images 1" , len(images1))
for x in images1:

    try:
        paths.append(get_paths(x))
        y.append(LABELS.index(df_train1[x]['label']))
    except Exception as err:
        # print(err)
        pass

images2 = list(df_train2.columns.values)
print("Images 2" ,len(images2))
for x in images2:

    try:
        paths.append(get_paths2(x))
        y.append(LABELS.index(df_train2[x]['label']))
    except Exception as err:
        # print(err)
        pass

images3 = list(df_train3.columns.values)
print("Images 3" ,len(images3))
for x in images3:

    try:
        paths.append(get_paths3(x))
        y.append( LABELS.index(df_train3[x]['label']))
    except Exception as err:
        # print(err)
        pass

images4 = list(df_train4.columns.values)
print("Images 4" ,len(images4))
for x in images4:

    try:
        paths.append(get_paths4(x))
        y.append(LABELS.index(df_train4[x]['label']))
    except Exception as err:
        # print(err)
        pass

images5 = list(df_train5.columns.values)
print("Images 5" ,len(images5))
for x in images5:

    try:
        paths.append(get_paths5(x))
        y.append(LABELS.index(df_train5[x]['label']))
    except Exception as err:
        # print(err)
        pass

images6 = list(df_train6.columns.values)
print("Images 6" ,len(images6))
for x in images6:

    try:
        paths.append(get_paths6(x))
        y.append(LABELS.index(df_train6[x]['label']))
    except Exception as err:
        # print(err)
        pass

images7 = list(df_train7.columns.values)
print("Images 7" ,len(images7))
for x in images7:

    try:
        paths.append(get_paths7(x))
        y.append(LABELS.index(df_train7[x]['label']))
    except Exception as err:
        # print(err)
        pass

images8 = list(df_train8.columns.values)
print("Images 8" ,len(images8))
for x in images8:

    try:
        paths.append(get_paths8(x))
        y.append(LABELS.index(df_train8[x]['label']))
    except Exception as err:
        # print(err)
        pass


images9 = list(df_train8.columns.values)
print(len(images9))
for x in images9:

    try:
        paths.append(get_paths9(x))
        y.append(LABELS.index(df_train9[x]['label']))
    except Exception as err:
        # print(err)
        pass

images10 = list(df_train10.columns.values)
print(len(images10))
for x in images10:

    try:
        paths.append(get_paths10(x))
        y.append(LABELS.index(df_train10[x]['label']))
    except Exception as err:
        # print(err)
        pass

images_vox = os.listdir('/home/aelbakry1999/images/VoxCeleb2/')[:2000]
paths_vox = []
y_vox = []
print("Images Vox ", len(images_vox))
for x in images_vox:

    try:
        paths_vox.append(get_paths_vox(x))
        y_vox.append(0)
    except Exception as err:
        # print(err)
        pass
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


#balance with YouTube faces dataset
# youtube_faces = sorted(os.listdir('/home/aelbakry1999/YouTubeFaces/aligned_images_DB'))
# youtube_faces_path = '/home/aelbakry1999/YouTubeFaces/aligned_images_DB'
# X_ = []
# y_ = []
# for dirpaths in tqdm(youtube_faces):
#     dirpath = sorted(os.listdir(os.path.join(youtube_faces_path, dirpaths)))
#     for subdirpaths in dirpath:
#         subdirpath = sorted(os.listdir(os.path.join(youtube_faces_path, dirpaths, subdirpaths)))[:max_frames]
#         frames_path = []
#         for filename in subdirpath:
#             path = os.path.join(youtube_faces_path, dirpaths, subdirpaths, filename)
#             frames_path.append(cv2.resize(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB), (160, 160)))
#         y_.append(0)
#         X_.append(frames_path)



# print("X_", np.shape(X_))
# print("y_", np.shape(y_))

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

def embed(frames):
    faces_embedded = []
    tf_img = lambda i: ToTensor()(i).unsqueeze(0)
    embeddings = lambda input: resnet(input)

    with torch.no_grad():
        for faces in tqdm(frames):
            vid_embs = []
            for i in range(max_frames):
                t = tf_img(faces[i]).to(device)
                e = embeddings(t).squeeze().cpu().tolist()
                vid_embs.append(e)
            faces_embedded.append(vid_embs)

    return  faces_embedded

print("y before balancing", np.shape(y))
y = y + y_vox
print("y after balancing", np.shape(y))

y = to_categorical(y, num_classes=2) #convert y training to one hot encodings
y_test = to_categorical(y_test, num_classes=2) #convert y testing to one hot encodings
# print(np.shape(paths))
# print("y after balancing", np.shape(y))

print("Paths test shape" ,np.shape(paths_test))
print("y test shape" ,np.shape(y_test))

X=[]
for img in tqdm(paths):
    X.append(read_img(img))

X_vox=[]
for img in tqdm(paths_vox):
    X_vox.append(read_img(img))


print("X before balancing", np.shape(X))
X = X + X_vox
print("X after balancing", np.shape(X))
dataset_size = len(X)


X_test=[]
for img in tqdm(paths_test):
    X_test.append(read_img(img))

print("X_test shape", np.shape(X_test))


X_embedded = embed(X)
X_test_embedded = embed(X_test)


print("X_embedded shape", np.shape(X_embedded))
print("X_test_embedded shape",np.shape(X_test_embedded))

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

optimizer = Adam(lr=1e-5*7, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())



X_embedded = np.reshape(X_embedded, (dataset_size, max_frames, 512))
X_test_embedded = np.reshape(X_test_embedded, (np.shape(X_test_embedded)[0], max_frames, 512))

y = np.reshape(y, (dataset_size, 2))

history = model.fit(X_embedded, y, epochs=20, batch_size=64, shuffle=True, validation_split=0.2)

model.save_weights("model.h5")

y_preds = model.predict_classes(X_test_embedded)


y_test = np.array(y_test).argmax(axis=1) #convert back from one hot encodings
y_preds = np.reshape(y_preds, (np.shape(y_preds)[0], 1))

print("Predictions", y_preds)
print("True Labels", y_test)


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
