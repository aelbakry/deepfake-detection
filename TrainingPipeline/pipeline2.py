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
import time



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import torch.nn as nn
# import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")



max_frames = 5
max_df = 3
df_train0 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_0/metadata.json')
df_train1 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_1/metadata.json')
df_train2 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_2/metadata.json')
df_train3 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_3/metadata.json')
df_train4 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_4/metadata.json')
# df_train5 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_5/metadata.json')
# df_train6 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_6/metadata.json')
# df_train7 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_7/metadata.json')
# df_train8 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_8/metadata.json')
# df_train9 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_9/metadata.json')


# df_test1 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_2/metadata.json')
# df_test1 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_3/metadata.json')


# df_train_all = [df_train0, df_train1, df_train2, df_train3, df_train4,
#                         df_train5, df_train6, df_train7, df_train8, df_train9][:max_df]

df_train_all = [df_train0, df_train1, df_train2]
df_test_all = [df_train3, df_train4]


LABELS = ['REAL','FAKE']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

"""function to flip video frames from given paths for data Augmentation """
def flip_horizontal(path):
    frames_flipped = []
    for i in range(max_frames):
        frames_flipped.append(cv2.flip(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB), 1))
    return frames_flipped

def Cloning(y):
    y_copy = y[:]
    return y_copy

"""function to read video frames from given paths"""
def read_img(path):
    frames = []
    for i in range(max_frames):
        frames.append(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB))
    return frames



def load_data(index, df_train, split):
    paths=[]
    y=[]

    df_train_values = list(df_train.columns.values)


    for value in df_train_values:
        image_paths=[]

        try:
            for num in range(max_frames):
                path = '/home/aelbakry1999/images/margin_0/dfdc_train_part_' + str(index) +"/"+ value.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
                image_paths.append(path)
                if not os.path.exists(path):
                    # print(path)
                    raise Exception

            paths.append(image_paths)
            y.append(LABELS.index(df_train[value]['label']))

        except Exception as err:
            # print(err)
                pass


    return paths, y



"""function to embed video frames with InceptionResnetV1"""
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


paths=[]
y=[]

"""Loading all paths and y_labels in df_train_all """
print("Loading training paths and y values from JSON files")
for index in tqdm(range(np.shape(df_train_all)[0])):
    path, labels = load_data(index, df_train_all[index], "training")
    paths.extend(path)
    y.extend(labels)

paths_test=[]
y_test=[]

print("Loading testing paths and y values from JSON files")
for index in tqdm(range(np.shape(df_test_all)[0])):
    path, labels = load_data(index+np.shape(df_train_all)[0], df_test_all[index], "testing")
    paths_test.extend(path)
    y_test.extend(labels)


# paths = np.array(paths)
# y = np.array(y)


paths_test = np.array(paths_test)
y_test = np.array(y_test)

y = to_categorical(y, num_classes=2) #convert y training to one hot encodings
y_test = to_categorical(y_test, num_classes=2) #convert y testing to one hot encodings


# print("X_paths", paths.shape)
# print("y", y.shape)
#
print("X_test_paths", paths_test.shape)
print("y_test", y_test.shape)


X=[]
print("Loading frames original")
for img in tqdm(paths):
    X.append(read_img(img))

print("Loading frames flipped")
for img in tqdm(paths):
    X.append(flip_horizontal(img))
    
y_clone = Cloning(y)
y = y + y_clone #replicating targets after flipping


print("X training", np.shape(X))
print("Y training", np.shape(y))



train_size = len(X)


X_test=[]
for img in tqdm(paths_test):
    X_test.append(read_img(img))

test_size = len(X_test)

start = time.time()

X_embedded = embed(X)

print(
    f'Time on multiple GPUs: {(time.time() - start):.3f}'
)

X_embedded = np.reshape(X_embedded, (train_size, max_frames, 512))
y = np.reshape(y, (train_size, 2))


X_test_embedded = embed(X_test)
X_test_embedded = np.reshape(X_test_embedded, (test_size, max_frames, 512))


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

optimizer = Adam(lr=1e-5*5, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())


history = model.fit(X_embedded, y, epochs=5, batch_size=64, shuffle=True)

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
