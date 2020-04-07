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
df_train = pd.read_json('/home/aelbakry1999/sample_data/train_sample_videos/metadata.json')
LABELS = ['REAL','FAKE']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()




def get_paths(x):
    image_paths=[]

    for num in range(max_frames):
        path = '/home/aelbakry1999/images/train_sample_videos/'+ x.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
        image_paths.append(path)
        if not os.path.exists(path):
            print(path)
            raise Exception
    return image_paths

def read_img(path):
    frames = []
    for i in range(max_frames):
        frames.append(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB))
    return frames


paths=[]
y=[]
images = list(df_train.columns.values)
for x in images:

    try:
        paths.append(get_paths(x))
        y.append(to_categorical(np.full((max_frames), LABELS.index(df_train[x]['label'])), num_classes=2))
    except Exception as err:
        # print(err)
        pass


# y = to_categorical(y, num_classes=2)
print(np.shape(paths))
print(np.shape(y))

X=[]
for img in tqdm(paths):
    X.append(read_img(img))

print(np.shape(X))


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
    model.add(LSTM(256, return_sequences=True, input_shape=(10, 512) ,dropout=0.5))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

model = lstm()

optimizer = Adam(lr=1e-5/10, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=optimizer,
                   metrics=['accuracy'])

print(model.summary())

X_embedded = np.reshape(X_embedded, (302, 10, 512))
y = np.reshape(y, (302,10, 2))
history = model.fit(X_embedded, y, epochs=100, batch_size=1, validation_split=0.3)

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
