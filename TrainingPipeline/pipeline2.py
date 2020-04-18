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
max_df = 3
df_train0 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_0/metadata.json')
df_train1 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_1/metadata.json')
df_train2 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_2/metadata.json')
df_train3 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_3/metadata.json')
df_train4 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_4/metadata.json')
df_train5 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_5/metadata.json')
df_train6 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_6/metadata.json')
df_train7 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_7/metadata.json')
df_train8 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_8/metadata.json')
df_train9 = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_9/metadata.json')


df_test = pd.read_json('/home/aelbakry1999/dfdc/dfdc_train_part_10/metadata.json')

df_train_all = [df_train0, df_train1, df_train2, df_train3, df_train4,
                        df_train5, df_train6, df_train7, df_train8, df_train9][:max_df]


LABELS = ['REAL','FAKE']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""function to read video frames from given paths"""
def read_img(path):
    frames = []
    for i in range(max_frames):
        frames.append(cv2.cvtColor(cv2.imread(path[i]),cv2.COLOR_BGR2RGB))
    return frames


def load_data(index, df_train):
    paths=[]
    y=[]
    df_train_values= list(df_train.columns.values)

    for value in df_train_values:
        image_paths=[]

        try:

            for num in range(max_frames):
                path = '/home/aelbakry1999/images/dfdc_train_part_' + str(index) +"/"+ value.replace('.mp4', '') + '/frame' + str(num) +'.jpeg'
                image_paths.append(path)
                if not os.path.exists(path):
                    # print(path)
                    raise Exception

            paths.append(image_paths)
            y.append(LABELS.index(df_train[value]['label']))

        except Exception as err:
            # print(err)
                pass

    shape = np.shape(y)

    return paths, y, shape

paths=[]
y=[]

"""Loading all paths and y_labels in df_train_all """
print("Loading paths and y values from JSON files")
for index in tqdm(range(np.shape(df_train_all)[0])):
    path, labels, shape = load_data(index, df_train_all[index])
    paths.extend(path)
    y.extend(labels)

paths = np.array(paths)
y = np.array(y)


print(paths.shape)
print(y.shape)
